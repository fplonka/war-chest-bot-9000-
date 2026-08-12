//! The value network, for inference inside the Rust self-play workers.
//!
//! Training happens in PyTorch; the learned weights are pushed here as flat f32
//! row-major `[in, out]` matrices (torch's `weight.t()`). On macOS the matmuls
//! go through Accelerate's BLAS; elsewhere a plain triple loop keeps the crate
//! dependency-free.
//!
//! # Shape
//!
//! The network answers `v(PBS, config)`: the counterfactual value of one
//! player's exact private state at a public belief state. It has two towers.
//!
//! ```text
//!   config tower:  z(c)  = relu(phi(c) Wc + bc)                 [dg]
//!                  g(c)  = z(c) Wg + bg                         [rank + 1]
//!
//!   PBS tower:     hpub  = relu(LN(x_pub W0 + b0)) W1
//!                  e_p   = sum_c beta_p(c) z(c)                 [dg] per player
//!                  h     = relu(LN(hpub + [e_0; e_1] Wb + b1))  [hidden]
//!                  u     = h Wu + bu                            [rank]
//!
//!   value:         v(c)  = <u, g(c)[..rank]> + g(c)[rank]
//! ```
//!
//! This is the reference implementation's shape, generalised. `csrc/liars_dice`
//! ends in `hidden -> num_hands`, which is exactly `<h, W2[:, c]> + b2[c]` — an
//! embedding *table* over private states. War Chest's private states do not fit
//! in a table (hand times face-down runs to ~145k and varies by draft), so the
//! table becomes an embedding *network* `g`. The belief is the same substitution
//! on the input side: instead of a fixed-length vector of per-private-state
//! probabilities, it is the belief-weighted sum of the same config embeddings.
//!
//! `rank` is where the two sides meet, and it is the one dimension that has to
//! be chosen rather than inherited. The reference gets `rank = hidden` for free
//! because its readout is a lookup; here every config costs a `rank`-long dot
//! product, per leaf, per iteration. A config is described by sixteen numbers,
//! so 64 is not a binding constraint on what the value can depend on — and it
//! is 6x less per-config work than tying it to the hidden width.
//!
//! # What is cached, and why the shape is arranged this way
//!
//! Inside a CFR solve the same leaf is queried once per iteration and only the
//! beliefs move. Three things therefore survive the whole solve and are computed
//! once: the public tower `hpub` (the widest matmul in the network), and both
//! `z(c)` and `g(c)` for every config in the tree — a config's features do not
//! depend on the iteration. Per iteration what remains is one `2*dg -> hidden`
//! matmul per leaf, one LayerNorm, and one dot product per config. That is less
//! per-iteration work than the fixed-width output head it replaces.

use crate::board::N_HEXES;
use crate::rebel::{
    AFEAT, AOFF_PAYS, CCOUNTS, CFEAT, CPRIVATE, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_CARDS,
    OFF_LOOSE, OFF_PILES, PILE_COUNTS, PUBFEAT,
};
use crate::units::CARD_FEATS;

#[cfg(target_vendor = "apple")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// `c[m x n] = a[m x k] * b[k x n] + beta * c`, row-major, with explicit
/// leading dimensions so a caller can pass a sub-block of a wider matrix
/// without copying it.
#[allow(clippy::too_many_arguments)]
fn gemm_ld(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    debug_assert!(a.len() >= (m - 1) * lda + k);
    debug_assert!(b.len() >= (k - 1) * ldb + n);
    debug_assert!(c.len() >= (m - 1) * ldc + n);
    #[cfg(target_vendor = "apple")]
    unsafe {
        // 101 = CblasRowMajor, 111 = CblasNoTrans.
        cblas_sgemm(
            101,
            111,
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            beta,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
    // Everywhere else: matrixmultiply, a portable single-threaded GEMM with
    // SIMD kernels. Single-threaded is correct here — the workers already run
    // one game per core, so a threaded BLAS would fight them for cores.
    #[cfg(not(target_vendor = "apple"))]
    // SAFETY: the debug_asserts above are the exact bounds sgemm reads.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            lda as isize,
            1,
            b.as_ptr(),
            ldb as isize,
            1,
            beta,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
    }
}

/// `c[m x n] = a[m x k] * b[n x k]^T`, row-major. The second operand is stored
/// by *row* rather than by column, which is how the readouts hold their action
/// and config embeddings — so this is the shape a dot product of two lists of
/// vectors takes, and it goes through the same coprocessor the other matmuls do.
#[allow(clippy::too_many_arguments)]
fn gemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    debug_assert!(a.len() >= (m - 1) * lda + k);
    debug_assert!(b.len() >= (n - 1) * ldb + k);
    debug_assert!(c.len() >= (m - 1) * ldc + n);
    #[cfg(target_vendor = "apple")]
    unsafe {
        // 101 = CblasRowMajor, 111 = CblasNoTrans, 112 = CblasTrans.
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            0.0,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
    // The transposed operand is a stride swap: B[n x k] row-major read as
    // B^T[k x n] has row stride 1 and column stride ldb.
    #[cfg(not(target_vendor = "apple"))]
    // SAFETY: the debug_asserts above are the exact bounds sgemm reads.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            lda as isize,
            1,
            b.as_ptr(),
            1,
            ldb as isize,
            0.0,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
    }
}

/// Every PBS row against every config embedding at once: `out[r][c]` is
/// `<u[r], g[c][..rk]>`. The bias column is left to the caller, which has to
/// touch each entry anyway to scale by the opponent's reach.
#[allow(clippy::too_many_arguments)]
pub fn dots(
    u: &[f32],
    rk: usize,
    g: &[f32],
    ldg: usize,
    rows: usize,
    ncfg: usize,
    out: &mut [f32],
) {
    gemm_nt(rows, ncfg, rk, u, rk, g, ldg, &mut out[..rows * ncfg], ncfg);
}

/// Must match `torch.nn.LayerNorm`'s default.
const LN_EPS: f32 = 1e-5;

// ------------------------------------------------- LayerNorm kernels
//
// The three passes a LayerNorm needs, hand-vectorised. Written out with
// intrinsics because LLVM would not vectorise the portable version: it unrolled
// the accumulator array into eight *scalar* `fadd`s and left the NEON registers
// idle, which put this pass at five times the cost of the matmuls it wraps.
// Two accumulators per pass keep the 3-cycle FP add latency covered.

/// `row += bias (+ add)`, returning the sum of the result.
#[inline]
fn add_bias_sum(row: &mut [f32], bias: &[f32], add: Option<&[f32]>) -> f32 {
    let n = row.len();
    assert!(bias.len() >= n);
    if let Some(a) = add {
        assert!(a.len() >= n);
    }
    let mut i = 0usize;
    let mut s = 0.0f32;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (p, b) = (row.as_mut_ptr(), bias.as_ptr());
        let a0 = add.map_or(std::ptr::null(), |a| a.as_ptr());
        while i + 8 <= n {
            let mut x0 = vaddq_f32(vld1q_f32(p.add(i)), vld1q_f32(b.add(i)));
            let mut x1 = vaddq_f32(vld1q_f32(p.add(i + 4)), vld1q_f32(b.add(i + 4)));
            if !a0.is_null() {
                x0 = vaddq_f32(x0, vld1q_f32(a0.add(i)));
                x1 = vaddq_f32(x1, vld1q_f32(a0.add(i + 4)));
            }
            vst1q_f32(p.add(i), x0);
            vst1q_f32(p.add(i + 4), x1);
            s0 = vaddq_f32(s0, x0);
            s1 = vaddq_f32(s1, x1);
            i += 8;
        }
        s = vaddvq_f32(vaddq_f32(s0, s1));
    }
    while i < n {
        row[i] += bias[i] + add.map_or(0.0, |a| a[i]);
        s += row[i];
        i += 1;
    }
    s
}

/// Sum of squared deviations from `mean`.
#[inline]
fn sq_dev(row: &[f32], mean: f32) -> f32 {
    let n = row.len();
    let mut i = 0usize;
    let mut v = 0.0f32;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (mut v0, mut v1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let m = vdupq_n_f32(mean);
        let p = row.as_ptr();
        while i + 8 <= n {
            let d0 = vsubq_f32(vld1q_f32(p.add(i)), m);
            let d1 = vsubq_f32(vld1q_f32(p.add(i + 4)), m);
            v0 = vfmaq_f32(v0, d0, d0);
            v1 = vfmaq_f32(v1, d1, d1);
            i += 8;
        }
        v = vaddvq_f32(vaddq_f32(v0, v1));
    }
    while i < n {
        let d = row[i] - mean;
        v += d * d;
        i += 1;
    }
    v
}

/// `row = max(((row - mean) * inv) * g + bt, floor)`.
#[inline]
fn scale_shift(row: &mut [f32], mean: f32, inv: f32, g: &[f32], bt: &[f32], floor: f32) {
    let n = row.len();
    assert!(g.len() >= n && bt.len() >= n);
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (m, sc, lo) = (vdupq_n_f32(mean), vdupq_n_f32(inv), vdupq_n_f32(floor));
        let (p, gp, bp) = (row.as_mut_ptr(), g.as_ptr(), bt.as_ptr());
        while i + 8 <= n {
            let d0 = vmulq_f32(vsubq_f32(vld1q_f32(p.add(i)), m), sc);
            let d1 = vmulq_f32(vsubq_f32(vld1q_f32(p.add(i + 4)), m), sc);
            let y0 = vmaxq_f32(
                vfmaq_f32(vld1q_f32(bp.add(i)), d0, vld1q_f32(gp.add(i))),
                lo,
            );
            let y1 = vmaxq_f32(
                vfmaq_f32(vld1q_f32(bp.add(i + 4)), d1, vld1q_f32(gp.add(i + 4))),
                lo,
            );
            vst1q_f32(p.add(i), y0);
            vst1q_f32(p.add(i + 4), y1);
            i += 8;
        }
    }
    while i < n {
        let x = (row[i] - mean) * inv * g[i] + bt[i];
        row[i] = if x > floor { x } else { floor };
        i += 1;
    }
}

/// `row = max(row, 0)`.
#[inline]
fn relu(row: &mut [f32]) {
    let n = row.len();
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let z = vdupq_n_f32(0.0);
        let p = row.as_mut_ptr();
        while i + 8 <= n {
            vst1q_f32(p.add(i), vmaxq_f32(vld1q_f32(p.add(i)), z));
            vst1q_f32(p.add(i + 4), vmaxq_f32(vld1q_f32(p.add(i + 4)), z));
            i += 8;
        }
    }
    while i < n {
        if row[i] < 0.0 {
            row[i] = 0.0;
        }
        i += 1;
    }
}

/// `sum(a * b)`. Written like the LayerNorm kernels above and for the same
/// reason: this is the per-config readout, so it runs once per config per leaf
/// per CFR iteration, and a serial reduction chain 384 long is the whole cost.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert!(b.len() >= n);
    let mut i = 0usize;
    let mut s = 0.0f32;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (p, q) = (a.as_ptr(), b.as_ptr());
        while i + 8 <= n {
            s0 = vfmaq_f32(s0, vld1q_f32(p.add(i)), vld1q_f32(q.add(i)));
            s1 = vfmaq_f32(s1, vld1q_f32(p.add(i + 4)), vld1q_f32(q.add(i + 4)));
            i += 8;
        }
        s = vaddvq_f32(vaddq_f32(s0, s1));
    }
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// `out = sum_i w[i] * z[idx[i] * n ..][..n]` — the belief embedding.
///
/// One of the two hot loops of a solve: it runs once per leaf per player per
/// CFR iteration over a support averaging ~20 configs, and it is a gather
/// followed by an axpy, which is exactly the shape LLVM leaves scalar. Two
/// accumulator lanes, unrolled by eight, the same treatment the LayerNorm
/// reductions needed.
pub fn accumulate(z: &[f32], idx: &[u32], w: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(idx.len(), w.len());
    debug_assert_eq!(out.len(), n);
    out.fill(0.0);
    for (c, &i) in idx.iter().enumerate() {
        let src = &z[i as usize * n..i as usize * n + n];
        let wc = w[c];
        let mut k = 0usize;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let vw = vdupq_n_f32(wc);
            let (p, q) = (out.as_mut_ptr(), src.as_ptr());
            while k + 8 <= n {
                let a0 = vfmaq_f32(vld1q_f32(p.add(k)), vld1q_f32(q.add(k)), vw);
                let a1 = vfmaq_f32(vld1q_f32(p.add(k + 4)), vld1q_f32(q.add(k + 4)), vw);
                vst1q_f32(p.add(k), a0);
                vst1q_f32(p.add(k + 4), a1);
                k += 8;
            }
        }
        while k < n {
            out[k] += wc * src[k];
            k += 1;
        }
    }
}

/// Grow `v` to at least `n` without ever re-zeroing what is already there.
/// Every buffer here is fully overwritten by the matmul that follows, so the
/// `clear() + resize()` this replaces was a megabyte-scale memset per call.
#[inline]
pub fn fit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// One linear layer: row-major `[i, o]` weights and an `o`-long bias.
///
/// The whole network is chains of these. Depth and width live in the chain
/// lengths and layer shapes, which come from the checkpoint (`dims`), so a
/// deeper or narrower tower is a different checkpoint, never different code.
#[derive(Clone, Default)]
pub struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    pub i: usize,
    pub o: usize,
}

impl Lin {
    /// `out[rows, o] += src[rows, i] . w` (or `=` when `beta` is 0). `lda` is
    /// the row stride of `src`, so a chain can read a sub-block of a wider
    /// matrix without copying it.
    fn gemm(&self, src: &[f32], rows: usize, lda: usize, beta: f32, out: &mut [f32]) {
        gemm_ld(
            rows, self.o, self.i, src, lda, &self.w, self.o, beta, out, self.o,
        );
    }

    fn bias(&self, rows: usize, out: &mut [f32]) {
        for r in 0..rows {
            let row = &mut out[r * self.o..(r + 1) * self.o];
            for (x, b) in row.iter_mut().zip(&self.b) {
                *x += *b;
            }
        }
    }

    fn bias_relu(&self, rows: usize, out: &mut [f32]) {
        for r in 0..rows {
            let row = &mut out[r * self.o..(r + 1) * self.o];
            for (x, b) in row.iter_mut().zip(&self.b) {
                *x += *b;
            }
            relu(row);
        }
    }
}

/// The value network, as tower chains. See the module docs for the shape; the
/// wiring between towers is fixed (it is the game-specific part: the card
/// table, the set sums, the belief sums, the bilinear readout), while every
/// tower's depth and width comes from the checkpoint.
///
/// Towers, in the order the data flows:
///
/// * `card`: `CARD_FEATS -> ... -> de`, ReLU between, linear out, `wid[id]`
///   added. One row per coin type, once per solve.
/// * `pile`: one shared layer over `[4 counts | e]`, summed per player.
/// * `pub_lin` + `pub_ln`: the public trunk. Every layer is
///   `relu(LN(x W + b))`; `pub_out` then projects to the head width with *no*
///   norm — its bias is applied at the head entry, inside `ln1`. That split
///   is what lets a solve cache `h0` and re-run only the head per iteration.
/// * `hmlp` + `wu`: the per-iteration head. Entry:
///   `relu(LN1(h0 + [b_me|b_opp] Wb))`, then plain ReLU layers, then the
///   readout `u`.
/// * `slot` + `slot_out`: the holding tower, per coin-type row, rectified and
///   summed over the five slots; `res` blocks refine the sum
///   (`z += B(relu(A z))`); `wg` reads out `[rank + 1]`.
/// * `wq`/`wk`/`wp`: the policy readout's three matrices.
#[derive(Clone, Default)]
pub struct Mlp {
    /// The checkpoint's shape vector, verbatim (v3: see `from_flat_v3`).
    pub dims: Vec<usize>,
    de: usize,
    dg: usize,
    rank: usize,
    head_in: usize,
    card: Vec<Lin>,
    /// Learned per-unit identity embedding `[N_UNITS, de]`.
    wid: Vec<f32>,
    pile: Lin,
    pub_lin: Vec<Lin>,
    pub_ln: Vec<(Vec<f32>, Vec<f32>)>,
    pub_out: Lin,
    /// `[2 * dg, head_in]`, no bias — it feeds a layer that already has one.
    wb: Vec<f32>,
    ln1: (Vec<f32>, Vec<f32>),
    hmlp: Vec<Lin>,
    wu: Lin,
    slot: Vec<Lin>,
    slot_out: Lin,
    res: Vec<(Lin, Lin)>,
    wg: Lin,
    wq: Lin,
    wk: Lin,
    wp: Lin,
}

/// One matrix's place in the flat arrays: `[i, o]` weights at `w`, bias at
/// `b` (biasless matrices use `usize::MAX`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Span {
    pub w: usize,
    pub b: usize,
    pub i: usize,
    pub o: usize,
}

/// Where every v3 matrix lives in the flat arrays — the one description of
/// the blob. `from_flat_v3` slices through it and the GPU service points
/// device GEMMs at it, so the two sides cannot disagree.
#[derive(Clone, Debug, Default)]
pub struct V3Layout {
    pub de: usize,
    pub dg: usize,
    pub rank: usize,
    pub head_in: usize,
    pub head_out: usize,
    pub nres: usize,
    pub card: Vec<Span>,
    /// `[N_UNITS, de]`, weights only.
    pub wid: usize,
    pub pile: Span,
    pub pub_lin: Vec<Span>,
    /// Per public layer: (gain, bias) offsets into the ln array.
    pub pub_ln: Vec<(usize, usize)>,
    pub pub_out: Span,
    /// `[2 * dg, head_in]`, weights only.
    pub wb: usize,
    pub ln1: (usize, usize),
    pub hmlp: Vec<Span>,
    pub wu: Span,
    pub slot: Vec<Span>,
    pub slot_out: Span,
    pub res: Vec<(Span, Span)>,
    pub wg: Span,
    pub wq: Span,
    pub wk: Span,
    pub wp: Span,
    pub w_len: usize,
    pub b_len: usize,
    pub ln_len: usize,
}

impl V3Layout {
    /// Walk the v3 dims (see `from_flat_v3`) into spans.
    pub fn new(dims: &[usize]) -> Result<V3Layout, String> {
        if dims.first() != Some(&3) {
            return Err(format!("not a v3 dims vector: {dims:?}"));
        }
        let mut at = 1;
        let mut scalar = |name: &str| -> Result<usize, String> {
            let v = *dims.get(at).ok_or(format!("dims truncated at {name}"))?;
            at += 1;
            Ok(v)
        };
        let (de, dg, rank, head_in, nres) = (
            scalar("de")?,
            scalar("dg")?,
            scalar("rank")?,
            scalar("head_in")?,
            scalar("nres")?,
        );
        let mut list = |name: &str| -> Result<Vec<usize>, String> {
            let n = *dims.get(at).ok_or(format!("dims truncated at {name}"))?;
            at += 1;
            if at + n > dims.len() {
                return Err(format!("dims truncated inside {name}"));
            }
            let v = dims[at..at + n].to_vec();
            at += n;
            Ok(v)
        };
        let (card_w, pub_w, hmlp_w, slot_w) =
            (list("card")?, list("pub")?, list("hmlp")?, list("slot")?);
        if at != dims.len() {
            return Err(format!("dims has {} trailing entries", dims.len() - at));
        }
        if pub_w.is_empty() {
            return Err("the public tower needs at least one layer".into());
        }
        let mut l = V3Layout {
            de,
            dg,
            rank,
            head_in,
            nres,
            head_out: hmlp_w.last().copied().unwrap_or(head_in),
            ..Default::default()
        };
        let (mut w, mut b, mut ln) = (0usize, 0usize, 0usize);
        let mut lin = |w: &mut usize, b: &mut usize, i: usize, o: usize| {
            let s = Span { w: *w, b: *b, i, o };
            *w += i * o;
            *b += o;
            s
        };
        let chain = |w: &mut usize,
                     b: &mut usize,
                     first: usize,
                     mid: &[usize],
                     last: usize,
                     lin: &mut dyn FnMut(&mut usize, &mut usize, usize, usize) -> Span|
         -> Vec<Span> {
            let mut v = Vec::new();
            let mut prev = first;
            for &h in mid.iter().chain(std::iter::once(&last)) {
                v.push(lin(w, b, prev, h));
                prev = h;
            }
            v
        };
        l.card = chain(&mut w, &mut b, CARD_FEATS, &card_w, de, &mut lin);
        l.wid = w;
        w += crate::units::N_UNITS * de;
        l.pile = lin(&mut w, &mut b, PILE_COUNTS + de, de);
        let mut prev = xdim_of(de);
        for &h in &pub_w {
            l.pub_lin.push(lin(&mut w, &mut b, prev, h));
            prev = h;
        }
        l.pub_out = lin(&mut w, &mut b, prev, head_in);
        l.wb = w;
        w += 2 * dg * head_in;
        let mut prev = head_in;
        for &h in &hmlp_w {
            l.hmlp.push(lin(&mut w, &mut b, prev, h));
            prev = h;
        }
        l.wu = lin(&mut w, &mut b, prev, rank);
        let mut prev = hfeat(de);
        for &h in &slot_w {
            l.slot.push(lin(&mut w, &mut b, prev, h));
            prev = h;
        }
        l.slot_out = lin(&mut w, &mut b, prev, dg);
        for _ in 0..nres {
            let a = lin(&mut w, &mut b, dg, dg);
            let bb = lin(&mut w, &mut b, dg, dg);
            l.res.push((a, bb));
        }
        l.wg = lin(&mut w, &mut b, dg, rank + 1);
        l.wq = lin(&mut w, &mut b, AFEAT + de, rank);
        l.wk = lin(&mut w, &mut b, dg, rank);
        l.wp = lin(&mut w, &mut b, l.head_out, rank);
        for &h in &pub_w {
            l.pub_ln.push((ln, ln + h));
            ln += 2 * h;
        }
        l.ln1 = (ln, ln + head_in);
        ln += 2 * head_in;
        l.w_len = w;
        l.b_len = b;
        l.ln_len = ln;
        Ok(l)
    }

    /// The trunk input width this layout assembles.
    pub fn xdim(&self) -> usize {
        xdim_of(self.de)
    }
    /// The holding tower's per-slot input width.
    pub fn hfeat(&self) -> usize {
        hfeat(self.de)
    }
    /// Tower widths: (pub, hmlp, card, slot) layer output widths.
    pub fn widths(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
        let o = |v: &[Span]| v.iter().map(|s| s.o).collect();
        (
            o(&self.pub_lin),
            o(&self.hmlp),
            o(&self.card),
            o(&self.slot),
        )
    }
}

impl Mlp {
    /// Build from the flat arrays the trainer ships.
    pub fn from_flat(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Result<Mlp, String> {
        Mlp::from_flat_v3(dims, w, b, ln)
    }

    /// The v3 tower format:
    ///
    /// ```text
    /// dims = [3, de, dg, rank, head_in, nres,
    ///         |card|, card widths...,     // hidden widths, CARD_FEATS -> .. -> de
    ///         |pub|,  pub widths...,      // LN+ReLU layers, xdim -> .. (>= 1)
    ///         |hmlp|, hmlp widths...,     // extra ReLU head layers (may be 0)
    ///         |slot|, slot widths...]     // hidden widths, hfeat -> .. -> dg
    /// ```
    ///
    /// Weight blob order (each matrix row-major `[in, out]`): card layers,
    /// wid, pile, pub layers, pub_out, wb, hmlp layers, wu, slot layers,
    /// slot_out, res pairs, wg, wq, wk, wp. Biases in the same order (wid and
    /// wb have none). LayerNorms: one (gain, bias) pair per pub layer, then
    /// ln1. `train/value_net.py::flat` writes this and nothing else knows it.
    fn from_flat_v3(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Result<Mlp, String> {
        let l = V3Layout::new(dims)?;
        if w.len() != l.w_len || b.len() != l.b_len || ln.len() != l.ln_len {
            return Err(format!(
                "weight sizes {}/{}/{} do not match dims {dims:?} (want {}/{}/{})",
                w.len(),
                b.len(),
                ln.len(),
                l.w_len,
                l.b_len,
                l.ln_len
            ));
        }
        let lin = |s: &Span| Lin {
            w: w[s.w..s.w + s.i * s.o].to_vec(),
            b: b[s.b..s.b + s.o].to_vec(),
            i: s.i,
            o: s.o,
        };
        let norm =
            |(g, bt): (usize, usize), n: usize| (ln[g..g + n].to_vec(), ln[bt..bt + n].to_vec());
        Ok(Mlp {
            dims: dims.to_vec(),
            de: l.de,
            dg: l.dg,
            rank: l.rank,
            head_in: l.head_in,
            card: l.card.iter().map(&lin).collect(),
            wid: w[l.wid..l.wid + crate::units::N_UNITS * l.de].to_vec(),
            pile: lin(&l.pile),
            pub_lin: l.pub_lin.iter().map(&lin).collect(),
            pub_ln: l
                .pub_lin
                .iter()
                .zip(&l.pub_ln)
                .map(|(s, &o)| norm(o, s.o))
                .collect(),
            pub_out: lin(&l.pub_out),
            wb: w[l.wb..l.wb + 2 * l.dg * l.head_in].to_vec(),
            ln1: norm(l.ln1, l.head_in),
            hmlp: l.hmlp.iter().map(&lin).collect(),
            wu: lin(&l.wu),
            slot: l.slot.iter().map(&lin).collect(),
            slot_out: lin(&l.slot_out),
            res: l.res.iter().map(|(a, bb)| (lin(a), lin(bb))).collect(),
            wg: lin(&l.wg),
            wq: lin(&l.wq),
            wk: lin(&l.wk),
            wp: lin(&l.wp),
        })
    }

    /// Read the flat weight dump `train/export_weights.py` writes:
    ///
    /// ```text
    /// u32 encoding_version,
    /// u32 n_dims, n_dims * u32 dims,
    /// u32 n_w, n_w * f32,   u32 n_b, n_b * f32,   u32 n_ln, n_ln * f32
    /// ```
    pub fn load_flat_bin(
        path: &str,
    ) -> std::io::Result<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let raw = std::fs::read(path)?;
        let mut at = 0usize;
        let u32_at = |b: &[u8], at: &mut usize| -> usize {
            let v = u32::from_le_bytes(b[*at..*at + 4].try_into().unwrap()) as usize;
            *at += 4;
            v
        };
        let f32s_at = |b: &[u8], at: &mut usize| -> Vec<f32> {
            let n = u32_at(b, at);
            let v = (0..n)
                .map(|i| f32::from_le_bytes(b[*at + i * 4..*at + i * 4 + 4].try_into().unwrap()))
                .collect();
            *at += n * 4;
            v
        };
        let version = u32_at(&raw, &mut at) as u32;
        if version != crate::rebel::ENCODING_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weight encoding version {version}, expected {}",
                    crate::rebel::ENCODING_VERSION
                ),
            ));
        }
        let nd = u32_at(&raw, &mut at);
        let dims: Vec<usize> = (0..nd).map(|_| u32_at(&raw, &mut at)).collect();
        let (w, b, ln) = (
            f32s_at(&raw, &mut at),
            f32s_at(&raw, &mut at),
            f32s_at(&raw, &mut at),
        );
        Ok((dims, w, b, ln))
    }

    pub fn load_bin(path: &str) -> std::io::Result<Mlp> {
        let (dims, w, b, ln) = Self::load_flat_bin(path)?;
        Mlp::from_flat(&dims, &w, &b, &ln)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }
    /// Width of `h0` and of the head entry (`ln1`).
    pub fn head(&self) -> usize {
        self.head_in
    }
    /// Width the readouts consume: the last head layer's output.
    pub fn head_out(&self) -> usize {
        self.hmlp.last().map_or(self.head_in, |l| l.o)
    }
    /// Width of one config vector.
    pub fn cfeat(&self) -> usize {
        CFEAT
    }
    /// Width of a config embedding, and of one player's belief block.
    pub fn dg(&self) -> usize {
        self.dg
    }
    /// Width of the value readout's inner product.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Width of both players' belief blocks together.
    pub fn belief_dim(&self) -> usize {
        2 * self.dg
    }
    /// Width of one stored action vector, before the paying card's embedding.
    pub fn afeat(&self) -> usize {
        AFEAT
    }
    /// Width of a card embedding.
    pub fn de(&self) -> usize {
        self.de
    }
    /// Width of the trunk's input, once the card embeddings are spliced in.
    pub fn xdim(&self) -> usize {
        xdim_of(self.de)
    }
    /// The tower shapes, for anything (the GPU service) that mirrors the
    /// chains: (card, pub widths, pub_out, hmlp widths, slot widths, nres).
    #[allow(clippy::type_complexity)]
    pub fn towers(&self) -> (Vec<usize>, Vec<usize>, usize, Vec<usize>, Vec<usize>, usize) {
        (
            self.card.iter().map(|l| l.o).collect(),
            self.pub_lin.iter().map(|l| l.o).collect(),
            self.head_in,
            self.hmlp.iter().map(|l| l.o).collect(),
            self.slot.iter().map(|l| l.o).collect(),
            self.res.len(),
        )
    }

    /// LayerNorm + ReLU over `rows x n`, in place, with an optional cached
    /// addend folded into the bias pass. Hand-vectorised: see the kernels at
    /// the top of this file for why.
    fn ln_relu(
        &self,
        rows: usize,
        n: usize,
        bias: &[f32],
        g: &[f32],
        bt: &[f32],
        add: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let inv_n = 1.0 / n as f32;
        for r in 0..rows {
            let row = &mut out[r * n..r * n + n];
            let sum = add_bias_sum(row, bias, add.map(|a| &a[r * n..r * n + n]));
            // Biased variance (divide by n, not n-1) to match torch.
            let mean = sum * inv_n;
            let var = sq_dev(row, mean) * inv_n;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            scale_shift(row, mean, inv, g, bt, 0.0);
        }
    }

    /// The card table `e`, `[NTYPE, de]` — one embedding per coin type,
    /// `relu` chain over the rulebook facts plus the learned id embedding.
    /// Runs once per solve; every other tower reads rows of the result.
    pub fn cards(&self, xpub_row: &[f32], ids: &[u8], e: &mut Vec<f32>) {
        let de = self.de;
        let cards = &xpub_row[OFF_CARDS..OFF_CARDS + NTYPE * CARD_FEATS];
        let mut cur: Vec<f32> = cards.to_vec();
        let mut nxt = Vec::new();
        let last = self.card.len() - 1;
        for l in &self.card[..last] {
            fit(&mut nxt, NTYPE * l.o);
            l.gemm(&cur, NTYPE, l.i, 0.0, &mut nxt[..NTYPE * l.o]);
            l.bias_relu(NTYPE, &mut nxt);
            std::mem::swap(&mut cur, &mut nxt);
        }
        let l = &self.card[last];
        fit(e, NTYPE * de);
        l.gemm(&cur, NTYPE, l.i, 0.0, &mut e[..NTYPE * de]);
        l.bias(NTYPE, e);
        for t in 0..NTYPE {
            let id = ids[t] as usize;
            for j in 0..de {
                e[t * de + j] += self.wid[id * de + j];
            }
        }
    }

    /// The trunk's input, assembled from a stored row and the card table:
    /// the raw hex facts, then each hex's occupant embedding, then the pile
    /// summary, then the loose scalars. See `train/value_net.py::trunk_input`
    /// for the layout rationale; the two must agree block for block.
    fn assemble(&self, xpub: &[f32], rows: usize, stride: usize, e: &[f32], x: &mut Vec<f32>) {
        let (de, xd) = (self.de, self.xdim());
        if rows == 0 {
            return;
        }
        let (hex_e, piles) = (N_HEXES * HEX_FACTS, N_HEXES * (HEX_FACTS + de));
        fit(x, rows * xd);
        x[..rows * xd].fill(0.0);
        // The card half of the pile summary is the same at every row, so it
        // folds into the bias once; only the four counts move per row.
        let mut pe = vec![0.0f32; NTYPE * de];
        for t in 0..NTYPE {
            let out = &mut pe[t * de..(t + 1) * de];
            out.copy_from_slice(&self.pile.b);
            for i in 0..de {
                let v = e[t * de + i];
                let w = &self.pile.w[(PILE_COUNTS + i) * de..(PILE_COUNTS + i + 1) * de];
                for j in 0..de {
                    out[j] += v * w[j];
                }
            }
        }
        let mut cnt = vec![0.0f32; rows * NTYPE * PILE_COUNTS];
        let step = NTYPE * PILE_COUNTS;
        for r in 0..rows {
            cnt[r * step..(r + 1) * step]
                .copy_from_slice(&xpub[r * stride + OFF_PILES..r * stride + OFF_PILES + step]);
        }
        let mut ph = vec![0.0f32; rows * NTYPE * de];
        gemm_ld(
            rows * NTYPE,
            de,
            PILE_COUNTS,
            &cnt,
            PILE_COUNTS,
            &self.pile.w,
            de,
            0.0,
            &mut ph,
            de,
        );
        for r in 0..rows {
            let src = &xpub[r * stride..r * stride + PUBFEAT];
            let dst = &mut x[r * xd..(r + 1) * xd];
            for h in 0..N_HEXES {
                let hx = &src[h * HEX_CH..(h + 1) * HEX_CH];
                dst[h * HEX_FACTS..(h + 1) * HEX_FACTS].copy_from_slice(&hx[..HEX_FACTS]);
                debug_assert!(hx[HEX_FACTS..].iter().filter(|v| **v != 0.0).count() <= 1);
                if let Some(t) = hx[HEX_FACTS..].iter().position(|&v| v != 0.0) {
                    dst[hex_e + h * de..hex_e + (h + 1) * de]
                        .copy_from_slice(&e[t * de..(t + 1) * de]);
                }
            }
            for t in 0..NTYPE {
                let acc = &mut dst[piles + (t / NSLOT) * de..piles + (t / NSLOT) * de + de];
                let o = &ph[(r * NTYPE + t) * de..(r * NTYPE + t + 1) * de];
                for j in 0..de {
                    acc[j] += (o[j] + pe[t * de + j]).max(0.0);
                }
            }
            dst[piles + 2 * de..].copy_from_slice(&src[OFF_LOOSE..OFF_LOOSE + LOOSE]);
        }
    }

    /// The holding tower over `n` config vectors: per coin-type row through
    /// the slot chain, rectified, summed over the five slots, refined by the
    /// residual blocks. Produces `z` (`[n, dg]`) and the readout `g`
    /// (`[n, rank + 1]`). Runs once per distinct config per solve.
    pub fn embed(&self, phi: &[f32], n: usize, e: &[f32], z: &mut Vec<f32>, g: &mut Vec<f32>) {
        let (rk, dg) = (self.rank, self.dg);
        debug_assert_eq!(phi.len(), n * self.cfeat());
        fit(z, n * dg);
        {
            let (de, hf) = (self.de, hfeat(self.de));
            let cf = self.cfeat();
            // The five slot rows are independent and identically shaped, so
            // the tower is matmuls over [n * NSLOT, .] and a segmented sum.
            let mut inp = vec![0.0f32; n * NSLOT * hf];
            for r in 0..n {
                let p = &phi[r * cf..(r + 1) * cf];
                let seat = p[CPRIVATE];
                for k in 0..NSLOT {
                    let row = &mut inp[(r * NSLOT + k) * hf..(r * NSLOT + k + 1) * hf];
                    row[0] = p[k];
                    row[1] = p[NSLOT + k];
                    row[2] = p[2 * NSLOT + k];
                    // Centred. As a raw 0/1 this channel is inert for seat 0
                    // and active for seat 1, and everything after it is
                    // rectified before it is summed, so the asymmetry cannot
                    // cancel -- it lands in the readout's per-config bias term
                    // as a constant that differs by seat. That is a common-mode
                    // offset in a value that must be antisymmetric. Must match
                    // `value_net.py::holdings`.
                    row[3] = p[CCOUNTS + k];
                    row[4] = p[CCOUNTS + NSLOT + k];
                    row[5] = seat - 0.5;
                    let t = seat as usize * NSLOT + k;
                    row[6..].copy_from_slice(&e[t * de..(t + 1) * de]);
                }
            }
            let mut a = inp;
            let mut b = Vec::new();
            for l in &self.slot {
                fit(&mut b, n * NSLOT * l.o);
                l.gemm(&a, n * NSLOT, l.i, 0.0, &mut b[..n * NSLOT * l.o]);
                l.bias_relu(n * NSLOT, &mut b);
                std::mem::swap(&mut a, &mut b);
            }
            let mut slot = vec![0.0f32; n * NSLOT * dg];
            self.slot_out
                .gemm(&a, n * NSLOT, self.slot_out.i, 0.0, &mut slot);
            for r in 0..n {
                let out = &mut z[r * dg..(r + 1) * dg];
                out.fill(0.0);
                for k in 0..NSLOT {
                    let o = &slot[(r * NSLOT + k) * dg..(r * NSLOT + k + 1) * dg];
                    // Rectify before the sum: a sum of raw linear maps forgets
                    // which count belongs to which card.
                    for j in 0..dg {
                        out[j] += (o[j] + self.slot_out.b[j]).max(0.0);
                    }
                }
            }
        }
        // Residual blocks: z += B(relu(A z)). A fresh checkpoint zeroes each
        // B, so training starts from the additive tower.
        let mut r1 = Vec::new();
        for (a, bb) in &self.res {
            fit(&mut r1, n * dg);
            a.gemm(&z[..n * dg], n, dg, 0.0, &mut r1[..n * dg]);
            a.bias_relu(n, &mut r1);
            bb.gemm(&r1[..n * dg], n, dg, 1.0, &mut z[..n * dg]);
            bb.bias(n, z);
        }
        fit(g, n * (rk + 1));
        self.wg
            .gemm(&z[..n * dg], n, dg, 0.0, &mut g[..n * (rk + 1)]);
        self.wg.bias(n, g);
    }

    /// The public tower: every layer `relu(LN(x W + b))`, then `pub_out`
    /// projects to the head width. The result is `h0`, pre-norm — `pub_out`'s
    /// bias and `ln1` are applied at the head entry, per iteration. Computed
    /// once per leaf per solve.
    pub fn trunk(
        &self,
        xpub: &[f32],
        rows: usize,
        stride: usize,
        e: &[f32],
        scratch: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        let mut x = Vec::new();
        self.assemble(xpub, rows, stride, e, &mut x);
        let (src, lda) = (&x[..], self.xdim());
        let l0 = &self.pub_lin[0];
        let mut a = std::mem::take(scratch);
        fit(&mut a, rows * l0.o);
        l0.gemm(src, rows, lda, 0.0, &mut a[..rows * l0.o]);
        self.ln_relu(
            rows,
            l0.o,
            &l0.b,
            &self.pub_ln[0].0,
            &self.pub_ln[0].1,
            None,
            &mut a,
        );
        let mut b = Vec::new();
        for (l, ln) in self.pub_lin[1..].iter().zip(&self.pub_ln[1..]) {
            fit(&mut b, rows * l.o);
            l.gemm(&a, rows, l.i, 0.0, &mut b[..rows * l.o]);
            self.ln_relu(rows, l.o, &l.b, &ln.0, &ln.1, None, &mut b);
            std::mem::swap(&mut a, &mut b);
        }
        fit(out, rows * self.head_in);
        self.pub_out.gemm(
            &a,
            rows,
            self.pub_out.i,
            0.0,
            &mut out[..rows * self.head_in],
        );
        *scratch = a;
    }

    /// The PBS side of the value, given the cached trunk in `pre` and the two
    /// belief embeddings in `xbel` (`[rows * 2 * dg]`). Writes `[rows * rank]`.
    /// This is the only part of the network that runs per CFR iteration.
    pub fn pbs_head(
        &self,
        xbel: &[f32],
        rows: usize,
        pre: &[f32],
        scratch: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        self.hidden_layer(xbel, rows, pre, scratch);
        self.readout(scratch, rows, &self.wu, out);
    }

    /// The head: `relu(LN1(h0 + xbel Wb))`, then the extra head layers. Both
    /// readouts start here — the value's runs every CFR iteration, the
    /// policy's once per solve.
    fn hidden_layer(&self, xbel: &[f32], rows: usize, pre: &[f32], out: &mut Vec<f32>) {
        let (hd, bd) = (self.head_in, self.belief_dim());
        debug_assert_eq!(xbel.len(), rows * bd);
        fit(out, rows * hd);
        gemm_ld(
            rows,
            hd,
            bd,
            xbel,
            bd,
            &self.wb,
            hd,
            0.0,
            &mut out[..rows * hd],
            hd,
        );
        self.ln_relu(
            rows,
            hd,
            &self.pub_out.b,
            &self.ln1.0,
            &self.ln1.1,
            Some(pre),
            &mut out[..rows * hd],
        );
        if self.hmlp.is_empty() {
            return;
        }
        let mut a = std::mem::take(out);
        let mut b = Vec::new();
        for l in &self.hmlp {
            fit(&mut b, rows * l.o);
            l.gemm(&a, rows, l.i, 0.0, &mut b[..rows * l.o]);
            l.bias_relu(rows, &mut b);
            std::mem::swap(&mut a, &mut b);
        }
        *out = a;
    }

    /// `hid W + b`, into `[rows * rank]`.
    fn readout(&self, hid: &[f32], rows: usize, w: &Lin, out: &mut Vec<f32>) {
        fit(out, rows * w.o);
        w.gemm(hid, rows, w.i, 0.0, &mut out[..rows * w.o]);
        w.bias(rows, out);
    }

    /// The action tower: `q(a) = relu([psi(a) | e(paying card)] Wq + bq)` for
    /// `na` actions. Cheap and per node, not per config.
    pub fn embed_actions(&self, psi: &[f32], na: usize, e: &[f32], out: &mut Vec<f32>) {
        let (af, rk, de) = (self.afeat(), self.rank, self.de);
        debug_assert_eq!(psi.len(), na * af);
        let mut inp = vec![0.0f32; na * (af + de)];
        for r in 0..na {
            let src = &psi[r * af..(r + 1) * af];
            let dst = &mut inp[r * (af + de)..(r + 1) * (af + de)];
            dst[..af].copy_from_slice(src);
            let pays = &src[AOFF_PAYS..AOFF_PAYS + NTYPE];
            debug_assert!(pays.iter().filter(|v| **v != 0.0).count() <= 1);
            if let Some(t) = pays.iter().position(|&v| v != 0.0) {
                dst[af..].copy_from_slice(&e[t * de..(t + 1) * de]);
            }
        }
        fit(out, na * rk);
        self.wq.gemm(&inp, na, af + de, 0.0, &mut out[..na * rk]);
        self.wq.bias_relu(na, out);
    }

    /// Policy logits for one decision node: `[nc * na]`, row-major by config.
    /// The caller softmaxes and masks legality.
    #[allow(clippy::too_many_arguments)]
    pub fn policy(
        &self,
        xbel: &[f32],
        pre: &[f32],
        z: &[f32],
        cidx: &[u32],
        q: &[f32],
        na: usize,
        scratch: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        let (dg, rk, nc) = (self.dg, self.rank, cidx.len());
        debug_assert_eq!(out.len(), nc * na);
        self.hidden_layer(xbel, 1, pre, scratch);
        let mut upi = Vec::new();
        self.readout(scratch, 1, &self.wp, &mut upi);
        // `k(c) = u_pi + z(c) Wk + bk`, then every logit is a dot of a `k`
        // row with a `q` row — both matmuls.
        let mut zc = vec![0.0f32; nc * dg];
        for (ci, &c) in cidx.iter().enumerate() {
            zc[ci * dg..(ci + 1) * dg].copy_from_slice(&z[c as usize * dg..(c as usize + 1) * dg]);
        }
        let mut k = vec![0.0f32; nc * rk];
        self.wk.gemm(&zc, nc, dg, 0.0, &mut k);
        for ci in 0..nc {
            for j in 0..rk {
                k[ci * rk + j] += self.wk.b[j] + upi[j];
            }
        }
        gemm_nt(nc, na, rk, &k, rk, q, rk, out, na);
    }

    /// The per-config readout: `v = <u, g[..rank]> + g[rank]`, for the configs
    /// `idx` names in a `g` table built by `embed`.
    pub fn values(&self, u: &[f32], g: &[f32], idx: &[u32], out: &mut [f32]) {
        let rk = self.rank;
        debug_assert_eq!(idx.len(), out.len());
        for (o, &i) in out.iter_mut().zip(idx.iter()) {
            let row = &g[i as usize * (rk + 1)..];
            *o = dot(u, &row[..rk]) + row[rk];
        }
    }

    /// One unprojected score per row for the Torch/Rust parity diagnostic.
    /// Search uses `Solver::readout`, which projects both players together.
    pub fn raw_scores(
        &self,
        xpub: &[f32],
        xbel: &[f32],
        phi: &[f32],
        ids: &[u8],
        rows: usize,
    ) -> Vec<f32> {
        let (rk, pd) = (self.rank, PUBFEAT);
        let (mut sb, mut pre, mut e, mut z, mut g, mut u) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        (0..rows)
            .map(|r| {
                self.cards(
                    &xpub[r * pd..(r + 1) * pd],
                    &ids[r * NTYPE..(r + 1) * NTYPE],
                    &mut e,
                );
                self.trunk(&xpub[r * pd..], 1, pd, &e, &mut sb, &mut pre);
                let bd = self.belief_dim();
                self.pbs_head(&xbel[r * bd..(r + 1) * bd], 1, &pre, &mut sb, &mut u);
                self.embed(
                    &phi[r * self.cfeat()..(r + 1) * self.cfeat()],
                    1,
                    &e,
                    &mut z,
                    &mut g,
                );
                dot(&u[..rk], &g[..rk]) + g[rk]
            })
            .collect()
    }
}

/// One coin type's input to the holding tower: its three counts, next/queued
/// forced-play flags, the seat, and its card embedding.
#[allow(non_snake_case)]
/// Named here because the GPU build has to cut the same row.
pub const fn hfeat(de: usize) -> usize {
    6 + de
}

/// Width of the trunk's input, once the card embeddings are spliced in: the raw
/// hex facts, one embedding per hex, the per-player pile summary, the loose
/// scalars.
const fn xdim_of(de: usize) -> usize {
    N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE
}

#[cfg(test)]
mod gemm_tests {
    use super::*;

    /// Scalar reference for both GEMM shapes. The production path is a real
    /// GEMM backend (Accelerate on Apple, matrixmultiply elsewhere); this pins
    /// it to the naive definition on every platform.
    fn reference(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bt: bool,
        beta: f32,
        c: &mut [f32],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut s = if beta == 0.0 { 0.0 } else { c[i * n + j] };
                for p in 0..k {
                    let bv = if bt { b[j * k + p] } else { b[p * n + j] };
                    s += a[i * k + p] * bv;
                }
                c[i * n + j] = s;
            }
        }
    }

    fn filled(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::rng::Rng::new(seed);
        (0..n)
            .map(|_| (rng.next_u64() % 2000) as f32 / 1000.0 - 1.0)
            .collect()
    }

    #[test]
    fn gemm_matches_reference() {
        for &(m, n, k) in &[(1, 1, 1), (3, 5, 7), (17, 64, 33), (40, 96, 128)] {
            let (a, b) = (filled(m * k, 1), filled(k * n, 2));
            let mut c = filled(m * n, 3);
            let mut want = c.clone();
            gemm_ld(m, n, k, &a, k, &b, n, 1.0, &mut c, n);
            reference(m, n, k, &a, &b, false, 1.0, &mut want);
            for (x, y) in c.iter().zip(&want) {
                assert!((x - y).abs() < 1e-3, "gemm_ld {m}x{n}x{k}: {x} vs {y}");
            }

            let bt = filled(n * k, 4);
            let (mut ct, mut wt) = (vec![0.0; m * n], vec![0.0; m * n]);
            gemm_nt(m, n, k, &a, k, &bt, k, &mut ct, n);
            reference(m, n, k, &a, &bt, true, 0.0, &mut wt);
            for (x, y) in ct.iter().zip(&wt) {
                assert!((x - y).abs() < 1e-3, "gemm_nt {m}x{n}x{k}: {x} vs {y}");
            }
        }
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    /// Restate the v3 blob sizes independently of the parser, so a slip in
    /// either shows as a mismatch here rather than as garbage values.
    fn v3_sizes(
        de: usize,
        dg: usize,
        rk: usize,
        hd: usize,
        nres: usize,
        card: &[usize],
        pubw: &[usize],
        hmlp: &[usize],
        slot: &[usize],
    ) -> (usize, usize, usize) {
        let chain = |first: usize, mid: &[usize], last: usize| -> (usize, usize) {
            let mut w = 0;
            let mut b = 0;
            let mut prev = first;
            for &h in mid.iter().chain(std::iter::once(&last)) {
                w += prev * h;
                b += h;
                prev = h;
            }
            (w, b)
        };
        let (cw, cb) = chain(CARD_FEATS, card, de);
        let (pw, pb) = chain(xdim_of(de), &pubw[..pubw.len() - 1], pubw[pubw.len() - 1]);
        let (ow, ob) = (pubw[pubw.len() - 1] * hd, hd);
        let (hw, hb) = chain(hd, hmlp, rk); // hmlp chain ends in wu
        let head_out = hmlp.last().copied().unwrap_or(hd);
        let (sw, sb) = chain(hfeat(de), slot, dg);
        let w = cw
            + crate::units::N_UNITS * de
            + (PILE_COUNTS + de) * de
            + pw
            + ow
            + 2 * dg * hd
            + hw
            + sw
            + nres * 2 * dg * dg
            + dg * (rk + 1)
            + (AFEAT + de) * rk
            + dg * rk
            + head_out * rk;
        let b = cb + de + pb + ob + hb + sb + nres * 2 * dg + (rk + 1) + 3 * rk;
        let ln = 2 * pubw.iter().sum::<usize>() + 2 * hd;
        (w, b, ln)
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect()
    }

    fn dims_of(
        de: usize,
        dg: usize,
        rk: usize,
        hd: usize,
        nres: usize,
        card: &[usize],
        pubw: &[usize],
        hmlp: &[usize],
        slot: &[usize],
    ) -> Vec<usize> {
        let mut d = vec![3, de, dg, rk, hd, nres];
        for list in [card, pubw, hmlp, slot] {
            d.push(list.len());
            d.extend_from_slice(list);
        }
        d
    }

    #[test]
    fn v3_loads_at_any_depth() {
        for (de, dg, rk, hd, nres, card, pubw, hmlp, slot) in [
            (32, 64, 64, 384, 1, vec![64], vec![384], vec![], vec![]), // = v2 default
            (
                16,
                96,
                48,
                128,
                2,
                vec![24, 24],
                vec![192, 96],
                vec![96],
                vec![24],
            ),
            (
                8,
                32,
                16,
                64,
                0,
                vec![],
                vec![64, 64, 64],
                vec![48, 48],
                vec![],
            ),
        ] {
            let (nw, nb, nl) = v3_sizes(de, dg, rk, hd, nres, &card, &pubw, &hmlp, &slot);
            let dims = dims_of(de, dg, rk, hd, nres, &card, &pubw, &hmlp, &slot);
            let net = Mlp::from_flat(&dims, &ramp(nw), &ramp(nb), &ramp(nl))
                .unwrap_or_else(|e| panic!("{dims:?}: {e}"));
            assert_eq!(net.de(), de);
            assert_eq!(net.head(), hd);
            assert_eq!(net.head_out(), hmlp.last().copied().unwrap_or(hd));
            assert_eq!(net.rank(), rk);
            // A forward pass over structured rows must be finite and move.
            let rows = 3;
            let mut xpub = vec![0.0f32; rows * PUBFEAT];
            for (i, x) in xpub.iter_mut().enumerate() {
                *x = ((i % 7) as f32) * 0.1;
            }
            // One-hot occupant blocks, as the encoder writes them.
            for r in 0..rows {
                for h in 0..N_HEXES {
                    let at = r * PUBFEAT + h * HEX_CH + HEX_FACTS;
                    for j in 0..NTYPE {
                        xpub[at + j] = 0.0;
                    }
                    xpub[at + (h + r) % NTYPE] = 1.0;
                }
            }
            let ids: Vec<u8> = (0..rows * NTYPE).map(|i| (i % 19) as u8).collect();
            let xbel = ramp(rows * net.belief_dim());
            let phi = ramp(rows * net.cfeat());
            let out = net.raw_scores(&xpub, &xbel, &phi, &ids, rows);
            assert!(out.iter().all(|v| v.is_finite()), "{dims:?}: {out:?}");
            assert!(
                out.iter().any(|v| v.abs() > 1e-6),
                "{dims:?}: all-zero output"
            );
        }
    }

    #[test]
    fn unversioned_weight_binary_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("warchest-unversioned-{}.bin", std::process::id()));
        let old = crate::rebel::ENCODING_VERSION - 1;
        std::fs::write(&path, old.to_le_bytes()).unwrap();
        let err = Mlp::load_flat_bin(path.to_str().unwrap()).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(err
            .to_string()
            .contains(&format!("weight encoding version {old}")));
    }
}
