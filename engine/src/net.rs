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
    AOFF_PAYS, CCOUNTS, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE, OFF_PILES,
    PILE_COUNTS, PUBFEAT,
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
    #[cfg(not(target_vendor = "apple"))]
    {
        for i in 0..m {
            if beta == 0.0 {
                for j in 0..n {
                    c[i * ldc + j] = 0.0;
                }
            }
            for p in 0..k {
                let av = a[i * lda + p];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * ldc + j] += av * b[p * ldb + j];
                }
            }
        }
    }
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
fn fit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// The value network. See the module docs for the shape; the field names here
/// are the same symbols.
///
/// Every matrix is row-major `[in, out]`. `dims` is `[pub_dim, hidden, cfeat,
/// dg, rank]` and is the single source of truth for every buffer size, so the
/// trainer and the workers agree on the layout from one array.
#[derive(Clone, Default)]
pub struct Mlp {
    /// `[pub_dim, hidden, cfeat, dg, rank]`.
    pub dims: Vec<usize>,
    /// Public tower.
    w0: Vec<f32>,
    b0: Vec<f32>,
    ln0_w: Vec<f32>,
    ln0_b: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    /// `[2 * dg, hidden]`: both players' belief embeddings into the hidden
    /// layer. No bias — it is added to a layer that already has one.
    wb: Vec<f32>,
    /// Config tower: `[cfeat, dg]` and its bias.
    wc: Vec<f32>,
    bc: Vec<f32>,
    /// Readout embedding: `[dg, rank + 1]` and its bias. The trailing column is
    /// the per-config bias term, which is why this is `rank + 1` and not `rank`.
    wg: Vec<f32>,
    bg: Vec<f32>,
    /// The PBS side of the readout: `[hidden, rank]` and its bias.
    wu: Vec<f32>,
    bu: Vec<f32>,
    // ------------------------------------------------------------ policy head
    // Three more matrices, sharing both towers with the value. An action is
    // described rather than indexed (`rebel::write_action_feats`), so this is an
    // embedding network over a node-dependent action list instead of a
    // fixed-width output vector — the same substitution the config tower makes,
    // and for the same reason.
    //
    //   q(a)       = relu(psi(a) Wq + bq)          [rank]
    //   logit(a,c) = <u_pi + k(c), q(a)>
    //
    // where `u_pi = h Wp + bp` comes from the PBS hidden layer the value
    // readout also reads, and `k(c) = z(c) Wk + bk` from the config embedding.
    /// Action tower: `[afeat, rank]` and its bias.
    wq: Vec<f32>,
    bq: Vec<f32>,
    /// The config side of the policy readout: `[dg, rank]` and its bias.
    wk: Vec<f32>,
    bk: Vec<f32>,
    /// The PBS side of the policy readout: `[hidden, rank]` and its bias.
    wp: Vec<f32>,
    bp: Vec<f32>,
    // -------------------------------------------------------- card describer
    // `e(card) = relu(card Wd0 + bd0) Wd1 + bd1`, `[NTYPE, de]`. Runs once per
    // game: the cards in play do not change. Everything else that refers to a
    // card — the hex block, the pile summary, the holding tower, the action
    // tower — refers to it by coin-type index and reads its row out of this
    // table, so nothing anywhere names a *unit*, and a draft the network has
    // never seen is describable rather than an unknown identity code.
    wd0: Vec<f32>,
    bd0: Vec<f32>,
    wd1: Vec<f32>,
    bd1: Vec<f32>,
    /// Pile summary: `[PILE_COUNTS + de, de]` and its bias. Per coin type, its
    /// four public counts alongside its card embedding, summed per player. A sum
    /// has no order, so any draft fits.
    wpile: Vec<f32>,
    bpile: Vec<f32>,
}

impl Mlp {
    /// Build from the flat arrays the trainer ships. `w` is every matrix
    /// concatenated in the order `W0, W1, Wb, Wc, Wg`; `b` is `b0, b1, bc, bg`;
    /// `ln` is `LN0.weight, LN0.bias, LN1.weight, LN1.bias`.
    ///
    /// One function builds the network for both entry points — the pyo3
    /// `set_weights` and the `.bin` loader — so there is nowhere for the two to
    /// drift apart.
    pub fn from_flat(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Result<Mlp, String> {
        // A five-entry `dims` is a checkpoint from before the card describer.
        // It is loadable so the pool can still be played against; see `v1`.
        if dims.len() == 5 {
            return Mlp::from_flat_v1(dims, w, b, ln);
        }
        if dims.len() != 8 {
            return Err(format!(
                "expected 8 dims [pub, hidden, cfeat, dg, rank, afeat, de, dc], got {dims:?}"
            ));
        }
        let (h, dg, rk, de, dc) = (dims[1], dims[3], dims[4], dims[6], dims[7]);
        let (af, hf, xd) = (dims[5] + de, HFEAT_OF(de), xdim_of(de));
        let want_w = CARD_FEATS * dc
            + dc * de
            + (PILE_COUNTS + de) * de
            + xd * h
            + h * h
            + 2 * dg * h
            + hf * dg
            + dg * (rk + 1)
            + h * rk
            + af * rk
            + dg * rk
            + h * rk;
        let want_b = dc + de + de + h + h + dg + (rk + 1) + 4 * rk;
        let want_ln = 4 * h;
        if w.len() != want_w || b.len() != want_b || ln.len() != want_ln {
            return Err(format!(
                "weight sizes {}/{}/{} do not match dims {dims:?} (want {want_w}/{want_b}/{want_ln})",
                w.len(),
                b.len(),
                ln.len()
            ));
        }
        let mut wi = 0usize;
        let mut take = |n: usize| {
            let v = w[wi..wi + n].to_vec();
            wi += n;
            v
        };
        let (wd0, wd1, wpile, w0, w1, wb, wc, wg, wu, wq, wk, wp) = (
            take(CARD_FEATS * dc),
            take(dc * de),
            take((PILE_COUNTS + de) * de),
            take(xd * h),
            take(h * h),
            take(2 * dg * h),
            take(hf * dg),
            take(dg * (rk + 1)),
            take(h * rk),
            take(af * rk),
            take(dg * rk),
            take(h * rk),
        );
        let mut bi = 0usize;
        let mut takeb = |n: usize| {
            let v = b[bi..bi + n].to_vec();
            bi += n;
            v
        };
        let (bd0, bd1, bpile, b0, b1, bc, bg, bu, bq, bk, bp) = (
            takeb(dc),
            takeb(de),
            takeb(de),
            takeb(h),
            takeb(h),
            takeb(dg),
            takeb(rk + 1),
            takeb(rk),
            takeb(rk),
            takeb(rk),
            takeb(rk),
        );
        Ok(Mlp {
            dims: dims.to_vec(),
            w0,
            b0,
            ln0_w: ln[..h].to_vec(),
            ln0_b: ln[h..2 * h].to_vec(),
            w1,
            b1,
            ln1_w: ln[2 * h..3 * h].to_vec(),
            ln1_b: ln[3 * h..4 * h].to_vec(),
            wb,
            wc,
            bc,
            wg,
            bg,
            wu,
            bu,
            wq,
            bq,
            wk,
            bk,
            wp,
            bp,
            wd0,
            bd0,
            wd1,
            bd1,
            wpile,
            bpile,
        })
    }

    /// Read the flat weight dump `train/export_weights.py` writes:
    ///
    /// ```text
    /// u32 n_dims, n_dims * u32 dims,
    /// u32 n_w, n_w * f32,   u32 n_b, n_b * f32,   u32 n_ln, n_ln * f32
    /// ```
    pub fn load_bin(path: &str) -> std::io::Result<Mlp> {
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
        let nd = u32_at(&raw, &mut at);
        let dims: Vec<usize> = (0..nd).map(|_| u32_at(&raw, &mut at)).collect();
        let (w, b, ln) = (
            f32s_at(&raw, &mut at),
            f32s_at(&raw, &mut at),
            f32s_at(&raw, &mut at),
        );
        Mlp::from_flat(&dims, &w, &b, &ln)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }
    /// Width of the public encoding.
    pub fn pub_dim(&self) -> usize {
        self.dims[0]
    }
    pub fn hidden(&self) -> usize {
        self.dims[1]
    }
    /// Width of one config vector.
    pub fn cfeat(&self) -> usize {
        self.dims[2]
    }
    /// Width of a config embedding, and of one player's belief block.
    pub fn dg(&self) -> usize {
        self.dims[3]
    }
    /// Width of the value readout's inner product.
    pub fn rank(&self) -> usize {
        self.dims[4]
    }
    /// Width of both players' belief blocks together.
    pub fn belief_dim(&self) -> usize {
        2 * self.dims[3]
    }
    /// Width of one stored action vector, before the paying card's embedding is
    /// appended.
    pub fn afeat(&self) -> usize {
        self.dims[5]
    }
    /// Width of a card embedding.
    pub fn de(&self) -> usize {
        self.dims[6]
    }
    /// Whether this is a checkpoint from before the card describer, which reads
    /// the frozen `v1` encoding and has no policy head.
    pub fn v1(&self) -> bool {
        self.dims.len() == 5
    }
    /// Width of the trunk's input, once the card embeddings are spliced in.
    pub fn xdim(&self) -> usize {
        if self.v1() {
            self.pub_dim()
        } else {
            xdim_of(self.de())
        }
    }

    /// A pre-describer checkpoint: six matrices, the flat public encoding fed
    /// straight to `W0`, and a holding tower that is one linear map of the
    /// counts. Frozen alongside `v1`'s encoder and deleted with it.
    fn from_flat_v1(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Result<Mlp, String> {
        let (p, h, cf, dg, rk) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let want_w = p * h + h * h + 2 * dg * h + cf * dg + dg * (rk + 1) + h * rk;
        let want_b = h + h + dg + (rk + 1) + rk;
        if w.len() != want_w || b.len() != want_b || ln.len() != 4 * h {
            return Err(format!("v1 weight sizes do not match dims {dims:?}"));
        }
        let mut wi = 0usize;
        let mut take = |n: usize| {
            let v = w[wi..wi + n].to_vec();
            wi += n;
            v
        };
        let (w0, w1, wb, wc, wg, wu) = (
            take(p * h),
            take(h * h),
            take(2 * dg * h),
            take(cf * dg),
            take(dg * (rk + 1)),
            take(h * rk),
        );
        Ok(Mlp {
            dims: dims.to_vec(),
            w0,
            b0: b[..h].to_vec(),
            ln0_w: ln[..h].to_vec(),
            ln0_b: ln[h..2 * h].to_vec(),
            w1,
            b1: b[h..2 * h].to_vec(),
            ln1_w: ln[2 * h..3 * h].to_vec(),
            ln1_b: ln[3 * h..4 * h].to_vec(),
            wb,
            wc,
            bc: b[2 * h..2 * h + dg].to_vec(),
            wg,
            bg: b[2 * h + dg..2 * h + dg + rk + 1].to_vec(),
            wu,
            bu: b[2 * h + dg + rk + 1..].to_vec(),
            ..Default::default()
        })
    }

    /// LayerNorm + ReLU over `rows x n`, in place, with an optional cached
    /// addend folded into the bias pass.
    ///
    /// Written for the vectoriser, not for brevity, because it is the hot half
    /// of inference: the matmuls run at ~1.3 Tflop/s through AMX, and a
    /// straightforward `row.iter().sum()` LayerNorm — a serial chain of 3-cycle
    /// adds, 384 long, twice per row — took **five times** as long as all the
    /// matmuls put together. The re-association changes the last bits of the
    /// statistics; `train/test_parity.py` bounds the result against torch.
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

    /// The config tower, for `n` config vectors (`phi` is `[n * cfeat]`).
    /// Produces the belief embedding `z` (`[n * dg]`) and the readout embedding
    /// `g` (`[n * (rank + 1)]`).
    ///
    /// The card table `e`: `[NTYPE, de]`, one embedding per coin type in play.
    ///
    /// Reads the card block of any public row — the cards in play are fixed at
    /// the draft, so every row of a game carries the same block and a solve
    /// builds this once.
    pub fn cards(&self, xpub_row: &[f32], e: &mut Vec<f32>) {
        if self.v1() {
            e.clear();
            return;
        }
        let (de, dc) = (self.de(), self.dims[7]);
        let cards = &xpub_row[OFF_CARDS..OFF_CARDS + NTYPE * CARD_FEATS];
        let mut hid = vec![0.0f32; NTYPE * dc];
        gemm_ld(
            NTYPE, dc, CARD_FEATS, cards, CARD_FEATS, &self.wd0, dc, 0.0, &mut hid, dc,
        );
        for t in 0..NTYPE {
            let row = &mut hid[t * dc..(t + 1) * dc];
            for (x, b) in row.iter_mut().zip(self.bd0.iter()) {
                *x += *b;
            }
            relu(row);
        }
        fit(e, NTYPE * de);
        gemm_ld(
            NTYPE,
            de,
            dc,
            &hid,
            dc,
            &self.wd1,
            de,
            0.0,
            &mut e[..NTYPE * de],
            de,
        );
        for t in 0..NTYPE {
            for (x, b) in e[t * de..(t + 1) * de].iter_mut().zip(self.bd1.iter()) {
                *x += *b;
            }
        }
    }

    /// The trunk's input, assembled from a stored row and the card table:
    /// the raw hex facts, then each hex's occupant embedding, then the pile
    /// summary, then the loose scalars.
    ///
    /// The stored row holds a one-hot per hex rather than an embedding, because
    /// the embedding is learned and a replay row that contained it would go
    /// stale as training moved the weights. Gathering `e`'s row is exactly the
    /// one-hot matmul, since at most one entry is set.
    ///
    /// The blocks are concatenated rather than interleaved per hex. `W0` is
    /// fully connected over the result, so any fixed permutation of its input is
    /// the same network with permuted rows, and this one is two contiguous
    /// copies instead of `N_HEXES` strided ones.
    fn assemble(&self, xpub: &[f32], rows: usize, stride: usize, e: &[f32], x: &mut Vec<f32>) {
        let (de, xd) = (self.de(), self.xdim());
        debug_assert!(!self.v1());
        let (hex_e, piles) = (N_HEXES * HEX_FACTS, N_HEXES * (HEX_FACTS + de));
        fit(x, rows * xd);
        x[..rows * xd].fill(0.0);
        // The pile summary reads [4 counts | card embedding] per coin type. The
        // card half is the same at every leaf of a solve, so it is folded into
        // the bias once and only the four counts move per row -- which turns the
        // rest into a single matmul over every (leaf, coin type).
        let mut pe = vec![0.0f32; NTYPE * de];
        for t in 0..NTYPE {
            let out = &mut pe[t * de..(t + 1) * de];
            out.copy_from_slice(&self.bpile);
            for i in 0..de {
                let v = e[t * de + i];
                let w = &self.wpile[(PILE_COUNTS + i) * de..(PILE_COUNTS + i + 1) * de];
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
        gemm_ld(rows * NTYPE, de, PILE_COUNTS, &cnt, PILE_COUNTS, &self.wpile, de, 0.0,
                &mut ph, de);
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

    /// The holding tower. Per coin type, its three counts and the seat alongside
    /// that card's embedding, through one shared matrix, summed over the five
    /// slots and rectified. The sum is what makes any draft fit: it has no
    /// order, so nothing depends on which slot a card landed in.
    ///
    /// A config's features do not depend on the CFR iteration, so a solve runs
    /// this once for every distinct config in its tree and then never again.
    pub fn embed(&self, phi: &[f32], n: usize, e: &[f32], z: &mut Vec<f32>, g: &mut Vec<f32>) {
        let (rk, dg) = (self.rank(), self.dg());
        debug_assert_eq!(phi.len(), n * self.cfeat());
        fit(z, n * dg);
        if self.v1() {
            let cf = self.cfeat();
            gemm_ld(n, dg, cf, phi, cf, &self.wc, dg, 0.0, &mut z[..n * dg], dg);
            for r in 0..n {
                let row = &mut z[r * dg..r * dg + dg];
                for (x, b) in row.iter_mut().zip(self.bc.iter()) {
                    *x += *b;
                }
                relu(row);
            }
        } else {
            let (de, hf) = (self.de(), HFEAT_OF(self.de()));
        let cf = self.cfeat();
        // The five slot rows are independent and identically shaped, so the
        // whole tower is one matmul over [n * NSLOT, hf] and a segmented sum --
        // not a scalar triple loop per config, which is what this was and which
        // left the vector units idle for the widest per-config work there is.
        let mut inp = vec![0.0f32; n * NSLOT * hf];
        for r in 0..n {
            let p = &phi[r * cf..(r + 1) * cf];
            let seat = p[CCOUNTS];
            for k in 0..NSLOT {
                let row = &mut inp[(r * NSLOT + k) * hf..(r * NSLOT + k + 1) * hf];
                row[0] = p[k];
                row[1] = p[NSLOT + k];
                row[2] = p[2 * NSLOT + k];
                row[3] = seat;
                let t = seat as usize * NSLOT + k;
                row[4..].copy_from_slice(&e[t * de..(t + 1) * de]);
            }
        }
        let mut slot = vec![0.0f32; n * NSLOT * dg];
        gemm_ld(n * NSLOT, dg, hf, &inp, hf, &self.wc, dg, 0.0, &mut slot, dg);
        for r in 0..n {
            let out = &mut z[r * dg..(r + 1) * dg];
            out.fill(0.0);
            for k in 0..NSLOT {
                let o = &slot[(r * NSLOT + k) * dg..(r * NSLOT + k + 1) * dg];
                // Rectify before the sum: a sum of raw linear maps is a linear
                // map of the sum, which has forgotten which count belongs to
                // which card -- the one thing this tower exists to remember.
                for j in 0..dg {
                    out[j] += (o[j] + self.bc[j]).max(0.0);
                }
            }
        }
        }
        fit(g, n * (rk + 1));
        gemm_ld(
            n,
            rk + 1,
            dg,
            z,
            dg,
            &self.wg,
            rk + 1,
            0.0,
            &mut g[..n * (rk + 1)],
            rk + 1,
        );
        for r in 0..n {
            let row = &mut g[r * (rk + 1)..(r + 1) * (rk + 1)];
            for (x, b) in row.iter_mut().zip(self.bg.iter()) {
                *x += *b;
            }
        }
    }

    /// The public tower: `relu(LN(x W0 + b0))` pushed through `W1`, leaving the
    /// hidden layer's pre-activation minus its bias and minus the belief
    /// contribution. Computed once per leaf per solve.
    pub fn trunk(
        &self,
        xpub: &[f32],
        rows: usize,
        stride: usize,
        e: &[f32],
        scratch: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        let (xd, h) = (self.xdim(), self.hidden());
        // v1 feeds the stored row straight to `W0`; there is nothing to splice.
        let mut x = Vec::new();
        let (src, lda) = if self.v1() {
            (xpub, stride)
        } else {
            self.assemble(xpub, rows, stride, e, &mut x);
            (&x[..], xd)
        };
        fit(scratch, rows * h);
        gemm_ld(
            rows,
            h,
            xd,
            src,
            lda,
            &self.w0,
            h,
            0.0,
            &mut scratch[..rows * h],
            h,
        );
        self.ln_relu(
            rows,
            h,
            &self.b0,
            &self.ln0_w,
            &self.ln0_b,
            None,
            &mut scratch[..rows * h],
        );
        fit(out, rows * h);
        gemm_ld(
            rows,
            h,
            h,
            scratch,
            h,
            &self.w1,
            h,
            0.0,
            &mut out[..rows * h],
            h,
        );
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
        self.readout(scratch, rows, &self.wu, &self.bu, out);
    }

    /// The hidden layer itself: the cached trunk in `pre` plus the belief
    /// contribution, normalised and rectified. Both readouts start here, which
    /// is why it is separate — the value's runs every CFR iteration, the
    /// policy's runs once per solve.
    fn hidden_layer(&self, xbel: &[f32], rows: usize, pre: &[f32], out: &mut Vec<f32>) {
        let (h, bd) = (self.hidden(), self.belief_dim());
        debug_assert_eq!(xbel.len(), rows * bd);
        debug_assert!(pre.len() >= rows * h);
        fit(out, rows * h);
        gemm_ld(
            rows,
            h,
            bd,
            xbel,
            bd,
            &self.wb,
            h,
            0.0,
            &mut out[..rows * h],
            h,
        );
        self.ln_relu(
            rows,
            h,
            &self.b1,
            &self.ln1_w,
            &self.ln1_b,
            Some(pre),
            &mut out[..rows * h],
        );
    }

    /// `hid W + b`, into `[rows * rank]`.
    fn readout(&self, hid: &[f32], rows: usize, w: &[f32], b: &[f32], out: &mut Vec<f32>) {
        let (h, rk) = (self.hidden(), self.rank());
        fit(out, rows * rk);
        gemm_ld(rows, rk, h, hid, h, w, rk, 0.0, &mut out[..rows * rk], rk);
        for r in 0..rows {
            for (x, bb) in out[r * rk..r * rk + rk].iter_mut().zip(b.iter()) {
                *x += *bb;
            }
        }
    }

    /// The action tower: `q(a) = relu([psi(a) | e(paying card)] Wq + bq)` for
    /// `na` actions, `psi` being `[na * afeat]`. Writes `[na * rank]`.
    ///
    /// The paying card's embedding is gathered through the coin-type one-hot
    /// `psi` already carries, exactly as the hex block gathers the occupant's —
    /// so what pays for an action is described by what that card does, not by
    /// which slot it sits in.
    ///
    /// Cheap and per node, not per config: an action's description does not
    /// depend on who is holding what.
    pub fn embed_actions(&self, psi: &[f32], na: usize, e: &[f32], out: &mut Vec<f32>) {
        let (af, rk, de) = (self.afeat(), self.rank(), self.de());
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
        gemm_ld(
            na,
            rk,
            af + de,
            &inp,
            af + de,
            &self.wq,
            rk,
            0.0,
            &mut out[..na * rk],
            rk,
        );
        for r in 0..na {
            let row = &mut out[r * rk..r * rk + rk];
            for (x, b) in row.iter_mut().zip(self.bq.iter()) {
                *x += *b;
            }
            relu(row);
        }
    }

    /// Policy logits for one decision node: `[nc * na]`, row-major by config.
    ///
    /// `xbel`/`pre` are that node's single PBS row, `cidx` names its configs in
    /// the `z` table `embed` built, and `q` is `embed_actions`' output. The
    /// caller softmaxes; nothing here knows which actions are legal, because
    /// legality is the caller's mask and applying it twice is how a
    /// renormalisation goes wrong.
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
        let (dg, rk) = (self.dg(), self.rank());
        debug_assert_eq!(out.len(), cidx.len() * na);
        self.hidden_layer(xbel, 1, pre, scratch);
        let mut upi = Vec::new();
        self.readout(scratch, 1, &self.wp, &self.bp, &mut upi);
        let mut k = vec![0.0f32; rk];
        for (ci, &c) in cidx.iter().enumerate() {
            // k(c) = z(c) Wk + bk, added to the PBS readout before the dot with
            // each action: one vector per config, then `na` dot products.
            let zc = &z[c as usize * dg..(c as usize + 1) * dg];
            for j in 0..rk {
                let mut s = self.bk[j] + upi[j];
                for (i, zi) in zc.iter().enumerate() {
                    s += zi * self.wk[i * rk + j];
                }
                k[j] = s;
            }
            for a in 0..na {
                out[ci * na + a] = dot(&k, &q[a * rk..a * rk + rk]);
            }
        }
    }

    /// The per-config readout: `v = <u, g[..rank]> + g[rank]`, for the configs
    /// `idx` names in a `g` table built by `embed`.
    pub fn values(&self, u: &[f32], g: &[f32], idx: &[u32], out: &mut [f32]) {
        let rk = self.rank();
        debug_assert_eq!(idx.len(), out.len());
        for (o, &i) in out.iter_mut().zip(idx.iter()) {
            let row = &g[i as usize * (rk + 1)..];
            *o = dot(u, &row[..rk]) + row[rk];
        }
    }

    /// One value per row, for callers with no solve to amortise over: the torch
    /// parity check and the offline tools. `xpub` is `[rows * pub_dim]`, `xbel`
    /// `[rows * 2 * dg]`, `phi` `[rows * cfeat]` — one config per row.
    ///
    /// The card table is rebuilt per row here, because these callers batch rows
    /// from different games. A solve builds it once.
    pub fn forward(&self, xpub: &[f32], xbel: &[f32], phi: &[f32], rows: usize) -> Vec<f32> {
        let (rk, pd) = (self.rank(), self.pub_dim());
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
                self.cards(&xpub[r * pd..(r + 1) * pd], &mut e);
                self.trunk(&xpub[r * pd..], 1, pd, &e, &mut sb, &mut pre);
                self.pbs_head(&xbel[r * self.belief_dim()..], 1, &pre, &mut sb, &mut u);
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

/// One coin type's input to the holding tower: its three counts, the seat, and
/// its card embedding.
#[allow(non_snake_case)]
const fn HFEAT_OF(de: usize) -> usize {
    4 + de
}

/// Width of the trunk's input, once the card embeddings are spliced in: the raw
/// hex facts, one embedding per hex, the per-player pile summary, the loose
/// scalars.
const fn xdim_of(de: usize) -> usize {
    N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE
}
