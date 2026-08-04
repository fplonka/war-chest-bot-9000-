//! A minimal batched MLP for inference inside the Rust self-play workers.
//!
//! Training happens in PyTorch; the learned weights are pushed here as flat f32
//! row-major matrices (`w[l]` is `[in_dim * out_dim]`, i.e. already transposed
//! from torch's `[out, in]` layout). Forward is ReLU on every layer but the
//! last. On macOS the matmul goes through Accelerate's BLAS; elsewhere a plain
//! triple loop keeps the crate dependency-free.
//!
//! # The split forward
//!
//! Inside a CFR solve the same leaf is queried once per iteration, and between
//! iterations only the *belief* part of its encoding moves — the public part is
//! a property of the leaf's state and is fixed for the whole solve. Since the
//! belief block is the last `split` inputs and `w[0]` is input-major, the first
//! layer factorises as
//!
//! ```text
//! h = x_pub · W_pub  +  x_bel · W_bel
//!     ^^^^^^^^^^^^^   computed once per solve, reused every iteration
//! ```
//!
//! With `FEAT = 812` and a 132-wide belief block that removes 84% of the first
//! layer, which is the widest one. `prefix` computes the cached half and
//! `forward_split` the rest — and the latter also emits only the one output
//! head a CFR traversal actually reads.

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

/// c[m x n] = a[m x k] * b[k x n], all row-major and tightly packed.
fn gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    gemm_ld(m, n, k, a, k, b, n, 0.0, c, n);
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
        let a = add.map_or(std::ptr::null(), |a| a.as_ptr());
        while i + 8 <= n {
            let mut x0 = vaddq_f32(vld1q_f32(p.add(i)), vld1q_f32(b.add(i)));
            let mut x1 = vaddq_f32(vld1q_f32(p.add(i + 4)), vld1q_f32(b.add(i + 4)));
            if !a.is_null() {
                x0 = vaddq_f32(x0, vld1q_f32(a.add(i)));
                x1 = vaddq_f32(x1, vld1q_f32(a.add(i + 4)));
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
            let y0 = vmaxq_f32(vfmaq_f32(vld1q_f32(bp.add(i)), d0, vld1q_f32(gp.add(i))), lo);
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

/// Grow `v` to at least `n` without ever re-zeroing what is already there.
/// Every buffer here is fully overwritten by the matmul that follows, so the
/// `clear() + resize()` this replaces was a megabyte-scale memset per call.
#[inline]
fn fit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

#[derive(Clone, Default)]
pub struct Mlp {
    /// Layer dimensions: `dims[0]` is the input width, `dims[L]` the output.
    pub dims: Vec<usize>,
    /// `w[l]`: row-major `[dims[l] * dims[l+1]]`.
    pub w: Vec<Vec<f32>>,
    /// `b[l]`: `[dims[l+1]]`.
    pub b: Vec<Vec<f32>>,
    /// LayerNorm weight/bias for each hidden layer (`dims.len() - 2` of them,
    /// applied after the affine and before the activation, matching the
    /// reference implementation's `use_layer_norm`). Empty means no norm, which
    /// keeps older checkpoints loading unchanged.
    pub ln_w: Vec<Vec<f32>>,
    pub ln_b: Vec<Vec<f32>>,
}

/// Test/benchmark hook for the raw matmul.
#[allow(clippy::too_many_arguments)]
pub fn gemm_probe(
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
    gemm_ld(m, n, k, a, lda, b, ldb, 0.0, c, ldc);
}

impl Mlp {
    /// Test/benchmark hook for the elementwise pass.
    pub fn activate_probe(&self, l: usize, rows: usize, add: Option<&[f32]>, out: &mut [f32]) {
        self.activate(l, rows, add, out);
    }

    pub fn in_dim(&self) -> usize {
        self.dims[0]
    }
    pub fn out_dim(&self) -> usize {
        self.dims[self.dims.len() - 1]
    }
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }
    pub fn hidden(&self) -> usize {
        self.dims[1]
    }

    /// Bias (plus an optional cached addend) + optional LayerNorm + optional
    /// ReLU, in place over `rows x n`.
    ///
    /// This is written for the vectoriser, not for brevity, because it is the
    /// hot half of inference: the matmuls around it run at ~1.3 Tflop/s through
    /// AMX, and a straightforward `row.iter().sum()` LayerNorm — a serial chain
    /// of 3-cycle adds, 384 long, twice per row — took **five times** as long as
    /// all three matmuls put together. `LANES` independent accumulators break
    /// the chain and let the reductions run in NEON registers. The
    /// re-association changes the last bits of the LayerNorm statistics;
    /// `train/test_parity.py` bounds the result against torch.
    fn activate(&self, l: usize, rows: usize, add: Option<&[f32]>, out: &mut [f32]) {
        let n = self.dims[l + 1];
        let last = l + 1 == self.w.len();
        let bias = &self.b[l];
        let norm = if last || self.ln_w.is_empty() {
            None
        } else {
            Some((&self.ln_w[l], &self.ln_b[l]))
        };
        let inv_n = 1.0 / n as f32;
        for r in 0..rows {
            let row = &mut out[r * n..r * n + n];
            let sum = add_bias_sum(row, bias, add.map(|a| &a[r * n..r * n + n]));
            // LayerNorm over the feature dimension, then the activation.
            // Biased variance (divide by n, not n-1) to match torch.
            if let Some((g, bt)) = norm {
                let mean = sum * inv_n;
                let var = sq_dev(row, mean) * inv_n;
                let inv = 1.0 / (var + LN_EPS).sqrt();
                // Scale, shift and (for a hidden layer) rectify in one pass.
                let floor = if last { f32::NEG_INFINITY } else { 0.0 };
                scale_shift(row, mean, inv, g, bt, floor);
            } else if !last {
                relu(row);
            }
        }
    }

    /// Forward `rows` samples (`x` is `[rows * in_dim]` row-major) into `out`
    /// (`[rows * out_dim]`). `scratch` is reused across calls to avoid allocs.
    /// `out` may be left longer than `rows * out_dim`; only the prefix is
    /// meaningful.
    pub fn forward(&self, x: &[f32], rows: usize, scratch: &mut Vec<f32>, out: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), rows * self.in_dim());
        let layers = self.w.len();
        for l in 0..layers {
            let (k, n) = (self.dims[l], self.dims[l + 1]);
            let src: &[f32] = if l == 0 { x } else { scratch };
            fit(out, rows * n);
            gemm(rows, n, k, src, &self.w[l], &mut out[..rows * n]);
            self.activate(l, rows, None, &mut out[..rows * n]);
            if l + 1 != layers {
                std::mem::swap(scratch, out);
            }
        }
        out.truncate(rows * self.out_dim());
    }

    /// The part of the first layer that depends only on the leading
    /// `in_dim - split` inputs: `out[rows x hidden] = xpub · W[0][..pub]`.
    /// `xpub` is row-major with row stride `stride`, so callers can hold the
    /// public halves of many rows in one packed buffer.
    pub fn prefix(&self, xpub: &[f32], rows: usize, stride: usize, split: usize, out: &mut Vec<f32>) {
        let n = self.dims[1];
        let k = self.dims[0] - split;
        fit(out, rows * n);
        gemm_ld(rows, n, k, xpub, stride, &self.w[0], n, 0.0, out, n);
    }

    /// Finish a forward pass whose public half is already in `prefix`
    /// (`rows x hidden`, from `prefix`). `xbel` is `[rows * split]`. Only the
    /// output columns in `head` are produced, since a CFR traversal reads one
    /// player's head at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_split(
        &self,
        xbel: &[f32],
        rows: usize,
        split: usize,
        prefix: &[f32],
        head: std::ops::Range<usize>,
        scratch: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        let layers = self.w.len();
        let n1 = self.dims[1];
        debug_assert!(prefix.len() >= rows * n1);
        debug_assert_eq!(xbel.len(), rows * split);

        // Layer 0: the belief half, with the cached public half folded in
        // during the bias pass rather than pre-copied into the buffer.
        fit(out, rows * n1);
        let woff = (self.dims[0] - split) * n1;
        gemm_ld(
            rows,
            n1,
            split,
            xbel,
            split,
            &self.w[0][woff..],
            n1,
            0.0,
            &mut out[..rows * n1],
            n1,
        );
        self.activate(0, rows, Some(prefix), &mut out[..rows * n1]);

        for l in 1..layers {
            let (k, n) = (self.dims[l], self.dims[l + 1]);
            std::mem::swap(scratch, out);
            let last = l + 1 == layers;
            let (width, off) = if last { (head.len(), head.start) } else { (n, 0) };
            fit(out, rows * width);
            gemm_ld(
                rows,
                width,
                k,
                scratch,
                k,
                &self.w[l][off..],
                n,
                0.0,
                &mut out[..rows * width],
                width,
            );
            if last {
                // A bias add over the selected head; no norm, no activation.
                let bias = &self.b[l][off..off + width];
                for r in 0..rows {
                    let row = &mut out[r * width..r * width + width];
                    for j in 0..width {
                        row[j] += bias[j];
                    }
                }
            } else {
                self.activate(l, rows, None, &mut out[..rows * width]);
            }
        }
    }
}
