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

impl Mlp {
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

    /// Bias + optional LayerNorm + optional ReLU, in place over `rows x n`.
    fn activate(&self, l: usize, rows: usize, out: &mut [f32]) {
        let n = self.dims[l + 1];
        let last = l + 1 == self.w.len();
        let bias = &self.b[l];
        let norm = if last || self.ln_w.is_empty() {
            None
        } else {
            Some((&self.ln_w[l], &self.ln_b[l]))
        };
        for r in 0..rows {
            let row = &mut out[r * n..r * n + n];
            for j in 0..n {
                row[j] += bias[j];
            }
            // LayerNorm over the feature dimension, then the activation.
            // Biased variance (divide by n, not n-1) to match torch.
            if let Some((g, bt)) = norm {
                let mean = row.iter().sum::<f32>() / n as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
                let inv = 1.0 / (var + LN_EPS).sqrt();
                for j in 0..n {
                    row[j] = (row[j] - mean) * inv * g[j] + bt[j];
                }
            }
            if !last {
                for j in 0..n {
                    if row[j] < 0.0 {
                        row[j] = 0.0;
                    }
                }
            }
        }
    }

    /// Forward `rows` samples (`x` is `[rows * in_dim]` row-major) into `out`
    /// (`[rows * out_dim]`). `scratch` is reused across calls to avoid allocs.
    pub fn forward(&self, x: &[f32], rows: usize, scratch: &mut Vec<f32>, out: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), rows * self.in_dim());
        let layers = self.w.len();
        for l in 0..layers {
            let (k, n) = (self.dims[l], self.dims[l + 1]);
            let src: &[f32] = if l == 0 { x } else { scratch };
            out.clear();
            out.resize(rows * n, 0.0);
            gemm(rows, n, k, src, &self.w[l], out);
            self.activate(l, rows, out);
            if l + 1 != layers {
                std::mem::swap(scratch, out);
            }
        }
    }

    /// The part of the first layer that depends only on the leading
    /// `in_dim - split` inputs: `out[rows x hidden] = xpub · W[0][..pub]`.
    /// `xpub` is row-major with row stride `stride`, so callers can hold the
    /// public halves of many rows in one packed buffer.
    pub fn prefix(&self, xpub: &[f32], rows: usize, stride: usize, split: usize, out: &mut Vec<f32>) {
        let n = self.dims[1];
        let k = self.dims[0] - split;
        out.clear();
        out.resize(rows * n, 0.0);
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
        debug_assert_eq!(prefix.len(), rows * n1);
        debug_assert_eq!(xbel.len(), rows * split);

        // Layer 0: the cached public half plus the belief half.
        out.clear();
        out.extend_from_slice(prefix);
        let woff = (self.dims[0] - split) * n1;
        gemm_ld(rows, n1, split, xbel, split, &self.w[0][woff..], n1, 1.0, out, n1);
        self.activate(0, rows, out);

        for l in 1..layers {
            let (k, n) = (self.dims[l], self.dims[l + 1]);
            std::mem::swap(scratch, out);
            let last = l + 1 == layers;
            let (width, off) = if last { (head.len(), head.start) } else { (n, 0) };
            out.clear();
            out.resize(rows * width, 0.0);
            gemm_ld(rows, width, k, scratch, k, &self.w[l][off..], n, 0.0, out, width);
            if last {
                // A bias add over the selected head; no norm, no activation.
                let bias = &self.b[l][off..off + width];
                for r in 0..rows {
                    for j in 0..width {
                        out[r * width + j] += bias[j];
                    }
                }
            } else {
                self.activate(l, rows, out);
            }
        }
    }
}
