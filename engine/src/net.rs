//! A minimal batched MLP for inference inside the Rust self-play workers.
//!
//! Training happens in PyTorch; the learned weights are pushed here as flat f32
//! row-major matrices (`w[l]` is `[in_dim * out_dim]`, i.e. already transposed
//! from torch's `[out, in]` layout). Forward is ReLU on every layer but the
//! last. On macOS the matmul goes through Accelerate's BLAS; elsewhere a plain
//! triple loop keeps the crate dependency-free.

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

/// c[m x n] = a[m x k] * b[k x n], all row-major.
fn gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
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
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = 0.0;
            }
            for p in 0..k {
                let av = a[i * k + p];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
    }
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
            let last = l + 1 == layers;
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
            if !last {
                std::mem::swap(scratch, out);
            }
        }
    }
}
