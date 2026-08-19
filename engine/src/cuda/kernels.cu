// Elementwise and gather work for the value network. The matrix multiplies are
// cuBLAS; everything that is not a GEMM is here.
//
// Compiled by NVRTC at startup, so the shapes arrive as -D defines and the
// code below is the same arithmetic as `net.rs`, in the same order.
//
// Every kernel takes a whole round's batch. Calls of one kind are concatenated
// before they get here, so where the CPU network walks one solve's rows these
// walk every solve's rows at once. Two conventions carry the concatenation:
//
//  * a leaf's physical `xpub` row is `2 * r`, because the paired canonical
//    queries stay adjacent when the calls are joined, so a strided read
//    replaces the copy `net::board` makes;
//  * anything that was per-call and is now per-row — the card table a row
//    reads, the seat a join queries — arrives as an index array.

extern "C" {

// NVRTC compiles without the C headers, so `INFINITY` does not exist here.
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float gelu1(float x) {
    // tanh approximation, matching `net::gelu`.
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__global__ void k_gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu1(x[i]);
}

__global__ void k_add(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// One row per block: mean and variance over `width`, then scale and shift.
// `act` folds in the GELU, which is `Norm::apply`; without it this is
// `Norm::plain`. `blockDim.x` must be a power of two for the reduction.
__global__ void k_layernorm(float* x, const float* gamma, const float* beta,
                            int rows, int width, int act) {
    extern __shared__ float sh[];
    int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + (size_t)r * width;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) sum += row[j];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sh[0] / width;
    __syncthreads();
    float var = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float d = row[j] - mean;
        var += d * d;
    }
    sh[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / width + 1e-5f);
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = (row[j] - mean) * inv * gamma[j] + beta[j];
        row[j] = act ? gelu1(v) : v;
    }
}

// `out[r, j] += b[j]` — the per-column bias a GEMM does not carry.
__global__ void k_bias(float* out, const float* b, int rows, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) out[i] += b[i % width];
}

// `out[cell, j] += bias[cell / span, j]` — a per-row vector broadcast down a
// group of cells, which is how the pooled global bias reaches every hex.
__global__ void k_group_bias(float* out, const float* bias, int cells,
                             int width, int span) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells * width) return;
    int cell = i / width, j = i % width;
    out[i] += bias[(size_t)(cell / span) * width + j];
}

// A contiguous window out of each source row, into a packed matrix. Pile counts
// and the loose scalars both reach their GEMM this way; `stride` is `2 *
// PUBFEAT`, which picks the physical row of each leaf.
__global__ void k_window(const float* src, float* out, int rows, int stride,
                         int off, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    out[i] = src[(size_t)(i / width) * stride + off + (i % width)];
}

// Card token + the owner's seat embedding, added onto the projected pile
// counts already in `out`. A row reads the card table of the solve it came
// from, because a batch spans solves.
__global__ void k_tokens(const float* cards, const int* card_of_row,
                         const float* seat, float* out, int rows, int ntype,
                         int type, int nslot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ntype * type) return;
    int r = i / (ntype * type), t = (i / type) % ntype, j = i % type;
    out[i] += cards[((size_t)card_of_row[r] * ntype + t) * type + j]
            + seat[(size_t)(t / nslot) * type + j];
}

// Hex facts, and which coin type stands on each hex.
__global__ void k_hex_facts(const float* xpub, float* facts, int* occupant,
                            int rows, int stride, int nhex, int hex_ch,
                            int hex_facts, int ntype) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= rows * nhex) return;
    int r = cell / nhex, h = cell % nhex;
    const float* hex = xpub + (size_t)r * stride + (size_t)h * hex_ch;
    for (int j = 0; j < hex_facts; ++j) facts[(size_t)cell * hex_facts + j] = hex[j];
    int who = -1;
    for (int t = 0; t < ntype; ++t)
        if (hex[hex_facts + t] != 0.0f) { who = t; break; }
    occupant[cell] = who;
}

// Mean of the gelu'd projected tokens, per row.
__global__ void k_type_pool(const float* projected, float* out, int rows,
                            int ntype, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * c) return;
    int r = i / c, j = i % c;
    float acc = 0.0f;
    for (int t = 0; t < ntype; ++t)
        acc += gelu1(projected[((size_t)r * ntype + t) * c + j]);
    out[i] = acc / ntype;
}

// The stem sum: hex projection + occupant token + position + global + pooled.
__global__ void k_stem(float* x, const float* projected, const int* occupant,
                       const float* pos, const float* glob,
                       const float* type_pool, int cells, int nhex, int ntype,
                       int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells * c) return;
    int cell = i / c, j = i % c;
    int r = cell / nhex, h = cell % nhex;
    float acc = pos[(size_t)h * c + j] + glob[(size_t)r * c + j]
              + type_pool[(size_t)r * c + j];
    int t = occupant[cell];
    if (t >= 0) acc += projected[((size_t)r * ntype + t) * c + j];
    x[i] += acc;
}

// `[self | sum of neighbours]` per hex, the input to a block's mix.
__global__ void k_neighbour_mix(const float* a, const int* nb, float* mixed,
                                int cells, int nhex, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells * c) return;
    int cell = i / c, j = i % c;
    int h = cell % nhex, base = cell - h;
    mixed[(size_t)cell * 2 * c + j] = a[i];
    float acc = 0.0f;
    for (int k = 0; k < 6; ++k) {
        int n = nb[(size_t)h * 6 + k];
        if (n >= 0) acc += a[((size_t)base + n) * c + j];
    }
    mixed[(size_t)cell * 2 * c + c + j] = acc;
}

// Mean and max over a row's hexes, side by side.
__global__ void k_pool(const float* a, float* pooled, int rows, int nhex,
                       int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * c) return;
    int r = i / c, j = i % c;
    float mean = 0.0f, mx = neg_inf();
    for (int h = 0; h < nhex; ++h) {
        float v = a[((size_t)r * nhex + h) * c + j];
        mean += v / nhex;
        mx = fmaxf(mx, v);
    }
    pooled[(size_t)r * 2 * c + j] = mean;
    pooled[(size_t)r * 2 * c + c + j] = mx;
}

// The board head's input: pooled mean and max, then the loose scalars.
__global__ void k_board_input(const float* x, const float* xpub, float* out,
                              int rows, int nhex, int c, int stride,
                              int off_loose, int loose) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = 2 * c + loose;
    if (i >= rows * width) return;
    int r = i / width, j = i % width;
    if (j < c) {
        float acc = 0.0f;
        for (int h = 0; h < nhex; ++h) acc += x[((size_t)r * nhex + h) * c + j];
        out[i] = acc / nhex;
    } else if (j < 2 * c) {
        float mx = neg_inf();
        for (int h = 0; h < nhex; ++h)
            mx = fmaxf(mx, x[((size_t)r * nhex + h) * c + j - c]);
        out[i] = mx;
    } else {
        out[i] = xpub[(size_t)r * stride + off_loose + j - 2 * c];
    }
}

// One slot row per (config, slot): three counts then that slot's card token.
__global__ void k_cfg_slots(const float* phi, const unsigned int* owner,
                            const float* cards, float* slots, int n, int nslot,
                            int cfeat, int ntype, int type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = 3 + type;
    if (i >= n * nslot * width) return;
    int row = i / width, j = i % width;
    int cfg = row / nslot, k = row % nslot;
    if (j == 0) slots[i] = phi[(size_t)cfg * cfeat + k];
    else if (j == 1) slots[i] = phi[(size_t)cfg * cfeat + nslot + k];
    else if (j == 2) slots[i] = phi[(size_t)cfg * cfeat + 2 * nslot + k];
    else slots[i] = cards[((size_t)owner[cfg] * ntype + k) * type + j - 3];
}

// Sum a config's slot rows back into one vector.
__global__ void k_sum_slots(const float* hidden, float* out, int n, int nslot,
                            int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    int cfg = i / width, j = i % width;
    float acc = 0.0f;
    for (int k = 0; k < nslot; ++k)
        acc += hidden[((size_t)cfg * nslot + k) * width + j];
    out[i] = acc;
}

// The linear half of `g`: a count-weighted sum of per-zone card embeddings, so
// that pooling a belief carries its exact expected holding of every card.
__global__ void k_bag(const float* bag, const float* phi,
                      const unsigned int* owner, float* g, int n, int nslot,
                      int ntype, int cfeat, int pool) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * pool) return;
    int cfg = i / pool, j = i % pool;
    const float* v = bag + (size_t)owner[cfg] * ntype * 3 * pool;
    float acc = 0.0f;
    for (int k = 0; k < nslot; ++k)
        for (int zone = 0; zone < 3; ++zone) {
            float count = phi[(size_t)cfg * cfeat + zone * nslot + k];
            if (count != 0.0f) acc += count * v[((size_t)k * 3 + zone) * pool + j];
        }
    g[i] += acc;
}

// `[own pooled | opponent pooled | seat]`, the join's belief-dependent input.
// The queried seat is per row because a round joins solves that are asking
// about different players.
__global__ void k_join_input(const float* pooled, const int* player, float* out,
                             int rows, int pool) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = 2 * pool + 1;
    if (i >= rows * width) return;
    int r = i / width, j = i % width;
    int p = player[r];
    if (j < pool) out[i] = pooled[((size_t)2 * r + p) * pool + j];
    else if (j < 2 * pool)
        out[i] = pooled[((size_t)2 * r + 1 - p) * pool + j - pool];
    else out[i] = p == 0 ? -1.0f : 1.0f;
}

}  // extern "C"
