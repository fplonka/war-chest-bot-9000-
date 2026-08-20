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

// The belief block the join reads: `sum_c beta(c) g(c)` over one query's
// support. `coff` bounds a query's configs in `cidx`, which names each one's
// row in the resident `g`, and `w` is the normalised reach.
//
// This ran on the host, which meant `g` had to come back off the card and the
// block had to go up again every iteration. One query per thread block, one
// output channel per thread.
__global__ void k_belief_pool(const float* g, const unsigned int* cidx,
                              const unsigned int* coff, const float* w,
                              float* out, int queries, int pool) {
    int q = blockIdx.x;
    if (q >= queries) return;
    unsigned int lo = coff[q], hi = coff[q + 1];
    for (int j = threadIdx.x; j < pool; j += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = lo; k < hi; ++k)
            acc += w[k] * g[(size_t)cidx[k] * pool + j];
        out[(size_t)q * pool + j] = acc;
    }
}

// `v(c) = (<f(c), h_row> + bias) * opp_reach[row]`, for every config of the
// queried player at every row of the batch.
//
// One config per thread block so the D-wide dot product is a block reduction;
// `blockDim.x` must be a power of two.
__global__ void k_readout(const float* f, const float* h, const float* cf_bias,
                          const unsigned int* cidx, const unsigned int* coff,
                          const unsigned int* row_of, const float* opp,
                          float* out, int cells, int d) {
    extern __shared__ float red[];
    int cell = blockIdx.x;
    if (cell >= cells) return;
    int r = row_of[cell];
    const float* fr = f + (size_t)cidx[cell] * d;
    const float* hr = h + (size_t)r * d;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) acc += fr[j] * hr[j];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[cell] = (red[0] + *cf_bias) * opp[r];
    (void)coff;
}

// ---------------------------------------------------------------- CFR sweeps
//
// The arithmetic is `contract.rs`, which reproduces the solver's own walk bit
// for bit. One block per (node, player) and threads striding over that node's
// configs, so neither sweep needs a task list -- building one per level would
// cost what flattening the tree already cost, which is the thing that made
// this port worth doing only once it was incremental.
//
// A level's nodes never depend on each other (`a_level_never_depends_on_itself`
// pins it), so a whole level launches at once and the host walks the levels.

// Reach probabilities for one level. `NO_ROW` marks a node whose parent is not
// of the kind that gather applies to.
__global__ void k_reach_sweep(const unsigned int* level_node, int lo,
                              const unsigned int* parent, const unsigned char* player,
                              const unsigned int* nc, const unsigned int* roff,
                              const unsigned int* rev_base, const unsigned int* rev_start,
                              const unsigned int* rev_src, const unsigned int* rev_cell,
                              const unsigned int* rvd_base, const unsigned int* rvd_start,
                              const unsigned int* rvd_src, const float* rvd_p,
                              const float* cur, float* reach, int nodes) {
    int node = level_node[lo + blockIdx.x];
    int p = blockIdx.y;
    int par = parent[node];
    if (par == 0xffffffffu) return;
    int me = player[par];
    int n = nc[2 * node + p];
    // Where each player's block starts inside a node's reach region.
    int dst = roff[node] + (p == 1 ? nc[2 * node] : 0);
    int src = roff[par] + (p == 1 ? nc[2 * par] : 0);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        if (p != me) {
            // The idle player's information state does not move, and the
            // child's support for them is the same list.
            reach[dst + c] = reach[src + c];
            continue;
        }
        float v = 0.0f;
        unsigned int rb = rev_base[node];
        if (rb != 0xffffffffu) {
            unsigned int a = rev_start[rb + c], b = rev_start[rb + c + 1];
            for (unsigned int k = a; k < b; ++k)
                v += reach[src + rev_src[k]] * cur[rev_cell[k]];
        } else {
            unsigned int db = rvd_base[node];
            if (db != 0xffffffffu) {
                unsigned int a = rvd_start[db + c], b = rvd_start[db + c + 1];
                for (unsigned int k = a; k < b; ++k)
                    v += reach[src + rvd_src[k]] * rvd_p[k];
            }
        }
        reach[dst + c] = v;
    }
    (void)nodes;
}

// Value backpropagation and the regret update for one level, one traverser.
// `kind` is 0 decision, 1 chance, 2 leaf.
__global__ void k_backprop_sweep(const unsigned int* level_node, int lo,
                                 const unsigned char* kind, const unsigned char* player,
                                 const unsigned int* nc, const unsigned int* voff,
                                 const unsigned int* soff,
                                 const unsigned int* child_at, const unsigned int* child_n,
                                 const unsigned int* child,
                                 const unsigned int* legal_base, const unsigned int* legal_off,
                                 const unsigned int* legal_child, const unsigned int* legal_trans,
                                 const unsigned int* draw_base, const unsigned int* draw_start,
                                 const unsigned int* draw_to, const float* draw_p,
                                 float* vals, float* cur, float* regret, float* sum,
                                 int traverser, float da, float db, float dg, float predict) {
    const float EPS = 1e-6f;
    const unsigned int NO_TRANS = 0xffffffffu;
    int node = level_node[lo + blockIdx.x];
    if (kind[node] == 2) return;
    int me = player[node];
    int n = nc[2 * node + traverser];
    int vi = voff[node];

    if (kind[node] == 1) {
        int ch = child[child_at[node]];
        int cv = voff[ch];
        if (me == traverser) {
            unsigned int base = draw_base[node];
            for (int c = threadIdx.x; c < n; c += blockDim.x) {
                unsigned int a = draw_start[base + c], b = draw_start[base + c + 1];
                float v = 0.0f;
                for (unsigned int k = a; k < b; ++k) v += draw_p[k] * vals[cv + draw_to[k]];
                vals[vi + c] = v;
            }
        } else {
            for (int c = threadIdx.x; c < n; c += blockDim.x) vals[vi + c] = vals[cv + c];
        }
        return;
    }

    if (me != traverser) {
        // The traverser's information state is unchanged across an opponent
        // decision, and the opponent's strategy is already in the reaches the
        // leaf values carry.
        unsigned int a = child_at[node], k = child_n[node];
        for (int c = threadIdx.x; c < n; c += blockDim.x) {
            float v = 0.0f;
            for (unsigned int j = a; j < a + k; ++j) v += vals[voff[child[j]] + c];
            vals[vi + c] = v;
        }
        return;
    }

    unsigned int so = soff[node], lb = legal_base[node];
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        unsigned int a = legal_off[lb + c], b = legal_off[lb + c + 1];
        float base = 0.0f;
        for (unsigned int cell = a; cell < b; ++cell) {
            if (legal_trans[so + cell] == NO_TRANS) continue;
            base += vals[voff[legal_child[so + cell]] + legal_trans[so + cell]]
                  * cur[so + cell];
        }
        vals[vi + c] = base;
        float total = 0.0f;
        for (unsigned int cell = a; cell < b; ++cell) {
            float delta = 0.0f;
            if (legal_trans[so + cell] != NO_TRANS)
                delta += vals[voff[legal_child[so + cell]] + legal_trans[so + cell]];
            delta -= base;
            float old = regret[so + cell];
            float r = old * (old > 0.0f ? da : db) + delta;
            regret[so + cell] = r;
            float v = fmaxf(r + predict * delta, EPS);
            cur[so + cell] = v;
            total += v;
        }
        if (total > 0.0f) {
            float inv = 1.0f / total;
            for (unsigned int cell = a; cell < b; ++cell) cur[so + cell] *= inv;
        }
    }
    __syncthreads();
    unsigned int cells = legal_off[lb + n];
    for (unsigned int k = threadIdx.x; k < cells; k += blockDim.x) sum[so + k] *= dg;
}

}  // extern "C"
