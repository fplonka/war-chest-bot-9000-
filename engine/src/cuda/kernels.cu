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

// ------------------------------------------------------ a round of solves
//
// `f`, `g`, `p`, `jp` and the belief index belong to a solve, not to the round
// that batches it. A stage that launched once per solve to reach them cost the
// round more in launch latency than the arithmetic inside, so every stage below
// takes the solves' pointers as an array instead and launches once.
//
// `part_of_row` names the solve a batch row came from; `base_of_part` is where
// that solve's cells start in the round's `coff` and `w`, so a part-relative
// belief index and a round-relative offset table can be read together.

// Gather each solve's resident rows into the round's layout. `local_row` is the
// row's index inside its own solve.
__global__ void k_gather(const float* const* src, const int* part_of_row,
                         const int* local_row, float* out, int rows, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    int r = i / width, j = i % width;
    out[i] = src[part_of_row[r]][(size_t)local_row[r] * width + j];
}

// The belief block the join reads: `sum_c beta(c) g(c)` over one query's
// support. `coff` bounds a query's cells, `cidx` names each cell's row in its
// solve's `g`, and `w` is the normalised reach.
//
// This ran on the host, which meant `g` had to come back off the card and the
// block had to go up again every iteration. One query per thread block, one
// output channel per thread.
__global__ void k_belief_pool(const float* const* gp, const unsigned int* const* cip,
                              const int* part_of_row, const int* base_of_part,
                              const unsigned int* coff, const float* w,
                              float* out, int queries, int pool) {
    int q = blockIdx.x;
    if (q >= queries) return;
    int part = part_of_row[q >> 1];
    const float* g = gp[part];
    const unsigned int* cidx = cip[part];
    unsigned int base = base_of_part[part], lo = coff[q], hi = coff[q + 1];
    for (int j = threadIdx.x; j < pool; j += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = lo; k < hi; ++k)
            acc += w[k] * g[(size_t)cidx[k - base] * pool + j];
        out[(size_t)q * pool + j] = acc;
    }
}

// `v(c) = (<f(c), h_row> + bias) * opp_reach[row]`, for every config of the
// queried player at every row of the batch.
//
// One block per row, one config per warp, and the row's head vector staged in
// shared memory so it is read once for the row rather than once for each of its
// hundred-odd configs. A row's cells are the span its own query already names
// in `coff`, so nothing here needs a list of cells: building one cost the host
// twelve million pushes a round, which was three quarters of the whole pass.
__global__ void k_readout(const float* const* fp, const unsigned int* const* cip,
                          const int* part_of_row, const int* base_of_part,
                          const float* h, const float* cf_bias,
                          const unsigned int* coff, const unsigned int* vlo,
                          const int* player, const float* opp,
                          float* out, int rows, int d) {
    extern __shared__ float hs[];
    int r = blockIdx.x;
    if (r >= rows) return;
    for (int j = threadIdx.x + 32 * threadIdx.y; j < d; j += 32 * blockDim.y)
        hs[j] = h[(size_t)r * d + j];
    __syncthreads();
    int part = part_of_row[r];
    const float* f = fp[part];
    const unsigned int* cidx = cip[part];
    unsigned int base = base_of_part[part];
    unsigned int q = 2 * r + player[r], lo = coff[q], hi = coff[q + 1];
    float bias = *cf_bias, scale = opp[r];
    unsigned int at = vlo[r];
    for (unsigned int k = lo + threadIdx.y; k < hi; k += blockDim.y) {
        const float* fr = f + (size_t)cidx[k - base] * d;
        float acc = 0.0f;
        for (int j = threadIdx.x; j < d; j += 32) acc += fr[j] * hs[j];
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
        if (threadIdx.x == 0) out[at + k - lo] = (acc + bias) * scale;
    }
}

// ------------------------------------------------------------- the CFR loop
//
// The arithmetic is `contract.rs`, which reproduces the solver's own walk bit
// for bit. A level's nodes never depend on each other
// (`a_level_never_depends_on_itself` pins it), so one launch covers a whole
// level -- of *every* solve in the round, since solves do not depend on each
// other either. `blockIdx.y` is the solve, `blockIdx.x` the node within its
// level, and a solve with a shallower tree simply has no work at the deeper
// levels.
//
// Everything a solve's iteration reads or writes is named by one descriptor, so
// a stage takes one array of them rather than thirty arrays of pointers. The
// field order below is `Card::describe` in `cuda/mod.rs`; every field is eight
// bytes wide so the layout is positional and nothing needs packing rules.

struct Tree {
    const unsigned int* kind;
    const unsigned int* player;
    const unsigned int* nc;
    const unsigned int* parent;
    const unsigned int* roff;
    const unsigned int* voff;
    const unsigned int* soff;
    const float* util;
    const unsigned int* child_at;
    const unsigned int* child_n;
    const unsigned int* child;
    const unsigned int* legal_base;
    const unsigned int* legal_off;
    const unsigned int* legal_child;
    const unsigned int* legal_trans;
    const unsigned int* cell_row;
    const unsigned int* rev_base;
    const unsigned int* rev_start;
    const unsigned int* rev_src;
    const unsigned int* rev_cell;
    const unsigned int* rvd_base;
    const unsigned int* rvd_start;
    const unsigned int* rvd_src;
    const float* rvd_p;
    const unsigned int* draw_base;
    const unsigned int* draw_start;
    const unsigned int* draw_to;
    const float* draw_p;
    const unsigned int* level_start;
    const unsigned int* level_node;
    // The solve's own arenas, laid out exactly as `Solver` lays them out.
    float* reach;
    float* vals;
    float* cur;
    float* regret;
    float* sum;
    float* qval;
    float* visits;
    float* prior;
    const float* rootb;
    // Batch row -> node, for the leaves the network answers for.
    const unsigned int* leaf_node;
    unsigned long long levels;
    unsigned long long rows;
};

#define NO_ROW 0xffffffffu
#define NO_TRANS 0xffffffffu
#define KIND_LEAF 2u
#define KIND_CHANCE 1u

// The nodes of one level of one solve, or none when this solve is shallower.
__device__ __forceinline__ bool level_task(const Tree& t, int level, int slot,
                                           unsigned int* node) {
    if ((unsigned)level + 1 >= t.levels + 1) return false;
    unsigned int lo = t.level_start[level], hi = t.level_start[level + 1];
    if ((unsigned)slot >= hi - lo) return false;
    *node = t.level_node[lo + slot];
    return true;
}

// Where player `p`'s block starts inside node `i`'s reach region.
__device__ __forceinline__ unsigned int rbase(const Tree& t, unsigned int i, int p) {
    return t.roff[i] + (p == 1 ? t.nc[2 * i] : 0);
}

// The root beliefs, before the first level of the sweep reads them.
__global__ void k_seed_reach(const Tree* trees) {
    const Tree& t = trees[blockIdx.y];
    unsigned int n = t.nc[0] + t.nc[1];
    for (unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; c < n;
         c += gridDim.x * blockDim.x)
        t.reach[t.roff[0] + c] = t.rootb[c];
}

// Reach probabilities for one level, both players.
__global__ void k_reach_sweep(const Tree* trees, int level) {
    const Tree& t = trees[blockIdx.y];
    unsigned int node;
    if (!level_task(t, level, blockIdx.x, &node)) return;
    unsigned int par = t.parent[node];
    if (par == NO_ROW) return;
    unsigned int me = t.player[par];
    for (int p = 0; p < 2; ++p) {
        unsigned int n = t.nc[2 * node + p];
        unsigned int dst = rbase(t, node, p), src = rbase(t, par, p);
        for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
            if ((unsigned)p != me) {
                // The idle player's information state does not move, and the
                // child's support for them is the same list.
                t.reach[dst + c] = t.reach[src + c];
                continue;
            }
            float v = 0.0f;
            unsigned int rb = t.rev_base[node];
            if (rb != NO_ROW) {
                unsigned int a = t.rev_start[rb + c], b = t.rev_start[rb + c + 1];
                for (unsigned int k = a; k < b; ++k)
                    v += t.reach[src + t.rev_src[k]] * t.cur[t.rev_cell[k]];
            } else {
                unsigned int db = t.rvd_base[node];
                if (db != NO_ROW) {
                    unsigned int a = t.rvd_start[db + c], b = t.rvd_start[db + c + 1];
                    for (unsigned int k = a; k < b; ++k)
                        v += t.reach[src + t.rvd_src[k]] * t.rvd_p[k];
                }
            }
            t.reach[dst + c] = v;
        }
    }
}

// Value backpropagation and the regret update for one level, one traverser.
__global__ void k_backprop_sweep(const Tree* trees, int level, int traverser,
                                 float da, float db, float dg, float predict) {
    const float EPS = 1e-6f;
    const Tree& t = trees[blockIdx.y];
    unsigned int node;
    if (!level_task(t, level, blockIdx.x, &node)) return;
    if (t.kind[node] == KIND_LEAF) return;
    unsigned int me = t.player[node];
    unsigned int n = t.nc[2 * node + traverser];
    unsigned int vi = t.voff[node];

    if (t.kind[node] == KIND_CHANCE) {
        unsigned int cv = t.voff[t.child[t.child_at[node]]];
        if (me == (unsigned)traverser) {
            unsigned int base = t.draw_base[node];
            for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
                unsigned int a = t.draw_start[base + c], b = t.draw_start[base + c + 1];
                float v = 0.0f;
                for (unsigned int k = a; k < b; ++k)
                    v += t.draw_p[k] * t.vals[cv + t.draw_to[k]];
                t.vals[vi + c] = v;
            }
        } else {
            for (unsigned int c = threadIdx.x; c < n; c += blockDim.x)
                t.vals[vi + c] = t.vals[cv + c];
        }
        return;
    }

    if (me != (unsigned)traverser) {
        // The traverser's information state is unchanged across an opponent
        // decision, and the opponent's strategy is already in the reaches the
        // leaf values carry.
        unsigned int a = t.child_at[node], k = t.child_n[node];
        for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
            float v = 0.0f;
            for (unsigned int j = a; j < a + k; ++j) v += t.vals[t.voff[t.child[j]] + c];
            t.vals[vi + c] = v;
        }
        return;
    }

    unsigned int so = t.soff[node], lb = t.legal_base[node];
    // The expansion phase reads these as PUCT's Q. A sweep that computes the
    // action values and drops them leaves selection blind, and the tree it
    // grows is a different tree -- which is a wrong answer no shape check
    // would catch.
    unsigned int ncells = t.legal_off[lb + n];
    for (unsigned int k = threadIdx.x; k < ncells; k += blockDim.x) t.qval[so + k] = 0.0f;
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
        unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
        float base = 0.0f;
        for (unsigned int cell = a; cell < b; ++cell) {
            if (t.legal_trans[so + cell] == NO_TRANS) continue;
            float av = t.vals[t.voff[t.legal_child[so + cell]] + t.legal_trans[so + cell]];
            t.qval[so + cell] = av;
            base += av * t.cur[so + cell];
        }
        t.vals[vi + c] = base;
        float total = 0.0f;
        for (unsigned int cell = a; cell < b; ++cell) {
            float delta = t.qval[so + cell] - base;
            float old = t.regret[so + cell];
            float r = old * (old > 0.0f ? da : db) + delta;
            t.regret[so + cell] = r;
            float v = fmaxf(r + predict * delta, EPS);
            t.cur[so + cell] = v;
            total += v;
        }
        if (total > 0.0f) {
            float inv = 1.0f / total;
            for (unsigned int cell = a; cell < b; ++cell) t.cur[so + cell] *= inv;
        }
    }
    __syncthreads();
    for (unsigned int k = threadIdx.x; k < ncells; k += blockDim.x) t.sum[so + k] *= dg;
}

// The reach-weighted iterate, added to the running strategy sum. Both players
// in one pass: a decision node belongs to exactly one of them.
__global__ void k_avg_block(const Tree* trees, int level) {
    const Tree& t = trees[blockIdx.y];
    unsigned int node;
    if (!level_task(t, level, blockIdx.x, &node)) return;
    if (t.kind[node] != 0) return;
    unsigned int me = t.player[node];
    unsigned int n = t.nc[2 * node + me], so = t.soff[node], lb = t.legal_base[node];
    unsigned int ra = rbase(t, node, me);
    for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
        float r = t.reach[ra + c];
        for (unsigned int cell = t.legal_off[lb + c]; cell < t.legal_off[lb + c + 1]; ++cell)
            t.sum[so + cell] += r * t.cur[so + cell];
    }
}

// Every leaf's normalised belief, and the opponent's reach mass there. One
// block per (row, player); `w` and `oppmass` are what the network reads.
__global__ void k_beliefs(const Tree* trees, const int* row_of_part, const int* part_of_row,
                          const int* local_row, const unsigned int* coff, int traverser,
                          float* w, float* oppmass, int rows) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int part = part_of_row[r];
    const Tree& t = trees[part];
    unsigned int node = t.leaf_node[local_row[r]];
    int p = blockIdx.y;
    unsigned int n = t.nc[2 * node + p], ra = rbase(t, node, p);
    unsigned int lo = coff[2 * r + p];
    // One warp: the support is tens of configs, and the sum has to be seen by
    // every thread that then divides by it.
    float acc = 0.0f;
    for (unsigned int c = threadIdx.x; c < n; c += 32) acc += t.reach[ra + c];
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, s);
    float inv = acc > 0.0f ? 1.0f / acc : 1.0f / (float)max(n, 1u);
    for (unsigned int c = threadIdx.x; c < n; c += 32)
        w[lo + c] = acc > 0.0f ? t.reach[ra + c] * inv : inv;
    if (threadIdx.x == 0 && p == 1 - traverser) oppmass[r] = acc;
    (void)row_of_part;
}

// ------------------------------------------------------------ the expansion
//
// `xorshift64*` and the two draws built on it, transcribed from `rng.rs` and
// `search.rs::pick` so a device trajectory and a host one from the same seed
// take the same turns.

__device__ __forceinline__ unsigned long long rng_next(unsigned long long* s) {
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

__device__ __forceinline__ double rng_unit(unsigned long long* s) {
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

// Draw an index from non-negative weights. A row whose weights have all
// underflowed is drawn uniformly rather than dropped.
__device__ int pick(const float* w, int n, unsigned long long* s) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += (double)fmaxf(w[i], 0.0f);
    if (total == 0.0) return n > 0 ? (int)(rng_next(s) % (unsigned long long)n) : 0;
    double needle = rng_unit(s) * total;
    for (int i = 0; i < n; ++i) {
        needle -= (double)fmaxf(w[i], 0.0f);
        if (needle < 0.0) return i;
    }
    return n - 1;
}

// The cell PUCT would take from one config's legal row.
//
// `Q + c_puct * P * sqrt(sum N) / (1 + N)`, with `Q` divided by the opponent's
// reach mass at the node -- without it a node behind an unlikely opponent line
// looks worthless beside its siblings instead of being compared with them.
__device__ int puct_choice(const Tree& t, unsigned int node, unsigned int a,
                           unsigned int b, int opp, float c_puct) {
    unsigned int so = t.soff[node], ra = rbase(t, node, opp);
    float mass = 0.0f;
    for (unsigned int c = 0; c < t.nc[2 * node + opp]; ++c) mass += t.reach[ra + c];
    float scale = mass > 1e-30f ? 1.0f / mass : 0.0f;
    float total = 0.0f;
    for (unsigned int cell = a; cell < b; ++cell) total += t.visits[so + cell];
    float explore = c_puct * sqrtf(fmaxf(total, 0.0f));
    int best = (int)a;
    float best_score = neg_inf();
    for (unsigned int cell = a; cell < b; ++cell) {
        float score = t.qval[so + cell] * scale
                    + explore * t.prior[so + cell] / (1.0f + t.visits[so + cell]);
        if (score > best_score) {
            best_score = score;
            best = (int)cell;
        }
    }
    return best;
}

// One expansion phase: `sims` trajectories for every solve in the round, each
// sampling a world from the root beliefs and walking down under
// `pi_select = 1/2 pi_PUCT + 1/2 pi_CFR`. PUCT is a maximisation, so its half
// is a point mass on the argmax and sampling the mixture is a coin flip.
//
// The simulations of one phase are run in order by a single thread, not in
// parallel: each increments the visits it passes, which is the paper's virtual
// loss, and a later simulation of the same phase is meant to see it.
__global__ void k_expand(const Tree* trees, unsigned long long* seed,
                         unsigned int* out, int sims, float c_puct) {
    int part = blockIdx.x * blockDim.x + threadIdx.x;
    const Tree& t = trees[part];
    unsigned long long s = seed[part];
    unsigned int n0 = t.nc[0], n1 = t.nc[1];
    for (int sim = 0; sim < sims; ++sim) {
        int c[2];
        c[0] = pick(t.rootb, (int)n0, &s);
        c[1] = pick(t.rootb + n0, (int)n1, &s);
        unsigned int node = 0;
        unsigned int found = NO_ROW;
        for (;;) {
            unsigned int k = t.kind[node];
            if (k == KIND_LEAF) {
                found = node;
                break;
            }
            unsigned int me = t.player[node];
            if (k == KIND_CHANCE) {
                unsigned int base = t.draw_base[node];
                unsigned int a = t.draw_start[base + c[me]], b = t.draw_start[base + c[me] + 1];
                int j = pick(t.draw_p + a, (int)(b - a), &s);
                c[me] = (int)t.draw_to[a + j];
                node = t.child[t.child_at[node]];
                continue;
            }
            unsigned int lb = t.legal_base[node], so = t.soff[node];
            unsigned int a = t.legal_off[lb + c[me]], b = t.legal_off[lb + c[me] + 1];
            if (a == b) break;
            int cell;
            if (rng_unit(&s) < 0.5) {
                cell = puct_choice(t, node, a, b, 1 - me, c_puct);
            } else {
                bool any = false;
                for (unsigned int q = a; q < b; ++q) any |= t.sum[so + q] > 0.0f;
                const float* row = any ? t.sum + so + a : t.cur + so + a;
                cell = (int)a + pick(row, (int)(b - a), &s);
            }
            t.visits[so + cell] += 1.0f;
            unsigned int trans = t.legal_trans[so + cell];
            if (trans == NO_TRANS) break;
            c[me] = (int)trans;
            node = t.legal_child[so + cell];
        }
        out[part * sims + sim] = found;
    }
    seed[part] = s;
}

// Terminal leaves are scored from the game, not the network, so the backend
// never saw them. One block per terminal, listed per solve.
__global__ void k_terminals(const Tree* trees, const unsigned int* term,
                            const unsigned int* term_at, int traverser, int most) {
    const Tree& t = trees[blockIdx.y];
    unsigned int a = term_at[blockIdx.y], b = term_at[blockIdx.y + 1];
    if (blockIdx.x >= b - a || (int)blockIdx.x >= most) return;
    unsigned int node = term[a + blockIdx.x];
    int opp = 1 - traverser;
    unsigned int n = t.nc[2 * node + opp], ra = rbase(t, node, opp);
    float acc = 0.0f;
    for (unsigned int c = threadIdx.x; c < n; c += 32) acc += t.reach[ra + c];
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, s);
    // Zero-sum by construction, so one stored utility serves both seats.
    float u = t.player[node] == (unsigned)traverser ? t.util[node] : -t.util[node];
    unsigned int m = t.nc[2 * node + traverser], vo = t.voff[node];
    for (unsigned int c = threadIdx.x; c < m; c += 32) t.vals[vo + c] = u * acc;
}

}  // extern "C"
