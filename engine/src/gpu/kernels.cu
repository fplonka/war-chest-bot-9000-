// Phase kernels for the GPU solve service.
//
// Every kernel ports one function of engine/src/search.rs or engine/src/net.rs
// with the same formulas and the same reduction orders, so a solve's result
// cannot depend on which other solves share its tick. The CPU solver is the
// oracle; the in-crate tests compare kernel arenas with the Rust functions.
//
// Structure rules:
//
//   1. Every kernel has the same signature: (descs, group, ctx). `fire` in
//      service.rs is the only launch site.
//   2. A solve's arrays are reached only through the T_* and A_* accessors
//      that gpu/layout.rs generates. This file never names a byte offset.
//   3. Solve kernels are one *block per solve*: `g->slots[blockIdx.x]` is the
//      solve, warps stride over its rows/nodes, and a sweep orders its levels
//      with block barriers. All of a block's traffic is one solve's memory.
//   4. Network shape is compile-time: the service NVRTC-compiles this file
//      per checkpoint shape (DG, RK, HEADW, tower widths ...), so register
//      arrays size exactly and width loops unroll. Any shape works; changing
//      shape restarts the service.
//   5. Stages share launches: a kernel reads d->stage and returns early when
//      the phase does not apply. Iterate, value and carry solves therefore
//      advance in the same tick with no per-stage launch sets.
//
// The `Desc`, `T_*`/`A_*` accessors, the board geometry and the shape
// constants are prepended by gpu/layout.rs before NVRTC sees this file.

#define EPS 1e-6f
#define LN_EPS 1e-5f
#define SMOOTH 1e-30f
#define NONE 0xffffffffu
#define INFINITY_F (__int_as_float(0x7f800000))

#define DG_CH ((DG + 31) / 32)
#define RK_CH ((RK + 31) / 32)
#define HEAD_CH ((HEADW + 31) / 32)

// The network weights, the row pools and the build scratch: everything a
// tick shares across solves. Uploaded once per weight publication; the batch
// map buffer is owned by the service and refilled per admission batch.
typedef struct {
    // Card tower, identity table, pile layer.
    const float *card_w[8], *card_b[8];
    const float *wid;
    const float *pile_w, *pile_b;
    // Public tower (LN+ReLU layers), projection, head entry.
    const float *pub_w[8], *pub_b[8], *pub_lnw[8], *pub_lnb[8];
    const float *pub_out_w, *pub_out_b;
    const float *wb, *ln1w, *ln1b;
    // Extra head layers and the value readout.
    const float *hmlp_w[8], *hmlp_b[8];
    const float *wu_w, *wu_b;
    // Holding tower.
    const float *slot_w[8], *slot_b[8];
    const float *slot_out_w, *slot_out_b;
    const float *res_aw[4], *res_ab[4], *res_bw[4], *res_bb[4];
    const float *wg_w, *wg_b;
    // Row pools, indexed by a solve's stable row base (row0).
    float *h0, *xb, *h, *h2, *u;
    // Build scratch (batch admission): trunk input, ping/pong hidden, gather.
    float *bx, *bh, *bh2, *bg;
    // The admission batch map: nb entries of BMAP_INTS ints each
    // (slot, row0-in-batch, nrows, cfg0-in-batch, ncfg).
    const int *bmap;
} Ctx;

#define BMAP_INTS 5
// Entries the service's batch map can hold (`MAX_BATCH` in service.rs).
#define BMAP_CAP 32

// The solves a launch covers: one block per entry of `slots`. The scalars are
// per-launch switches; everything per-solve lives in the descriptor.
typedef struct {
    const int* slots;
    const int* prefix;
    int n;
    int mode;      // kernel-specific switch (documented at each kernel)
    int p_player;  // readout/backprop player: -1 = the solve's traverser
    int level;     // tower layer index for the activation kernels
    int total;     // batch item count for grid-stride build kernels
} Group;

// ------------------------------------------------------------------ helpers

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ int nc_of(const Desc* d, int node, int player) {
    const unsigned int* c = T_cfg_off(d);
    return (int)(c[2 * node + player + 1] - c[2 * node + player]);
}

__device__ __forceinline__ int na_of(const Desc* d, int node) {
    return (int)T_obs_start(d)[T_obs_off(d)[node + 1] - 1];
}

__device__ __forceinline__ int reach_at(const Desc* d, int node, int player, int c) {
    int off = (int)T_reach_off(d)[node];
    if (player == 1) off += nc_of(d, node, 0);
    return off + c;
}

__device__ __forceinline__ bool legal_cell(const Desc* d, int cell) {
    return (T_legal_bits(d)[cell >> 3] >> (cell & 7)) & 1;
}

// The strategy a normal sweep reads: current during CFR, final average during
// fixed-policy value passes. Kept intermediate averages are propagated when
// they are created, so no full-tree strategy snapshots exist.
__device__ __forceinline__ const float* strat_of(const Desc* d) {
    return d->stage == STAGE_ITERATE ? A_cur(d) : A_avg(d);
}

__device__ __forceinline__ bool snapshot_due(const Desc* d) {
    return d->stage == STAGE_ITERATE && d->snapshots
        && d->snap_t < d->nsnaps - 1
        && d->snap_iters[d->snap_t] == d->t + 1;
}

// The root belief a forward sweep seeds: the live root, or each carried root
// in turn during the value stage.
__device__ __forceinline__ const float* root_of(const Desc* d) {
    if (d->stage != STAGE_VALUE) return T_root(d);
    return T_carried(d) + (size_t)d->step * (d->nc_root[0] + d->nc_root[1]);
}

__device__ __forceinline__ int player_of(const Desc* d, const Group* g) {
    return g->p_player < 0 ? d->traverser : g->p_player;
}

// The reach block's total, summed across the warp.
__device__ __forceinline__ void warp_normalize(const float* w, int n, int lane,
                                               float* scale, float* flat) {
    float tot = 0.0f;
    for (int i = lane; i < n; i += 32) tot += w[i];
    tot = warp_sum(tot);
    *scale = tot > SMOOTH ? 1.0f / tot : 0.0f;
    *flat = tot > SMOOTH ? 0.0f : 1.0f / (float)(n > 0 ? n : 1);
}

// Block-per-solve prologue: bind the descriptor and the warp geometry.
#define SOLVE(d)                                                              \
    const Desc* d = descs + g->slots[blockIdx.x];                             \
    if (d->stage == STAGE_DRAIN) return;                                      \
    WC_DESC(d);                                                               \
    int lane = threadIdx.x & 31;                                              \
    int warp = threadIdx.x >> 5;                                              \
    int nwarps = blockDim.x >> 5;                                             \
    (void)lane; (void)warp; (void)nwarps;

// Map a flat task index to its solve and solve-local index. An eight-step
// binary search over the live prefix is cheaper than materialising and
// uploading a task record for every row or node.
__device__ __forceinline__ int flat_owner(const Group* g, int item, int* local) {
    int lo = 0, hi = g->n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (g->prefix[mid + 1] <= item) lo = mid + 1;
        else hi = mid;
    }
    WC_CHECK(lo < g->n, "flat_owner slot", lo, g->n);
    *local = item - g->prefix[lo];
    return g->slots[lo];
}

// Find the entry of a running-count table that owns `item`, searching
// `pfx[lo..hi]` (a sorted prefix, `pfx[hi]` past the end). Returns the entry
// index and leaves the offset inside it in `*within`. This is how a
// one-thread-per-config launch recovers the node it belongs to.
__device__ __forceinline__ int cfg_owner(
    const unsigned int* pfx, int lo, int hi, int item, int* within
) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if ((int)pfx[mid + 1] <= item) lo = mid + 1;
        else hi = mid;
    }
    *within = item - (int)pfx[lo];
    return lo;
}

// LayerNorm + ReLU of one row, split across a warp; `add` may be null.
// Ports net.rs::ln_relu (biased variance, torch's epsilon).
__device__ __forceinline__ void ln_relu_warp(float* row, const float* bias,
                                             const float* gain, const float* bt,
                                             const float* add, int n, int lane) {
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32) {
        float x = row[j] + bias[j];
        if (add) x += add[j];
        row[j] = x;
        sum += x;
    }
    float mean = warp_sum(sum) * (1.0f / n);
    float var = 0.0f;
    for (int j = lane; j < n; j += 32) {
        float e = row[j] - mean;
        var += e * e;
    }
    float inv = rsqrtf(warp_sum(var) * (1.0f / n) + LN_EPS);
    for (int j = lane; j < n; j += 32) {
        float x = (row[j] - mean) * inv * gain[j] + bt[j];
        row[j] = x > 0.0f ? x : 0.0f;
    }
}

// ------------------------------------------------ flattened live-set sweeps

// Clear and seed one solve's reach arena. The level propagation is split into
// `reach_level_flat` launches so each public node becomes an independent
// block instead of one solve block serialising the whole ragged tree.
extern "C" __global__ void reach_seed_flat(
    const Desc* descs, const Group* g, const Ctx* ctx
) {
    SOLVE(d)
    (void)ctx;
    if (g->mode == 2) { if (!snapshot_due(d)) return; }
    else if (g->mode == 1) { if (d->stage != STAGE_ITERATE) return; }
    else { if (d->stage != STAGE_VALUE) return; }
    float* reach = g->mode == 2 ? A_snap_reach(d) : A_reach(d);
    int rlen = (int)T_reach_off(d)[d->nodes];
    for (int i = threadIdx.x; i < rlen; i += blockDim.x) reach[i] = 0.0f;
    __syncthreads();
    const float* root = g->mode == 2 ? T_root(d) : root_of(d);
    int nr = d->nc_root[0] + d->nc_root[1];
    for (int i = threadIdx.x; i < nr; i += blockDim.x) {
        int p = i < d->nc_root[0] ? 0 : 1;
        int c = p == 0 ? i : i - d->nc_root[0];
        reach[reach_at(d, 0, p, c)] = root[i];
    }
}

extern "C" __global__ void reach_level_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int pass
) {
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= g->total) return;
    int slot = 0, local = 0;
    if (lane == 0) slot = flat_owner(g, task, &local);
    slot = __shfl_sync(0xffffffffu, slot, 0);
    local = __shfl_sync(0xffffffffu, local, 0);
    const Desc* d = descs + slot;
    WC_DESC(d);
    (void)ctx;
    if (d->stage == STAGE_DRAIN) return;
    if (pass == 2) { if (!snapshot_due(d)) return; }
    else if (pass == 1) { if (d->stage != STAGE_ITERATE) return; }
    else { if (d->stage != STAGE_VALUE) return; }
    WC_CHECK(g->level < d->nlevels, "reach level", g->level, d->nlevels);
    WC_CHECK(local < (int)(T_level_start(d)[g->level + 1]
                           - T_level_start(d)[g->level]),
             "reach local", local,
             (int)(T_level_start(d)[g->level + 1] - T_level_start(d)[g->level]));
    int at = (int)T_level_start(d)[g->level] + local;
    WC_CHECK(at < d->nodes, "reach at", at, d->nodes);
    int j = (int)T_bfs_order(d)[at];
    WC_CHECK(j < d->nodes, "reach node", j, d->nodes);
    if (j == 0) return;
    int p = (int)T_node_parent(d)[j];
    WC_CHECK(p < d->nodes, "reach parent", p, d->nodes);
    int me = T_node_player(d)[p];
    WC_CHECK(me == 0 || me == 1, "reach me", me, 2);
    float* reach = pass == 2 ? A_snap_reach(d) : A_reach(d);

    // CFR alternates traversers, and a player's counterfactual reach depends
    // only on that player's own decisions (plus fixed chance transitions).
    // After regret matching only the traverser's strategy changed, so update
    // that half in place and leave the opponent's still-current reach alone.
    // The root is unchanged, which also removes the iterate clear/seed pass.
    if (pass == 1) {
        int changed = d->traverser;
        int out_at = reach_at(d, j, changed, 0);
        int parent_at = reach_at(d, p, changed, 0);
        int n = nc_of(d, j, changed);
        WC_CHECK(out_at + n <= A_len_reach(d), "reach1 out", out_at + n,
                 A_len_reach(d));
        WC_CHECK(parent_at + nc_of(d, p, changed) <= A_len_reach(d),
                 "reach1 parent span", parent_at + nc_of(d, p, changed),
                 A_len_reach(d));
        if (me != changed) {
            WC_CHECK(n <= nc_of(d, p, changed), "reach1 copy", n,
                     nc_of(d, p, changed));
            for (int c = lane; c < n; c += 32) reach[out_at + c] = reach[parent_at + c];
        } else if (T_rev_row_of(d)[j] != NONE) {
            int row0 = (int)T_rev_row_of(d)[j];
            WC_CHECK(row0 + n < T_cap_rev_start(d), "reach1 rev row", row0 + n,
                     T_cap_rev_start(d));
            const float* strat = A_cur(d);
            for (int c = lane; c < n; c += 32) {
                int lo = (int)T_rev_start(d)[row0 + c];
                int hi = (int)T_rev_start(d)[row0 + c + 1];
                WC_CHECK(hi <= T_cap_rev_src(d), "reach1 rev nnz", hi,
                         T_cap_rev_src(d));
                float acc = 0.0f;
                for (int k = lo; k < hi; k++) {
                    WC_CHECK((int)T_rev_src(d)[k] < nc_of(d, p, changed),
                             "reach1 src", T_rev_src(d)[k], nc_of(d, p, changed));
                    WC_CHECK((int)T_rev_cell(d)[k] < d->ncells, "reach1 cell",
                             T_rev_cell(d)[k], d->ncells);
                    acc += reach[parent_at + T_rev_src(d)[k]] * strat[T_rev_cell(d)[k]];
                }
                reach[out_at + c] = acc;
            }
        } else {
            int row0 = (int)T_rvd_row_of(d)[j];
            WC_CHECK(T_rvd_row_of(d)[j] != NONE, "reach1 rvd none", j, d->nodes);
            WC_CHECK(row0 + n < T_cap_rvd_start(d), "reach1 rvd row", row0 + n,
                     T_cap_rvd_start(d));
            for (int c = lane; c < n; c += 32) {
                int lo = (int)T_rvd_start(d)[row0 + c];
                int hi = (int)T_rvd_start(d)[row0 + c + 1];
                WC_CHECK(hi <= T_cap_rvd_src(d), "reach1 rvd nnz", hi,
                         T_cap_rvd_src(d));
                float acc = 0.0f;
                for (int k = lo; k < hi; k++) {
                    WC_CHECK((int)T_rvd_src(d)[k] < nc_of(d, p, changed),
                             "reach1 rvd src", T_rvd_src(d)[k],
                             nc_of(d, p, changed));
                    acc += reach[parent_at + T_rvd_src(d)[k]] * T_rvd_p(d)[k];
                }
                reach[out_at + c] = acc;
            }
        }
        return;
    }

    int op = 1 - me;
    int me_at = reach_at(d, j, me, 0);
    int op_at = reach_at(d, j, op, 0);
    int pme_at = reach_at(d, p, me, 0);
    int pop_at = reach_at(d, p, op, 0);
    int nop = nc_of(d, j, op);
    for (int c = lane; c < nop; c += 32) {
        reach[op_at + c] = reach[pop_at + c];
    }
    int nme = nc_of(d, j, me);
    if (T_rev_row_of(d)[j] != NONE) {
        int row0 = (int)T_rev_row_of(d)[j];
        const float* strat = pass == 2 ? A_avg(d) : strat_of(d);
        for (int c = lane; c < nme; c += 32) {
            int lo = (int)T_rev_start(d)[row0 + c];
            int hi = (int)T_rev_start(d)[row0 + c + 1];
            float acc = 0.0f;
            for (int k = lo; k < hi; k++) {
                WC_CHECK((int)T_rev_src(d)[k] < nc_of(d, p, me), "reach src",
                         T_rev_src(d)[k], nc_of(d, p, me));
                WC_CHECK((int)T_rev_cell(d)[k] < d->ncells, "reach cell",
                         T_rev_cell(d)[k], d->ncells);
                acc += reach[pme_at + T_rev_src(d)[k]] * strat[T_rev_cell(d)[k]];
            }
            reach[me_at + c] = acc;
        }
    } else {
        int row0 = (int)T_rvd_row_of(d)[j];
        for (int c = lane; c < nme; c += 32) {
            int lo = (int)T_rvd_start(d)[row0 + c];
            int hi = (int)T_rvd_start(d)[row0 + c + 1];
            float acc = 0.0f;
            for (int k = lo; k < hi; k++) {
                acc += reach[pme_at + T_rvd_src(d)[k]] * T_rvd_p(d)[k];
            }
            reach[me_at + c] = acc;
        }
    }
}

// One warp per network leaf, spread across the whole live set. Most iterate
// ticks need one player's projection; the first query and value stage compute
// both players sequentially in the same warp instead of launching a grid with
// half of its warps returning.
extern "C" __global__ void belief_sums_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int unused
) {
    (void)unused;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= g->total) return;
    int slot = 0, local = 0;
    if (lane == 0) slot = flat_owner(g, task, &local);
    slot = __shfl_sync(0xffffffffu, slot, 0);
    local = __shfl_sync(0xffffffffu, local, 0);
    const Desc* d = descs + slot;
    WC_DESC(d);
    if (d->stage == STAGE_CARRY || d->stage == STAGE_DRAIN) return;
    int row = local;
    int both = (d->stage == STAGE_VALUE) || d->first_query;
    int np = both ? 2 : 1;
    for (int side = 0; side < np; side++) {
        int p = both ? side : 1 - d->traverser;
        int leaf = (int)T_leaf_rows(d)[row];
        int n = nc_of(d, leaf, p);
        const float* r = A_reach(d) + reach_at(d, leaf, p, 0);
        float scale, flat;
        warp_normalize(r, n, lane, &scale, &flat);
        float acc[DG_CH];
        for (int k = 0; k < DG_CH; k++) acc[k] = 0.0f;
        int c0 = (int)T_leaf_coff(d)[2 * row + p];
        for (int c = 0; c < n; c++) {
            int ci = (int)T_leaf_cidx(d)[c0 + c];
            WC_CHECK(ci < d->ncfg, "belief cfg", ci, d->ncfg);
            float wc = r[c] * scale + flat;
            const float* z = A_z(d) + (size_t)ci * DG;
            for (int k = 0; k < DG_CH; k++) {
                int j = (k << 5) + lane;
                if (j < DG) acc[k] += wc * z[j];
            }
        }
        float* out = ctx->xb + ((size_t)(d->row0 + row) * 2 + p) * DG;
        for (int k = 0; k < DG_CH; k++) {
            int j = (k << 5) + lane;
            if (j < DG) out[j] = acc[k];
        }
    }
}

// One warp per network row for the LayerNorm entry activation.
extern "C" __global__ void head_entry_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int unused
) {
    (void)unused;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= g->total) return;
    int slot = 0, row = 0;
    if (lane == 0) slot = flat_owner(g, task, &row);
    slot = __shfl_sync(0xffffffffu, slot, 0);
    row = __shfl_sync(0xffffffffu, row, 0);
    const Desc* d = descs + slot;
    WC_DESC(d);
    if (d->stage == STAGE_CARRY || d->stage == STAGE_DRAIN) return;
    // HEADW is compile-time, so keep this warp's row fragment in registers.
    // The generic LayerNorm helper spills the biased residual to `h`, then
    // reads it for variance and again for the output.  At tens of thousands
    // of rows per tick that extra write and two reads saturate DRAM.  This is
    // the same per-lane accumulation and warp reduction order, with a single
    // final store.
    float* dst = ctx->h + (size_t)(d->row0 + row) * H_STRIDE;
    const float* add = ctx->h0 + (size_t)(d->row0 + row) * HEADW;
    float x[HEAD_CH];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < HEAD_CH; k++) {
        int j = (k << 5) + lane;
        float v = 0.0f;
        if (j < HEADW) {
            v = dst[j] + ctx->pub_out_b[j];
            v += add[j];
            sum += v;
        }
        x[k] = v;
    }
    float mean = warp_sum(sum) * (1.0f / HEADW);
    float var = 0.0f;
    #pragma unroll
    for (int k = 0; k < HEAD_CH; k++) {
        int j = (k << 5) + lane;
        if (j < HEADW) {
            float e = x[k] - mean;
            var += e * e;
        }
    }
    float inv = rsqrtf(warp_sum(var) * (1.0f / HEADW) + LN_EPS);
    #pragma unroll
    for (int k = 0; k < HEAD_CH; k++) {
        int j = (k << 5) + lane;
        if (j < HEADW) {
            float v = (x[k] - mean) * inv * ctx->ln1w[j] + ctx->ln1b[j];
            dst[j] = v > 0.0f ? v : 0.0f;
        }
    }
}

// One warp per leaf. Typical trained-play leaves have far fewer than the 256
// configurations a whole block provides threads for; mapping eight leaves per
// block keeps those small readouts resident together while preserving each
// config dot product's lane reduction order.
extern "C" __global__ void readout_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int p_player
) {
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= g->total) return;
    int slot = 0, rp = 0;
    if (lane == 0) slot = flat_owner(g, task, &rp);
    slot = __shfl_sync(0xffffffffu, slot, 0);
    rp = __shfl_sync(0xffffffffu, rp, 0);
    const Desc* d = descs + slot;
    WC_DESC(d);
    if (d->stage == STAGE_DRAIN) return;
    if (p_player < 0) { if (d->stage != STAGE_ITERATE) return; }
    else              { if (d->stage != STAGE_VALUE) return; }
    int p = p_player < 0 ? d->traverser : p_player;
    int opp = 1 - p;
    WC_CHECK(rp < d->nleaf + d->nterm, "readout row", rp, d->nleaf + d->nterm);
    int leaf = rp < d->nleaf ? (int)T_leaf_rows(d)[rp]
                             : (int)T_term_leaves(d)[rp - d->nleaf];
    WC_CHECK(leaf < d->nodes, "readout leaf", leaf, d->nodes);
    int nop = nc_of(d, leaf, opp);
    const float* ro = A_reach(d) + reach_at(d, leaf, opp, 0);
    float opp_reach = 0.0f;
    for (int c = lane; c < nop; c += 32) opp_reach += ro[c];
    opp_reach = warp_sum(opp_reach);
    int n = nc_of(d, leaf, p);
    float* v = A_vals(d) + T_voff(d)[leaf];
    if (rp >= d->nleaf) {
        float ut = T_terminal_utility(d)[rp - d->nleaf];
        if (T_node_player(d)[leaf] != p) ut = -ut;
        float val = ut * opp_reach;
        for (int c = lane; c < n; c += 32) v[c] = val;
        return;
    }
    const float* u = ctx->u + (size_t)(d->row0 + rp) * RK;
    float ur[RK_CH];
    for (int k = 0; k < RK_CH; k++) {
        int j = (k << 5) + lane;
        ur[k] = j < RK ? u[j] + ctx->wu_b[j] : 0.0f;
    }
    int c0 = (int)T_leaf_coff(d)[2 * rp + p];
    for (int c = 0; c < n; c++) {
        int ci = (int)T_leaf_cidx(d)[c0 + c];
        WC_CHECK(ci < d->ncfg, "readout cfg", ci, d->ncfg);
        const float* gr = A_g(d) + (size_t)ci * (RK + 1);
        float part = 0.0f;
        for (int k = 0; k < RK_CH; k++) {
            int j = (k << 5) + lane;
            if (j < RK) part += ur[k] * gr[j];
        }
        part = warp_sum(part);
        if (lane == 0) v[c] = (part + gr[RK]) * opp_reach;
    }
}

// One thread per (node of this level, traverser config). Same reasoning as
// regret matching: the per-config work is independent, and node config counts
// span three orders of magnitude inside one tree.
extern "C" __global__ void backprop_level_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int p_player
) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= g->total) return;
    int local = 0;
    int slot = flat_owner(g, task, &local);
    const Desc* d = descs + slot;
    WC_DESC(d);
    (void)ctx;
    if (d->stage == STAGE_DRAIN) return;
    int mode;
    if (p_player < 0) {
        if (d->stage != STAGE_ITERATE) return;
        mode = 0;
    } else {
        if (d->stage != STAGE_VALUE) return;
        mode = 1;
    }
    int traverser = p_player < 0 ? d->traverser : p_player;
    const unsigned int* pfx = traverser ? T_sweep_cfg1(d) : T_sweep_cfg0(d);
    int lo = (int)T_sweep_level_start(d)[g->level];
    int hi = (int)T_sweep_level_start(d)[g->level + 1];
    int c = 0;
    int at = cfg_owner(pfx, lo, hi, (int)pfx[lo] + local, &c);
    if (at >= hi) return;
    int i = (int)T_sweep_order(d)[at];
    WC_CHECK(i < d->nodes, "backprop node", i, d->nodes);
    int kind = T_node_kind(d)[i];
    int me = T_node_player(d)[i];
    const unsigned int* child_start = T_node_child_start(d);
    const unsigned int* child = T_node_child(d);
    const unsigned int* voff = T_voff(d);
    float* vals = A_vals(d);
    int vbase = (int)voff[i];
    if (kind == 1) {
        int ch = (int)child[child_start[i]];
        const float* src = vals + voff[ch];
        if (me == traverser) {
            int d0 = (int)T_draw_off(d)[i];
            const unsigned int* b = T_draw_row_start(d) + T_draw_row_off(d)[i];
            float acc = 0.0f;
            for (int k = (int)b[c]; k < (int)b[c + 1]; k++) {
                WC_CHECK((int)T_draw_to(d)[d0 + k] < nc_of(d, ch, traverser),
                         "backprop draw", T_draw_to(d)[d0 + k],
                         nc_of(d, ch, traverser));
                acc += T_draw_p(d)[d0 + k] * src[T_draw_to(d)[d0 + k]];
            }
            vals[vbase + c] = acc;
        } else {
            vals[vbase + c] = src[c];
        }
        return;
    }
    if (me != traverser) {
        float acc = 0.0f;
        for (unsigned int ci = child_start[i]; ci < child_start[i + 1]; ci++) {
            acc += vals[voff[child[ci]] + c];
        }
        vals[vbase + c] = acc;
        return;
    }
    int na = na_of(d, i);
    int so = (int)T_soff(d)[i] + c * na;
    WC_CHECK(c < nc_of(d, i, traverser), "backprop cfg", c,
             nc_of(d, i, traverser));
    WC_CHECK(so + na <= d->ncells, "backprop cells", so + na, d->ncells);
    const float* strat = strat_of(d);
    float* inst = A_inst(d);
    float acc = 0.0f;
    for (int a = 0; a < na; a++) {
        int cell = so + a;
        if (mode == 0) inst[cell] = 0.0f;
        if (!legal_cell(d, cell)) continue;
        WC_CHECK(cell < T_cap_trans(d), "backprop trans cap", cell,
                 T_cap_trans(d));
        int tr = T_trans(d)[cell];
        if (tr < 0) continue;
        int ch = (int)child[child_start[i] + T_obs_child(d)[T_act_off(d)[i] + a]];
        WC_CHECK(ch < d->nodes, "backprop child", ch, d->nodes);
        WC_CHECK(tr < nc_of(d, ch, traverser), "backprop trans", tr,
                 nc_of(d, ch, traverser));
        float av = vals[voff[ch] + tr];
        if (mode == 0) inst[cell] = av;
        acc += av * strat[cell];
    }
    vals[vbase + c] = acc;
    if (mode == 0) {
        for (int a = 0; a < na; a++) {
            if (legal_cell(d, so + a)) inst[so + a] -= acc;
        }
    }
}

// One *thread* per (traverser decision node, config). A config's action loop
// needs no cross-lane reduction, and the config counts are wildly uneven — a
// subgame root can hold hundreds while a node two levels down holds one — so a
// warp per node left a single lane grinding through the root while the rest of
// the launch had already finished. `T_dec_cfg*` is the running config count
// along the node list, and the binary search over it is what turns a flat
// thread index back into (node, config).
extern "C" __global__ void regret_match_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int unused
) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= g->total) return;
    int local = 0;
    int slot = flat_owner(g, task, &local);
    const Desc* d = descs + slot;
    WC_DESC(d);
    (void)ctx; (void)unused;
    if (d->stage != STAGE_ITERATE) return;
    int tr = d->traverser;
    int c = 0;
    int ni = cfg_owner(tr ? T_dec_cfg1(d) : T_dec_cfg0(d), 0, d->ndec[tr], local, &c);
    // The launch size comes from the host's copy of the same running counts.
    // If the two ever disagreed this would index past the node list, so make
    // the disagreement a dropped thread (which the oracle test sees as a wrong
    // answer) rather than a stray write into another solve's arena.
    if (ni >= d->ndec[tr]) return;
    int i = tr ? (int)T_decision1(d)[ni] : (int)T_decision0(d)[ni];
    WC_CHECK(i < d->nodes, "regret node", i, d->nodes);
    int na = na_of(d, i);
    int so = (int)T_soff(d)[i] + c * na;
    WC_CHECK(so + na <= d->ncells, "regret cell", so + na, d->ncells);
    float* regret = A_regret(d);
    float* inst = A_inst(d);
    float* cur = A_cur(d);
    float* sum_strat = A_sum_strat(d);
    float sum = 0.0f;
    for (int a = 0; a < na; a++) {
        int cell = so + a;
        if (!legal_cell(d, cell)) { cur[cell] = 0.0f; continue; }
        float r = regret[cell] * (regret[cell] > 0.0f ? d->da : d->db) + inst[cell];
        regret[cell] = r;
        float v = fmaxf(r + d->predict * inst[cell], EPS);
        cur[cell] = v;
        sum += v;
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) cur[so + a] *= inv;
    }
    for (int a = 0; a < na; a++) sum_strat[so + a] *= d->ds;
}

extern "C" __global__ void average_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int unused
) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= g->total) return;
    int local = 0;
    int slot = flat_owner(g, task, &local);
    const Desc* d = descs + slot;
    WC_DESC(d);
    (void)ctx; (void)unused;
    if (d->stage != STAGE_ITERATE) return;
    int tr = d->traverser;
    int c = 0;
    int ni = cfg_owner(tr ? T_dec_cfg1(d) : T_dec_cfg0(d), 0, d->ndec[tr], local, &c);
    if (ni >= d->ndec[tr]) return;
    int i = tr ? (int)T_decision1(d)[ni] : (int)T_decision0(d)[ni];
    WC_CHECK(i < d->nodes, "average node", i, d->nodes);
    int na = na_of(d, i);
    int so = (int)T_soff(d)[i] + c * na;
    WC_CHECK(so + na <= d->ncells, "average cell", so + na, d->ncells);
    float* sum_strat = A_sum_strat(d);
    float* avg = A_avg(d);
    const float* cur = A_cur(d);
    float rc = A_reach(d)[reach_at(d, i, tr, c)];
    float sum = 0.0f;
    for (int a = 0; a < na; a++) {
        sum_strat[so + a] += rc * cur[so + a];
        sum += sum_strat[so + a];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) avg[so + a] = sum_strat[so + a] * inv;
    } else {
        int k = 0;
        for (int a = 0; a < na; a++) if (legal_cell(d, so + a)) k++;
        float u = 1.0f / (float)(k > 0 ? k : 1);
        for (int a = 0; a < na; a++) {
            avg[so + a] = legal_cell(d, so + a) ? u : 0.0f;
        }
    }
}

// ------------------------------------------------------- belief sums

// ------------------------------------------------------- the head

// relu(x + b) over one extra head layer's rows. `g->level` names the layer;
// `g->mode` picks which pool the GEMM wrote (0 = h, 1 = h2). Only present
// when the checkpoint has extra head layers.
extern "C" __global__ void hmlp_act(const Desc* descs, const Group* g, const Ctx* ctx) {
    SOLVE(d)
    if (d->stage == STAGE_CARRY) return;
    int wdt = HMLPW[g->level];
    const float* b = ctx->hmlp_b[g->level];
    float* pool = g->mode ? ctx->h2 : ctx->h;
    for (int i = threadIdx.x; i < d->nleaf * wdt; i += blockDim.x) {
        size_t at = (size_t)(d->row0 + i / wdt) * H_STRIDE + i % wdt;
        float v = pool[at] + b[i % wdt];
        pool[at] = v > 0.0f ? v : 0.0f;
    }
}

// ------------------------------------------------------- readout

// ------------------------------------------------------- backward sweep

// ------------------------------------------------------- regret matching

// ------------------------------------------------------- forward reach sweep

// Clear, seed, and propagate, all inside one block: levels are ordered by
// block barriers and every output config has exactly one writer, because the
// child *gathers* from its parent through the reverse tables (in the CPU's
// exact accumulation order). Ports Solver::propagate.
//
// g->mode picks who runs: 0 = value and carry solves (their pass starts by
// re-seeding), 1 = iterate solves (reach follows regret matching).
extern "C" __global__ void reach_prop(const Desc* descs, const Group* g, const Ctx* ctx) {
    SOLVE(d)
    if (g->mode == 1) { if (d->stage != STAGE_ITERATE) return; }
    else              { if (d->stage == STAGE_ITERATE) return; }
    if (d->stage == STAGE_CARRY && d->leaf < 0) return;
    float* reach = A_reach(d);
    int rlen = (int)(d->aoff[1] - d->aoff[0]);
    for (int i = threadIdx.x; i < rlen; i += blockDim.x) reach[i] = 0.0f;
    __syncthreads();
    const float* root = root_of(d);
    int nr = d->nc_root[0] + d->nc_root[1];
    for (int i = threadIdx.x; i < nr; i += blockDim.x) {
        int p = i < d->nc_root[0] ? 0 : 1;
        int c = p == 0 ? i : i - d->nc_root[0];
        reach[reach_at(d, 0, p, c)] = root[i];
    }
    __syncthreads();
    const float* strat = strat_of(d);
    const unsigned int* parent = T_node_parent(d);
    for (int lev = 1; lev < d->nlevels; lev++) {
        int l0 = (int)T_level_start(d)[lev];
        int l1 = (int)T_level_start(d)[lev + 1];
        for (int at = l0 + warp; at < l1; at += nwarps) {
            int j = (int)T_bfs_order(d)[at];
            int p = (int)parent[j];
            int me = T_node_player(d)[p];
            int op = 1 - me;
            int me_at = reach_at(d, j, me, 0);
            int op_at = reach_at(d, j, op, 0);
            int pme_at = reach_at(d, p, me, 0);
            int pop_at = reach_at(d, p, op, 0);
            int nop = nc_of(d, j, op);
            for (int c = lane; c < nop; c += 32) reach[op_at + c] = reach[pop_at + c];
            int nme = nc_of(d, j, me);
            if (T_rev_row_of(d)[j] != NONE) {
                int row0 = (int)T_rev_row_of(d)[j];
                for (int c = lane; c < nme; c += 32) {
                    int lo = (int)T_rev_start(d)[row0 + c];
                    int hi = (int)T_rev_start(d)[row0 + c + 1];
                    float acc = 0.0f;
                    for (int k = lo; k < hi; k++) {
                        acc += reach[pme_at + T_rev_src(d)[k]] * strat[T_rev_cell(d)[k]];
                    }
                    reach[me_at + c] = acc;
                }
            } else {
                int row0 = (int)T_rvd_row_of(d)[j];
                for (int c = lane; c < nme; c += 32) {
                    int lo = (int)T_rvd_start(d)[row0 + c];
                    int hi = (int)T_rvd_start(d)[row0 + c + 1];
                    float acc = 0.0f;
                    for (int k = lo; k < hi; k++) {
                        acc += reach[pme_at + T_rvd_src(d)[k]] * T_rvd_p(d)[k];
                    }
                    reach[me_at + c] = acc;
                }
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------- average strategy

// ------------------------------------------------------- harvests

// The value stage's harvest: carried root `step`'s per-config values, per
// player, into the solve's own arena. Runs after that player's backprop.
extern "C" __global__ void collect_root(const Desc* descs, const Group* g, const Ctx* ctx) {
    SOLVE(d)
    if (d->stage != STAGE_VALUE) return;
    int p = g->p_player;
    int n = d->nc_root[p];
    int base = p == 1 ? d->nc_root[0] : 0;
    int stride = d->nc_root[0] + d->nc_root[1];
    float* out = A_root_vals(d) + (size_t)d->step * stride + base;
    const float* v = A_vals(d) + T_voff(d)[0];
    for (int c = threadIdx.x; c < n; c += blockDim.x) out[c] = v[c];
}

// A kept intermediate average has just been propagated into `snap_reach`.
// Store its normalised belief at every possible exit leaf. Trip 2 then reads
// only the row the real walk chose; no full-tree strategy replay is needed.
extern "C" __global__ void snapshot_beliefs_flat(
    const Desc* descs, const Group* g, const Ctx* ctx, int unused
) {
    (void)ctx; (void)unused;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= g->total) return;
    int slot = 0, row = 0;
    if (lane == 0) slot = flat_owner(g, task, &row);
    slot = __shfl_sync(0xffffffffu, slot, 0);
    row = __shfl_sync(0xffffffffu, row, 0);
    const Desc* d = descs + slot;
    WC_DESC(d);
    if (!snapshot_due(d)) return;
    WC_CHECK(row < d->nleaf + d->nterm, "snap row", row, d->nleaf + d->nterm);
    int leaf = row < d->nleaf ? (int)T_leaf_rows(d)[row]
                              : (int)T_term_leaves(d)[row - d->nleaf];
    WC_CHECK(leaf < d->nodes, "snap leaf", leaf, d->nodes);
    WC_CHECK(d->snap_t < d->nsnaps - 1, "snap slot", d->snap_t, d->nsnaps - 1);
    int stride = (int)T_snap_coff(d)[2 * (d->nleaf + d->nterm)];
    for (int p = 0; p < 2; p++) {
        int n = nc_of(d, leaf, p);
        const float* r = A_snap_reach(d) + reach_at(d, leaf, p, 0);
        float scale, flat;
        warp_normalize(r, n, lane, &scale, &flat);
        int off = (int)T_snap_coff(d)[2 * row + p];
        float* o = A_snap_beliefs(d) + (size_t)d->snap_t * stride + off;
        for (int c = lane; c < n; c += 32) o[c] = r[c] * scale + flat;
    }
}

// Trip 2 selects one exit leaf's two config spans from each kept snapshot.
// Gather the strided source into one contiguous device row so the downloader
// issues one pinned D2H transfer rather than two tiny transfers per snapshot.
extern "C" __global__ void gather_trip2(
    unsigned long long src_addr, unsigned long long dst_addr,
    int leaf_configs, int off0, int off1,
    int n0, int n1, int nsnaps
) {
    const float* src = (const float*)src_addr;
    float* dst = (float*)dst_addr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = n0 + n1;
    int total = nsnaps * stride;
    if (i >= total) return;
    int s = i / stride;
    int c = i - s * stride;
    int off = c < n0 ? off0 + c : off1 + c - n0;
    dst[i] = src[(size_t)s * leaf_configs + off];
}

// ------------------------------------------------------- state advance

// One thread per live solve: the whole per-tick state machine, on the
// device, so the host never re-uploads descriptors. Mirrors the host's
// bookkeeping exactly (service.rs::advance); the host runs the same
// arithmetic on its shadow copies to know when trips are due.
extern "C" __global__ void advance_state(const Desc* descs_c, const Group* g, const Ctx* ctx) {
    Desc* descs = (Desc*)descs_c;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= g->n) return;
    Desc* d = descs + g->slots[k];
    if (d->stage == STAGE_DRAIN) return;
    if (d->stage == STAGE_ITERATE) {
        d->first_query = 0;
        d->steps[d->traverser] += 1;
        // Snapshot bookkeeping matches `average`'s due test.
        if (d->snapshots) {
            for (int s = 0; s < d->nsnaps; s++) {
                if (d->snap_iters[s] == d->t + 1) d->snap_t += 1;
            }
        }
        d->t += 1;
        d->traverser = d->t & 1;
        if (d->t == d->iters) {
            d->step = 0;
            d->stage = d->nroots > 0 ? STAGE_VALUE : STAGE_CARRY;
        }
        // The next iteration's DCFR discounts, once, instead of a powf per
        // lane per node: factor(m) = m^p / (m^p + 1), linear when p = inf.
        float m = (float)(d->steps[d->traverser] + 1);
        float pa = powf(m, d->alpha), pb = powf(m, d->beta);
        d->da = isinf(d->alpha) ? 1.0f : pa / (pa + 1.0f);
        d->db = isinf(d->beta) ? (d->beta > 0.0f ? 1.0f : 0.0f) : pb / (pb + 1.0f);
        d->ds = powf(m / (m + 1.0f), d->gamma);
    } else if (d->stage == STAGE_VALUE) {
        d->step += 1;
        if (d->step >= d->nroots) {
            d->step = 0;
            d->stage = STAGE_CARRY;
        }
    } else if (d->leaf >= 0) {
        d->step += 1;
    }
}

// ------------------------------------------------------- admission (build)

// Grid-stride helpers over the admission batch: find the batch entry that
// owns a flat item index by scanning the (tiny) batch map.
// The batch map holds at most `MAX_BATCH` entries. Walking past the last one
// reads whatever follows it and turns into `descs[garbage]`, so the walk is
// bounded: a launch whose task count disagrees with the entries' own counts
// is a host bug, and it should say so rather than wander off the buffer.
__device__ __forceinline__ const int* bmap_of(const Ctx* ctx, int item, int per, int* local) {
    // per = 1: rows; per = 0: configs.
    int start = item;
    for (int e = 0; e < BMAP_CAP; e++) {
        const int* m = ctx->bmap + e * BMAP_INTS;
        int n = per ? m[2] : m[4];
        if (item < n) { *local = item; return m; }
        item -= n;
    }
    WC_SAY("bmap_of item", start, BMAP_CAP);
    *local = 0;
    return ctx->bmap;
}

// As above, but network leaves rather than all public rows (warm-start rows
// may follow them in a solve's batch).
__device__ __forceinline__ const int* bmap_leaf_of(
    const Desc* descs, const Ctx* ctx, int item, int* local
) {
    int start = item;
    for (int e = 0; e < BMAP_CAP; e++) {
        const int* m = ctx->bmap + e * BMAP_INTS;
        int n = descs[m[0]].nleaf + descs[m[0]].nterm;
        if (item < n) { *local = item; return m; }
        item -= n;
    }
    WC_SAY("bmap_leaf_of item", start, BMAP_CAP);
    *local = 0;
    return ctx->bmap;
}

// Snapshot 0 is the initial uniform average, whose reach has already been
// built for CFR initialisation. Harvest every possible exit leaf now.
extern "C" __global__ void seed_snapshot_beliefs(
    const Desc* descs, const Group* g, const Ctx* ctx
) {
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= g->total) return;
    int row;
    const int* m = bmap_leaf_of(descs, ctx, task, &row);
    WC_CHECK(m[0] >= 0 && m[0] < SLOT_CAP, "seed slot", m[0], SLOT_CAP);
    const Desc* d = descs + m[0];
    WC_DESC(d);
    WC_CHECK(row < d->nleaf + d->nterm, "seed leaf row", row,
             d->nleaf + d->nterm);
    if (!d->snapshots || d->nsnaps <= 1) return;
    int leaf = row < d->nleaf ? (int)T_leaf_rows(d)[row]
                              : (int)T_term_leaves(d)[row - d->nleaf];
    WC_CHECK(leaf < d->nodes, "seed leaf", leaf, d->nodes);
    for (int p = 0; p < 2; p++) {
        int n = nc_of(d, leaf, p);
        int at = reach_at(d, leaf, p, 0);
        WC_CHECK(at + n <= A_len_reach(d), "seed reach", at + n,
                 A_len_reach(d));
        const float* r = A_reach(d) + at;
        float scale, flat;
        warp_normalize(r, n, lane, &scale, &flat);
        int off = (int)T_snap_coff(d)[2 * row + p];
        WC_CHECK(off + n <= A_len_snap_beliefs(d), "seed beliefs", off + n,
                 A_len_snap_beliefs(d));
        float* o = A_snap_beliefs(d) + off;
        for (int c = lane; c < n; c += 32) o[c] = r[c] * scale + flat;
    }
}

// The card facts of every job in the batch, packed for one GEMM chain:
// bg[(job * NTYPE + t)][CARD_FEATS]. The facts block is identical at every
// row of a job. The compact contract stores it once per solve.
extern "C" __global__ void pack_cards(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = NTYPE * CARD_FEATS;
    if (i >= g->n * per) return;
    const int* m = ctx->bmap + (i / per) * BMAP_INTS;
    const Desc* d = descs + m[0];
    WC_DESC(d);
    ctx->bg[(size_t)i] = T_card_feat(d)[i % per];
}

// Generic bias(+ReLU) over a build scratch matrix. g->mode picks the tower
// (0 card, 1 pub, 2 slot, 3 res-a, 4 res-b: bias only), g->level the layer,
// g->total the row count. The matrix is bh or bh2 by g->p_player.
extern "C" __global__ void bias_act(const Desc* descs, const Group* g, const Ctx* ctx) {
    int wdt;
    const float* b;
    int rl = 1;
    switch (g->mode) {
        case 0: wdt = CARDW[g->level]; b = ctx->card_b[g->level]; break;
        case 1: wdt = PUBW[g->level]; b = ctx->pub_b[g->level]; break;
        case 2: wdt = SLOTW[g->level]; b = ctx->slot_b[g->level]; break;
        case 3: wdt = DG; b = ctx->res_ab[g->level]; break;
        default: wdt = DG; b = ctx->res_bb[g->level]; rl = 0; break;
    }
    float* x = g->p_player ? ctx->bh2 : ctx->bh;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g->total * wdt) return;
    float v = x[i] + b[i % wdt];
    x[i] = rl && v < 0.0f ? 0.0f : v;
}

// e = (card chain output) + bd_last + wid[id], written to each solve's own
// card table. `mode` names the ping-pong buffer the last GEMM wrote.
extern "C" __global__ void cards_finish(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g->n * NTYPE * DE) return;
    const int* m = ctx->bmap + i / (NTYPE * DE) * BMAP_INTS;
    Desc* d = (Desc*)descs + m[0];
    WC_DESC(d);
    int t = (i / DE) % NTYPE, j = i % DE;
    int id = T_ids(d)[t];
    const float* x = g->mode ? ctx->bh2 : ctx->bh;
    A_e(d)[t * DE + j] = x[i] + ctx->card_b[NCARD - 1][j] + ctx->wid[id * DE + j];
}

// The card half of the pile summary, per job per coin type, into the tail of
// bg (after the batch's per-row count block). pe = bpile + e . Wpile[counts:].
extern "C" __global__ void pile_pe(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g->n * NTYPE * DE) return;
    const int* m = ctx->bmap + i / (NTYPE * DE) * BMAP_INTS;
    const Desc* d = descs + m[0];
    WC_DESC(d);
    int t = (i / DE) % NTYPE, j = i % DE;
    float acc = ctx->pile_b[j];
    const float* er = A_e(d) + (size_t)t * DE;
    for (int k = 0; k < DE; k++) acc += er[k] * ctx->pile_w[((size_t)PILE_COUNTS + k) * DE + j];
    ctx->bg[(size_t)g->total * NTYPE * DE + (size_t)i] = acc;
}

// Assemble one batch row of the trunk input x[XD] from the job's stored
// row, its card table and the pile blocks. One block per row.
// Ports net.rs::assemble.
extern "C" __global__ void assemble(const Desc* descs, const Group* g, const Ctx* ctx) {
    if (blockIdx.x >= g->total) return;
    int local;
    const int* m = bmap_of(ctx, blockIdx.x, 1, &local);
    const Desc* d = descs + m[0];
    WC_DESC(d);
    int batch_row = m[1] + local;
    const unsigned char* src = T_leaf_raw(d) + (size_t)local * GPU_ROW_BYTES;
    float* dst = ctx->bx + (size_t)batch_row * XD;
    int hex_e = N_HEXES * HEX_FACTS;
    int piles = hex_e + N_HEXES * DE;
    for (int j = threadIdx.x; j < XD; j += blockDim.x) dst[j] = 0.0f;
    __syncthreads();
    // Hex facts and the occupant's embedding (a one-hot gather).
    for (int h = threadIdx.x / 32; h < N_HEXES; h += blockDim.x / 32) {
        int lane = threadIdx.x & 31;
        int owner = src[GR_HEX_OWNER + h];
        int slot = src[GR_HEX_SLOT + h];
        if (lane < HEX_FACTS) {
            float v = 0.0f;
            if (lane < 2) v = owner == lane ? 1.0f : 0.0f;
            else if (lane == 2) v = (float)src[GR_HEX_HEIGHT + h] / 5.0f;
            else if (lane < 5) v = src[GR_HEX_MARKER + h] == lane - 3 ? 1.0f : 0.0f;
            else v = (float)HEX_LOCATION[h];
            dst[h * HEX_FACTS + lane] = v;
        }
        int t = owner < 2 && slot < NSLOT ? owner * NSLOT + slot : -1;
        if (t >= 0) {
            const float* e_row = A_e(d) + (size_t)t * DE;
            for (int j = lane; j < DE; j += 32) dst[hex_e + h * DE + j] = e_row[j];
        }
    }
    __syncthreads();
    // Pile summary: relu(count half + card half), summed per player.
    const float* ph = ctx->bh + (size_t)batch_row * NTYPE * DE;
    // pe rows are per (job, type), in the tail of bg past the count block.
    int job = (int)((m - ctx->bmap) / BMAP_INTS);
    const float* pe = ctx->bg + (size_t)g->total * NTYPE * DE + (size_t)job * NTYPE * DE;
    for (int t = 0; t < NTYPE; t++) {
        float* acc = dst + piles + (t / NSLOT) * DE;
        for (int j = threadIdx.x; j < DE; j += blockDim.x) {
            float v = ph[t * DE + j] + pe[t * DE + j];
            if (v > 0.0f) acc[j] += v;
        }
        __syncthreads();
    }
    float* loose = dst + piles + 2 * DE;
    for (int j = threadIdx.x; j < LOOSE; j += blockDim.x) {
        float v = 0.0f;
        if (j < 2 * 6) {
            int p = j / 6, k = j % 6;
            int markers = src[GR_MARKERS + p];
            if (k == 0) v = (float)markers / 6.0f;
            else if (k == 1) v = (float)(6 - markers) / 6.0f;
            else if (k == 2) v = (float)src[GR_HAND + p] / 3.0f;
            else if (k == 3) v = (float)src[GR_FD + p] / MAX_COINS;
            else if (k == 4) v = (float)src[GR_BAG + p] / MAX_COINS;
            else v = src[GR_INITIATIVE] == p ? 1.0f : 0.0f;
        } else if (j == 12) {
            int plies = src[GR_PLIES] | ((int)src[GR_PLIES + 1] << 8);
            v = (float)plies / MAX_PLIES;
        } else if (j == 13) {
            v = src[GR_INIT_MOVED] ? 1.0f : 0.0f;
        } else {
            v = src[GR_TO_ACT] == 0 ? 1.0f : 0.0f;
        }
        loose[j] = v;
    }
}

// The per-row count block of the pile summary, packed for one batched GEMM:
// bh[(batch_row * NTYPE + t)][PILE_COUNTS] <- row's pile counts.
extern "C" __global__ void pack_piles(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = NTYPE * PILE_COUNTS;
    if (i >= g->total * per) return;
    int local;
    const int* m = bmap_of(ctx, i / per, 1, &local);
    const Desc* d = descs + m[0];
    WC_DESC(d);
    ctx->bg[(size_t)i] = (float)T_leaf_raw(d)[(size_t)local * GPU_ROW_BYTES
                                               + GR_PILES + i % per] / 5.0f;
}

// LN + ReLU over one public tower layer's batch rows. g->level names the
// layer; the matrix is bh or bh2 by g->p_player. Warp per row.
extern "C" __global__ void trunk_norm(const Desc* descs, const Group* g, const Ctx* ctx) {
    int wdt = PUBW[g->level];
    float* x = g->p_player ? ctx->bh2 : ctx->bh;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= g->total) return;
    ln_relu_warp(x + (size_t)row * wdt, ctx->pub_b[g->level],
                 ctx->pub_lnw[g->level], ctx->pub_lnb[g->level], 0, wdt, lane);
}

// Scatter the batch's h0 rows (the pub_out GEMM's output, in bh/bh2) into
// each solve's stable rows of the h0 pool.
extern "C" __global__ void scatter_h0(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g->total * HEADW) return;
    int local;
    const int* m = bmap_of(ctx, i / HEADW, 1, &local);
    const Desc* d = descs + m[0];
    WC_DESC(d);
    const float* src = (g->p_player ? ctx->bh2 : ctx->bh);
    ctx->h0[((size_t)d->row0 + local) * HEADW + i % HEADW] =
        src[(size_t)(m[1] + local) * HEADW + i % HEADW];
}

// The holding tower's input rows for the whole batch: per (config, slot),
// [3 counts, seat, e(card)] into bg. Ports the input build of net.rs::embed.
extern "C" __global__ void holding_in(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g->total * NSLOT) return;
    int local;
    const int* m = bmap_of(ctx, i / NSLOT, 0, &local);
    const Desc* d = descs + m[0];
    WC_DESC(d);
    int k = i % NSLOT;
    const float* p = T_cphi(d) + (size_t)local * CFEAT;
    float seat = p[CFEAT - 1];
    float* row = ctx->bg + (size_t)(m[3] + local) * NSLOT * HF + (size_t)k * HF;
    row[0] = p[k];
    row[1] = p[NSLOT + k];
    row[2] = p[2 * NSLOT + k];
    row[3] = seat;
    const float* e_row = A_e(d) + (size_t)((int)seat * NSLOT + k) * DE;
    for (int j = 0; j < DE; j++) row[4 + j] = e_row[j];
}

// Sum the rectified per-slot outputs (slot_out GEMM result in bh) into the
// batch z scratch (bh2). Warp per config.
extern "C" __global__ void slot_sum(const Desc* descs, const Group* g, const Ctx* ctx) {
    int lane = threadIdx.x & 31;
    int cfg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (cfg >= g->total) return;
    const float* src = g->p_player ? ctx->bh2 : ctx->bh;
    float* out = g->p_player ? ctx->bh : ctx->bh2;
    out += (size_t)cfg * DG;
    for (int j = lane; j < DG; j += 32) {
        float acc = 0.0f;
        for (int k = 0; k < NSLOT; k++) {
            float v = src[((size_t)cfg * NSLOT + k) * DG + j] + ctx->slot_out_b[j];
            if (v > 0.0f) acc += v;
        }
        out[j] = acc;
    }
}

// z += res_b(relu(res_a z)) + bias: the residual's second half. The GEMM
// accumulated into bh2 with beta=1; this adds the bias only (bias_act mode 4
// covers it) — kept separate for the z scatter below.

// Scatter the batch's z (bh2) and g (bg) rows into each solve's arenas.
extern "C" __global__ void scatter_zg(const Desc* descs, const Group* g, const Ctx* ctx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g->total * (DG + RK + 1)) return;
    int cfg = i / (DG + RK + 1);
    int j = i % (DG + RK + 1);
    int local;
    const int* m = bmap_of(ctx, cfg, 0, &local);
    Desc* d = (Desc*)descs + m[0];
    WC_DESC(d);
    const float* zbuf = g->p_player ? ctx->bh2 : ctx->bh;
    if (j < DG) {
        A_z(d)[(size_t)local * DG + j] = zbuf[(size_t)cfg * DG + j];
    } else {
        int r = j - DG;
        A_g(d)[(size_t)local * (RK + 1) + r] =
            ctx->bg[(size_t)cfg * (RK + 1) + r] + ctx->wg_b[r];
    }
}

// Uniform strategy at every decision node: exactly Solver::new's init. The
// arenas were zeroed at admission. Warp per node, lane per config.
extern "C" __global__ void init_strategy(const Desc* descs, const Group* g, const Ctx* ctx) {
    SOLVE(d)
    float* cur = A_cur(d);
    float* avg = A_avg(d);
    for (int t = warp; t < d->nodes; t += nwarps) {
        int i = (int)T_bfs_order(d)[t];
        if (T_node_kind(d)[i] != 0) continue;
        int me = T_node_player(d)[i];
        int nc = nc_of(d, i, me);
        int na = na_of(d, i);
        int so = (int)T_soff(d)[i];
        for (int c = lane; c < nc; c += 32) {
            int k = 0;
            for (int a = 0; a < na; a++) {
                if (legal_cell(d, so + c * na + a)) k++;
            }
            float u = 1.0f / (float)(k > 0 ? k : 1);
            for (int a = 0; a < na; a++) {
                int cell = so + c * na + a;
                float v = legal_cell(d, cell) ? u : 0.0f;
                cur[cell] = v;
                avg[cell] = v;
            }
        }
    }
}

// The reach-weighted uniform seed of the strategy sum. Snapshot 0's beliefs
// are harvested separately from the initial uniform reach.
extern "C" __global__ void seed_avg(const Desc* descs, const Group* g, const Ctx* ctx) {
    SOLVE(d)
    float* sum_strat = A_sum_strat(d);
    const float* cur = A_cur(d);
    for (int t = warp; t < d->nodes; t += nwarps) {
        int i = (int)T_bfs_order(d)[t];
        if (T_node_kind(d)[i] != 0) continue;
        int me = T_node_player(d)[i];
        int nc = nc_of(d, i, me);
        int na = na_of(d, i);
        int so = (int)T_soff(d)[i];
        const float* r = A_reach(d) + reach_at(d, i, me, 0);
        for (int c = lane; c < nc; c += 32) {
            for (int a = 0; a < na; a++) {
                int cell = so + c * na + a;
                sum_strat[cell] = r[c] * cur[cell];
            }
        }
    }
}
