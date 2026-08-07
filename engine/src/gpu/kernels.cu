// Phase kernels for the GPU solve service (work package B).
//
// Every kernel ports one function of engine/src/search.rs or engine/src/net.rs,
// with the same formulas and the same reduction orders, so a solve's result
// cannot depend on which other solves share its tick. The CPU solver is the
// oracle; the in-crate tests compare kernel arenas with the Rust functions.
//
// Three rules hold throughout, and together they are why this file has no
// pointer arithmetic of its own to get wrong:
//
//   1. Every kernel has the same signature: (descs, group, ctx). Nothing is
//      passed positionally, so a kernel and its launch cannot disagree about
//      arity or order. `launch` in service.rs is the only launch site.
//   2. A solve's arrays are reached only through the T_* and A_* accessors
//      that gpu/layout.rs generates. This file never names a byte offset.
//   3. Wide phases launch flat over the group's total thread count and find
//      their solve by binary search over `starts`. Ragged trees therefore
//      cost their own size, not the group's maximum.
//
// The `Desc`, `T_*` and `A_*` definitions, and the board geometry, are
// prepended by gpu/layout.rs before NVRTC sees this file.

#define EPS 1e-6f
#define LN_EPS 1e-5f
#define SMOOTH 1e-30f
// NVRTC does not pull in math.h's INFINITY; spell the bit pattern out.
#define INFINITY_F (__int_as_float(0x7f800000))

// The network weights and the service's row pools: everything shared by all
// solves in a tick. Uploaded once per weight publication.
typedef struct {
    const float *w0, *b0, *ln0w, *ln0b;
    const float *w1, *b1, *ln1w, *ln1b;
    const float *wb, *wu, *bu;
    const float *wc, *bc, *wh1, *bh1, *wh2, *bh2, *wg, *bg;
    const float *wd0, *bd0, *wd1, *bd1, *wid, *wpile, *bpile;
    const float *wq, *bq, *wk, *bk, *wp, *bp;
    // Row pools, indexed by a solve's stable row base (`row0`). The build
    // writes h0 straight into the pool, so a tick never packs rows.
    float *h0, *xb, *h, *u;
    // Build scratch, reused by every admission: the assembled trunk input,
    // one hidden matrix, and one gather buffer. Nothing here outlives the
    // build, so one solve's build cannot disturb a live solve's rows.
    float *bx, *bh, *bgather;
    int hidden, head, dg, rk, de, dc, af, xd, hf, cfeat, pubfeat;
} Ctx;

// The solves a launch covers. `starts` is the cumulative thread count, so
// `starts[n]` is the total and a thread's solve is the last start not
// greater than it.
typedef struct {
    const int* slots;
    const int* starts;
    int n;
    int total;
    // Switches that belong to the launch, not the solve: every solve in a
    // group shares them, so they travel with the group and the descriptor
    // stays what the solve *is* rather than what this phase does to it.
    int mode;      // backward sweep: 0 regret, 1 value, 2 best response
    int p_player;  // the player a readout/backprop pass is for
    int nplayers;  // belief players this pass: 1 (traverser) or 2 (both)
    int strat_src; // sweep strategy: 0 current, 1 average, 2 snapshot[step]
} Group;

// ------------------------------------------------------------------ helpers

__device__ __forceinline__ int find_slot(const int* starts, int n, int t) {
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (starts[mid] <= t) lo = mid; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int nc_of(const Desc* d, int node, int player) {
    const unsigned int* c = T_cfg_off(d);
    return (int)(c[2 * node + player + 1] - c[2 * node + player]);
}

// The node's obs segment's last boundary is its action count.
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

__device__ __forceinline__ float factor(float t, float p) {
    if (isinf(p)) return p > 0.0f ? 1.0f : 0.0f;
    float x = powf(t, p);
    return x / (x + 1.0f);
}

// The belief weights of one config block: `w[i] * scale + flat`, which is the
// normalised reach, or uniform when the block has no mass.
__device__ __forceinline__ void normalize_scale(const float* w, int n,
                                                float* scale, float* flat) {
    float tot = 0.0f;
    for (int i = 0; i < n; i++) tot += w[i];
    *scale = tot > SMOOTH ? 1.0f / tot : 0.0f;
    *flat = tot > SMOOTH ? 0.0f : 1.0f / (float)(n > 0 ? n : 1);
}

// The strategy a sweep reads: the current iterate, the running average, or a
// kept snapshot (the carry stage replays each one).
__device__ __forceinline__ const float* strat_of(const Desc* d, const Group* g) {
    if (g->strat_src == 0) return A_cur(d);
    if (g->strat_src == 1) return A_avg(d);
    return A_snaps(d) + (size_t)d->step * d->ncells;
}

// Whose pass this is. The iterate stage runs each solve's own traverser, so
// the group defers with TRAVERSER; the value stage names a player, because it
// reads out both in turn under one fixed policy.
__device__ __forceinline__ int player_of(const Desc* d, const Group* g) {
    return g->p_player < 0 ? d->traverser : g->p_player;
}

// The root belief this pass propagates: the live root while iterating, and
// each carried root in turn during the value stage.
__device__ __forceinline__ const float* root_of(const Desc* d) {
    if (d->stage != STAGE_VALUE) return T_root(d);
    return T_carried(d) + (size_t)d->step * (d->nc_root[0] + d->nc_root[1]);
}

// Boilerplate for a flat phase: bind `d` and `i` (the thread's solve and its
// index within that solve) or return.
#define FLAT_THREAD(d, i)                                                     \
    int _t = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (_t >= g->total) return;                                               \
    int _k = find_slot(g->starts, g->n, _t);                                  \
    const Desc* d = descs + g->slots[_k];                                     \
    int i = _t - g->starts[_k];

// Boilerplate for a per-solve sweep: one block per solve.
#define BLOCK_SOLVE(d)                                                        \
    if (blockIdx.x >= g->n) return;                                           \
    const Desc* d = descs + g->slots[blockIdx.x];

// ------------------------------------------------------- phase 1: belief sums

// One thread per (leaf row, player): xb[row][p] = sum_c w_c z[c], into the
// row pool at the solve's stable base. Ports normalize_weights + accumulate
// of Solver::leaf_values, cache and all.
//
// With `nplayers == 2` both sides are built, which is what the first query of
// a solve and every fixed-policy pass need. With 1 only one side is rebuilt,
// and it is the *previous* traverser's: that is the player whose strategy
// regret matching just moved, so it is the only belief that has changed. The
// other side is still exactly what the last tick left in the pool.
extern "C" __global__ void belief_sums(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, rp)
    int row = rp / g->nplayers;
    int p = g->nplayers == 1 ? 1 - d->traverser : rp % 2;
    int leaf = (int)T_leaf_rows(d)[row];
    int n = nc_of(d, leaf, p);
    const float* r = A_reach(d) + reach_at(d, leaf, p, 0);
    float scale, flat;
    normalize_scale(r, n, &scale, &flat);
    int dg = ctx->dg;
    float* out = ctx->xb + ((size_t)(d->row0 + row) * 2 + p) * dg;
    for (int j = 0; j < dg; j++) out[j] = 0.0f;
    int c0 = (int)T_leaf_coff(d)[2 * row + p];
    for (int c = 0; c < n; c++) {
        int ci = (int)T_leaf_cidx(d)[c0 + c];
        float wc = r[c] * scale + flat;
        const float* z = A_z(d) + (size_t)ci * dg;
        for (int j = 0; j < dg; j++) out[j] += wc * z[j];
    }
}

// ------------------------------------------------------- phase 2: the head

// `row[j] = relu((row[j] - mean) * inv * gain[j] + bt[j])` over a row of the
// packed pool, where the row is first completed with its bias and, for LN1,
// the solve's cached h0. One thread per row. Ports net.rs::ln_relu.
__device__ __forceinline__ void ln_relu_row(float* row, const float* bias,
                                            const float* gain, const float* bt,
                                            const float* add, int n) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        float x = row[j] + bias[j];
        if (add) x += add[j];
        row[j] = x;
        sum += x;
    }
    float mean = sum * (1.0f / n);
    float var = 0.0f;
    for (int j = 0; j < n; j++) {
        float e = row[j] - mean;
        var += e * e;
    }
    float inv = rsqrtf(var * (1.0f / n) + LN_EPS);
    for (int j = 0; j < n; j++) {
        float x = (row[j] - mean) * inv * gain[j] + bt[j];
        row[j] = x > 0.0f ? x : 0.0f;
    }
}

// LN1 over the head rows: h = relu(LN1(h0 + xb·Wb)). The GEMM ran in cuBLAS.
extern "C" __global__ void head_norm(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, row)
    int hd = ctx->head;
    size_t at = (size_t)(d->row0 + row) * hd;
    ln_relu_row(ctx->h + at, ctx->b1, ctx->ln1w, ctx->ln1b, ctx->h0 + at, hd);
}

// u += bu after the second head GEMM. One thread per row.
extern "C" __global__ void head_bias(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, row)
    int rk = ctx->rk;
    float* u = ctx->u + (size_t)(d->row0 + row) * rk;
    for (int j = 0; j < rk; j++) u[j] += ctx->bu[j];
}

// ------------------------------------------------------- phase 3: readout

// One thread per non-terminal leaf row plus one per terminal leaf. Ports
// Solver::readout: v = <u, g[..rk]> + g[rk], times the opponent's reach;
// terminals use the game utility.
extern "C" __global__ void readout(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, rp)
    int p = player_of(d, g);
    int opp = 1 - p;
    int rk = ctx->rk;
    int leaf = rp < d->nleaf ? (int)T_leaf_rows(d)[rp]
                             : (int)T_term_leaves(d)[rp - d->nleaf];
    float opp_reach = 0.0f;
    int nop = nc_of(d, leaf, opp);
    const float* ro = A_reach(d) + reach_at(d, leaf, opp, 0);
    for (int c = 0; c < nop; c++) opp_reach += ro[c];
    int n = nc_of(d, leaf, p);
    float* v = A_vals(d) + T_voff(d)[leaf];
    if (rp < d->nleaf) {
        const float* u = ctx->u + (size_t)(d->row0 + rp) * rk;
        int c0 = (int)T_leaf_coff(d)[2 * rp + p];
        for (int c = 0; c < n; c++) {
            int ci = (int)T_leaf_cidx(d)[c0 + c];
            const float* gr = A_g(d) + (size_t)ci * (rk + 1);
            float acc = 0.0f;
            for (int j = 0; j < rk; j++) acc += u[j] * gr[j];
            v[c] = (acc + gr[rk]) * opp_reach;
        }
    } else {
        float ut = T_terminal_utility(d)[rp - d->nleaf];
        if (T_node_player(d)[leaf] != p) ut = -ut;
        float val = ut * opp_reach;
        for (int c = 0; c < n; c++) v[c] = val;
    }
}

// ------------------------------------------------------- phase 4: backward sweep

// One block per solve; levels sequential inside the block; one thread per node
// within a level. Ports Solver::backprop. mode: 0 regret, 1 value, 2 best
// response.
extern "C" __global__ void backprop(const Desc* descs, const Group* g, const Ctx* ctx) {
    BLOCK_SOLVE(d)
    int mode = g->mode;
    int traverser = player_of(d, g);
    const float* strat = strat_of(d, g);
    float* vals = A_vals(d);
    float* inst = A_inst(d);
    const unsigned int* ls = T_level_start(d);
    const unsigned int* order = T_bfs_order(d);
    const unsigned int* child_start = T_node_child_start(d);
    const unsigned int* child = T_node_child(d);
    const unsigned int* voff = T_voff(d);
    for (int lev = d->nlevels - 1; lev >= 0; lev--) {
        int lo = (int)ls[lev], hi = (int)ls[lev + 1];
        for (int i0 = lo + threadIdx.x; i0 < hi; i0 += blockDim.x) {
            int i = (int)order[i0];
            int kind = T_node_kind(d)[i];
            if (kind == 2) continue;
            int me = T_node_player(d)[i];
            int nc = nc_of(d, i, traverser);
            int vbase = (int)voff[i];
            if (kind == 1) {  // chance
                int ch = (int)child[child_start[i]];
                const float* src = vals + voff[ch];
                if (me == traverser) {
                    int d0 = (int)T_draw_off(d)[i];
                    const unsigned int* b = T_draw_row_start(d) + T_draw_row_off(d)[i];
                    for (int c = 0; c < nc; c++) {
                        float acc = 0.0f;
                        for (int k = (int)b[c]; k < (int)b[c + 1]; k++) {
                            acc += T_draw_p(d)[d0 + k] * src[T_draw_to(d)[d0 + k]];
                        }
                        vals[vbase + c] = acc;
                    }
                } else {
                    for (int c = 0; c < nc; c++) vals[vbase + c] = src[c];
                }
                continue;
            }
            int na = na_of(d, i);
            int so = (int)T_soff(d)[i];
            if (me == traverser) {
                if (mode == 0) {
                    for (int j = 0; j < nc * na; j++) inst[so + j] = 0.0f;
                }
                float neg_inf = -INFINITY_F;
                for (int c = 0; c < nc; c++) vals[vbase + c] = mode == 2 ? neg_inf : 0.0f;
                for (int a = 0; a < na; a++) {
                    int ch = (int)child[child_start[i] + T_obs_child(d)[T_act_off(d)[i] + a]];
                    const float* cv = vals + voff[ch];
                    for (int c = 0; c < nc; c++) {
                        int cell = so + c * na + a;
                        if (!legal_cell(d, cell)) continue;
                        int tr = T_trans(d)[cell];
                        if (tr < 0) continue;
                        float av = cv[tr];
                        if (mode == 0) {
                            inst[cell] += av;
                            vals[vbase + c] += av * strat[cell];
                        } else if (mode == 1) {
                            vals[vbase + c] += av * strat[cell];
                        } else if (av > vals[vbase + c]) {
                            vals[vbase + c] = av;
                        }
                    }
                }
                if (mode == 0) {
                    for (int c = 0; c < nc; c++) {
                        float base = vals[vbase + c];
                        for (int a = 0; a < na; a++) {
                            int cell = so + c * na + a;
                            if (legal_cell(d, cell)) inst[cell] -= base;
                        }
                    }
                } else if (mode == 2) {
                    for (int c = 0; c < nc; c++) {
                        if (vals[vbase + c] == neg_inf) vals[vbase + c] = 0.0f;
                    }
                }
            } else {
                for (int c = 0; c < nc; c++) vals[vbase + c] = 0.0f;
                for (unsigned int ci = child_start[i]; ci < child_start[i + 1]; ci++) {
                    const float* cv = vals + voff[child[ci]];
                    for (int c = 0; c < nc; c++) vals[vbase + c] += cv[c];
                }
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------- phase 5: regret matching

// One thread per node. Ports the RM block of Solver::step: discount and fold
// the instantaneous regret, clamp, normalise per config, discount sum_strat.
// Only the traverser's decision nodes are touched.
extern "C" __global__ void regret_match(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, t)
    int i = (int)T_bfs_order(d)[t];
    if (T_node_kind(d)[i] != 0 || T_node_player(d)[i] != d->traverser) return;
    float m = (float)(d->steps[d->traverser] + 1);
    float da = factor(m, d->alpha);
    float db = factor(m, d->beta);
    float ds = powf(m / (m + 1.0f), d->gamma);
    int nc = nc_of(d, i, d->traverser);
    int na = na_of(d, i);
    int so = (int)T_soff(d)[i];
    float* regret = A_regret(d);
    float* inst = A_inst(d);
    float* cur = A_cur(d);
    for (int c = 0; c < nc; c++) {
        float sum = 0.0f;
        for (int a = 0; a < na; a++) {
            int cell = so + c * na + a;
            if (!legal_cell(d, cell)) {
                cur[cell] = 0.0f;
                continue;
            }
            float r = regret[cell] * (regret[cell] > 0.0f ? da : db) + inst[cell];
            regret[cell] = r;
            float v = fmaxf(r + d->predict * inst[cell], EPS);
            cur[cell] = v;
            sum += v;
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int a = 0; a < na; a++) cur[so + c * na + a] *= inv;
        }
    }
    float* sum_strat = A_sum_strat(d);
    for (int j = 0; j < nc * na; j++) sum_strat[so + j] *= ds;
}

// ------------------------------------------------------- phase 6: forward reach

// One block per solve; levels sequential; one thread per node. Ports
// Solver::propagate, chance CSR included.
extern "C" __global__ void propagate(const Desc* descs, const Group* g, const Ctx* ctx) {
    BLOCK_SOLVE(d)
    const float* strat = strat_of(d, g);
    float* reach = A_reach(d);
    // The sweep accumulates into the child blocks, so the arena starts clean
    // on every pass, exactly as Solver::propagate refills it.
    for (int c = threadIdx.x; c < A_len_reach(d); c += blockDim.x) reach[c] = 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float* root = root_of(d);
        for (int p = 0; p < 2; p++) {
            int n = d->nc_root[p];
            float* dst = reach + reach_at(d, 0, p, 0);
            const float* src = root + (p == 1 ? d->nc_root[0] : 0);
            for (int c = 0; c < n; c++) dst[c] = src[c];
        }
    }
    __syncthreads();
    const unsigned int* ls = T_level_start(d);
    const unsigned int* order = T_bfs_order(d);
    const unsigned int* child_start = T_node_child_start(d);
    const unsigned int* child = T_node_child(d);
    const unsigned int* reach_off = T_reach_off(d);
    for (int lev = 0; lev < d->nlevels; lev++) {
        int lo = (int)ls[lev], hi = (int)ls[lev + 1];
        for (int i0 = lo + threadIdx.x; i0 < hi; i0 += blockDim.x) {
            int i = (int)order[i0];
            int kind = T_node_kind(d)[i];
            if (kind == 2) continue;
            int me = T_node_player(d)[i];
            int op = 1 - me;
            int n_me = nc_of(d, i, me);
            int n_op = nc_of(d, i, op);
            int base = (int)reach_off[i];
            int n0_i = nc_of(d, i, 0);
            int me_at = base + (me == 1 ? n0_i : 0);
            int op_at = base + (op == 1 ? n0_i : 0);
            const float* src_me = reach + me_at;
            if (kind == 1) {  // chance
                int ch = (int)child[child_start[i]];
                int cbase = (int)reach_off[ch];
                int n0_c = nc_of(d, ch, 0);
                int c_me_at = cbase + (me == 1 ? n0_c : 0);
                int c_op_at = cbase + (op == 1 ? n0_c : 0);
                for (int c = 0; c < n_op; c++) reach[c_op_at + c] = reach[op_at + c];
                int d0 = (int)T_draw_off(d)[i];
                const unsigned int* b = T_draw_row_start(d) + T_draw_row_off(d)[i];
                for (int c = 0; c < n_me; c++) {
                    float wc = src_me[c];
                    if (wc == 0.0f) continue;
                    for (int k = (int)b[c]; k < (int)b[c + 1]; k++) {
                        reach[c_me_at + T_draw_to(d)[d0 + k]] += wc * T_draw_p(d)[d0 + k];
                    }
                }
                continue;
            }
            int na = na_of(d, i);
            int so = (int)T_soff(d)[i];
            const float* cur = strat + so;
            int o0 = (int)T_obs_off(d)[i];
            int act_base = (int)T_act_off(d)[i];
            unsigned int first_ch = child_start[i];
            unsigned int nch = child_start[i + 1] - first_ch;
            for (unsigned int ci = 0; ci < nch; ci++) {
                int ch = (int)child[first_ch + ci];
                int cbase = (int)reach_off[ch];
                int n0_c = nc_of(d, ch, 0);
                int c_me_at = cbase + (me == 1 ? n0_c : 0);
                int c_op_at = cbase + (op == 1 ? n0_c : 0);
                for (int c = 0; c < n_op; c++) reach[c_op_at + c] = reach[op_at + c];
                unsigned int a0 = T_obs_start(d)[o0 + ci];
                unsigned int a1 = T_obs_start(d)[o0 + ci + 1];
                for (unsigned int ai = a0; ai < a1; ai++) {
                    int a = (int)T_obs_act(d)[act_base + ai];
                    for (int c = 0; c < n_me; c++) {
                        int cell = so + c * na + a;
                        if (!legal_cell(d, cell)) continue;
                        int tr = T_trans(d)[cell];
                        if (tr < 0) continue;
                        reach[c_me_at + tr] += src_me[c] * cur[c * na + a];
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------- phase 7: average strategy

// One thread per node. Ports the AVG block of Solver::step; must run after the
// forward sweep, because it reads the fresh reaches.
extern "C" __global__ void average(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, t)
    int i = (int)T_bfs_order(d)[t];
    if (T_node_kind(d)[i] != 0 || T_node_player(d)[i] != d->traverser) return;
    int nc = nc_of(d, i, d->traverser);
    int na = na_of(d, i);
    int so = (int)T_soff(d)[i];
    const float* r = A_reach(d) + reach_at(d, i, d->traverser, 0);
    float* sum_strat = A_sum_strat(d);
    float* avg = A_avg(d);
    const float* cur = A_cur(d);
    for (int c = 0; c < nc; c++) {
        float sum = 0.0f;
        for (int a = 0; a < na; a++) {
            int cell = so + c * na + a;
            sum_strat[cell] += r[c] * cur[cell];
            sum += sum_strat[cell];
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int a = 0; a < na; a++) avg[so + c * na + a] = sum_strat[so + c * na + a] * inv;
        } else {
            int k = 0;
            for (int a = 0; a < na; a++) {
                if (legal_cell(d, so + c * na + a)) k++;
            }
            float u = 1.0f / (float)(k > 0 ? k : 1);
            for (int a = 0; a < na; a++) {
                int cell = so + c * na + a;
                avg[cell] = legal_cell(d, cell) ? u : 0.0f;
            }
        }
    }
}

// ------------------------------------------------------- bookkeeping

// Copy the running average into snapshot `snap_t`. One thread per cell.
extern "C" __global__ void snapshot(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, i)
    A_snaps(d)[(size_t)d->snap_t * d->ncells + i] = A_avg(d)[i];
}

// The value stage's harvest: the root values of carried root `step`, per
// player, into the solve's own arena. One thread per (player, config).
extern "C" __global__ void collect_root(const Desc* descs, const Group* g, const Ctx* ctx) {
    // The two players' blocks sit side by side, so the thread index is
    // already the slot to write; only its own player's pass fills it.
    FLAT_THREAD(d, i)
    int p = i < d->nc_root[0] ? 0 : 1;
    if (p != g->p_player) return;
    int c = p == 0 ? i : i - d->nc_root[0];
    int stride = d->nc_root[0] + d->nc_root[1];
    A_root_vals(d)[(size_t)d->step * stride + i] = A_vals(d)[T_voff(d)[0] + c];
}

// The carry stage: normalise the reach at the exit leaf into the solve's
// belief arena, one thread per (snapshot, player). Ports
// Solver::carried_beliefs.
extern "C" __global__ void leaf_beliefs(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, rp)
    int p = rp & 1;
    int leaf = d->leaf;
    int n = nc_of(d, leaf, p);
    const float* r = A_reach(d) + reach_at(d, leaf, p, 0);
    float scale, flat;
    normalize_scale(r, n, &scale, &flat);
    int stride = d->nc_leaf[0] + d->nc_leaf[1];
    float* o = A_beliefs(d) + (size_t)d->step * stride + (p == 1 ? d->nc_leaf[0] : 0);
    for (int c = 0; c < n; c++) o[c] = r[c] * scale + flat;
}

// ------------------------------------------------------- admission

// Uniform strategy at every decision node, and the reach-weighted seed of the
// average: exactly Solver::new's sequence. One thread per node.
extern "C" __global__ void init_strategy(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, i)
    if (T_node_kind(d)[i] != 0) return;
    int me = T_node_player(d)[i];
    int nc = nc_of(d, i, me);
    int na = na_of(d, i);
    int so = (int)T_soff(d)[i];
    const float* r = A_reach(d) + reach_at(d, i, me, 0);
    float* cur = A_cur(d);
    float* avg = A_avg(d);
    float* sum_strat = A_sum_strat(d);
    for (int c = 0; c < nc; c++) {
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
            sum_strat[cell] = r[c] * v;
        }
    }
}

// ------------------------------------------------------- build

// e = relu(facts·Wd0 + bd0)·Wd1 + bd1 + wid[id]: the card table, once per
// solve. The two GEMMs ran in cuBLAS into the ph scratch; this adds the
// biases and the identity embedding. One thread per (type, de) element.
extern "C" __global__ void cards_finish(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, t)
    int de = ctx->de;
    int ty = t / de, j = t % de;
    A_e(d)[t] += ctx->bd1[j] + ctx->wid[T_ids(d)[ty] * de + j];
}

// relu(x + bias) over a scratch matrix. One thread per element; `mode` picks
// the matrix, because the build reuses this for three different stages.
__device__ __forceinline__ void relu_bias(float* x, const float* bias, int i, int width) {
    float v = x[i] + bias[i % width];
    x[i] = v > 0.0f ? v : 0.0f;
}

// pe[t][j] = bpile[j] + sum_k e[t][k]·wpile[4 + k][j]: the card half of the
// pile summary, folded in once per solve. Written to the tail of the ph
// arena, which the assemble step reads. One thread per (type, de).
extern "C" __global__ void pile_pe(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, t)
    int de = ctx->de;
    int ty = t / de, j = t % de;
    float acc = ctx->bpile[j];
    const float* er = A_e(d) + (size_t)ty * de;
    for (int k = 0; k < de; k++) acc += er[k] * ctx->wpile[((size_t)PILE_COUNTS + k) * de + j];
    // The pe block lives after the rows of ph.
    ctx->bgather[(size_t)d->rows * NTYPE * de + t] = acc;
}

// Assemble the trunk input x[rows][xd] from xpub, e, and the pile summary:
// per-hex facts and occupant embedding, the pile block, the loose scalars.
// One thread per row. Ports net.rs::assemble. The result goes to the xb pool,
// which is wide enough to hold it before the trunk GEMM consumes it.
extern "C" __global__ void assemble(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, r)
    int de = ctx->de;
    const float* src = T_leaf_xpub(d) + (size_t)r * ctx->pubfeat;
    float* dst = ctx->bx + (size_t)r * ctx->xd;
    int hex_e = N_HEXES * HEX_FACTS;
    int piles = hex_e + N_HEXES * de;
    for (int j = 0; j < piles + 2 * de + LOOSE; j++) dst[j] = 0.0f;
    for (int h = 0; h < N_HEXES; h++) {
        const float* hx = src + h * HEX_CH;
        for (int j = 0; j < HEX_FACTS; j++) dst[h * HEX_FACTS + j] = hx[j];
        for (int k = 0; k < NTYPE; k++) {
            if (hx[HEX_FACTS + k] == 0.0f) continue;
            const float* e_row = A_e(d) + (size_t)k * de;
            for (int j = 0; j < de; j++) dst[hex_e + h * de + j] = e_row[j];
            break;
        }
    }
    const float* ph = ctx->bgather + (size_t)r * NTYPE * de;
    const float* pe = ctx->bgather + (size_t)d->rows * NTYPE * de;
    for (int t = 0; t < NTYPE; t++) {
        float* acc = dst + piles + (t / NSLOT) * de;
        for (int j = 0; j < de; j++) {
            float v = ph[(size_t)t * de + j] + pe[(size_t)t * de + j];
            acc[j] += v > 0.0f ? v : 0.0f;
        }
    }
    for (int j = 0; j < LOOSE; j++) dst[piles + 2 * de + j] = src[OFF_LOOSE + j];
}

// LN0 + ReLU over the trunk's hidden rows, in the h pool. One thread per row.
extern "C" __global__ void trunk_norm(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, row)
    int hidden = ctx->hidden;
    ln_relu_row(ctx->bh + (size_t)row * hidden, ctx->b0, ctx->ln0w, ctx->ln0b, 0, hidden);
}

// The holding tower's input rows, [ncfg * NSLOT][4 + de], from cphi and e.
// One thread per (config, slot). Ports the input build of net.rs::embed.
extern "C" __global__ void holding_in(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, t)
    int de = ctx->de, cfeat = ctx->cfeat;
    int ci = t / NSLOT, k = t % NSLOT;
    const float* p = T_cphi(d) + (size_t)ci * cfeat;
    float seat = p[cfeat - 1];
    float* row = ctx->bgather + (size_t)t * (4 + de);
    row[0] = p[k];
    row[1] = p[NSLOT + k];
    row[2] = p[2 * NSLOT + k];
    row[3] = seat;
    const float* e_row = A_e(d) + (size_t)((int)seat * NSLOT + k) * de;
    for (int j = 0; j < de; j++) row[4 + j] = e_row[j];
}

// Sum the rectified per-slot outputs into z. One thread per config. Ports the
// slot sum of net.rs::embed; the Wc GEMM ran before it, into the h pool.
extern "C" __global__ void slot_sum(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, t)
    int dg = ctx->dg;
    float* out = A_z(d) + (size_t)t * dg;
    for (int j = 0; j < dg; j++) out[j] = 0.0f;
    for (int k = 0; k < NSLOT; k++) {
        const float* o = ctx->bh + ((size_t)t * NSLOT + k) * dg;
        for (int j = 0; j < dg; j++) {
            float v = o[j] + ctx->bc[j];
            out[j] += v > 0.0f ? v : 0.0f;
        }
    }
}

// relu(hidden + bd0) over the card describer's hidden rows.
extern "C" __global__ void cards_relu(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, i)
    relu_bias(ctx->bh, ctx->bd0, i, ctx->dc);
}

// relu(res + bh1) over the holding residual's hidden rows.
extern "C" __global__ void embed_relu(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, i)
    relu_bias(ctx->bh, ctx->bh1, i, ctx->dg);
}

// z += bh2 after the residual's second GEMM accumulated into it.
extern "C" __global__ void embed_bias(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, i)
    A_z(d)[i] += ctx->bh2[i % ctx->dg];
}

// g += bg after its GEMM.
extern "C" __global__ void readout_bias(const Desc* descs, const Group* g, const Ctx* ctx) {
    FLAT_THREAD(d, i)
    A_g(d)[i] += ctx->bg[i % (ctx->rk + 1)];
}
