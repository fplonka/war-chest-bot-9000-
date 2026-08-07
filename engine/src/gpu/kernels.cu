// Phase kernels for the GPU solve service (work package B).
//
// Every kernel ports one Rust function of engine/src/search.rs or
// engine/src/net.rs, with the same formulas and the same reduction orders,
// so a solve's result cannot depend on which other solves share its tick.
// The CPU solver is the oracle; the in-crate tests compare kernel arenas
// with the matching Rust functions.
//
// A solve's state lives in one `SolveDesc` (device-side, uploaded once per
// tick) plus its arenas and tables. Wide phases launch over a group of
// solves: `slots` names the solve slots, `starts` their cumulative thread
// spans, and each kernel maps a thread to (slot, row-or-node) by binary
// search over `starts`. Sweeps launch one block per solve.
//
// Geometry constants below are frozen by the job format (docs/TREE.md v2 and
// rebel.rs): they are layout, not data, and the oracle tests pin them.

#define NONE 255
#define EPS 1e-6f
#define LN_EPS 1e-5f
#define SMOOTH 1e-30f
// NVRTC does not pull in math.h's INFINITY; spell the bit pattern out.
#define INFINITY_F (__int_as_float(0x7f800000))

#define N_HEXES 37
#define NSLOT 5
#define NTYPE 10
#define HEX_FACTS 6
#define HEX_CH 16
#define OFF_LOOSE 882
#define LOOSE 15
#define AOFF_PAYS 153

typedef struct {
    // Arenas (resident per solve).
    float* reach; float* vals; float* regret; float* inst; float* cur;
    float* sum_strat; float* avg; float* snaps;
    float* cz; float* cg; float* q;
    const float* root0; const float* root1;
    // Tables (uploaded once per solve).
    const unsigned char* node_kind; const unsigned char* node_player;
    const unsigned char* node_leaf;
    const unsigned int* node_child_start; const unsigned int* node_child;
    const unsigned int* obs_off; const unsigned int* obs_start;
    const unsigned int* obs_act; const unsigned int* obs_child;
    const unsigned char* legal_bits; const int* trans;
    const unsigned int* draw_off; const unsigned int* draw_to; const float* draw_p;
    const unsigned char* draw_steps;
    const unsigned int* draw_row_off; const unsigned int* draw_row_start;
    const unsigned int* cfg_off; const unsigned int* reach_off;
    const unsigned int* soff; const unsigned int* voff; const unsigned int* act_off;
    const unsigned int* leaf_rows; const unsigned int* term_leaves;
    const float* terminal_utility; const unsigned int* leaf_coff;
    const unsigned int* leaf_cidx;
    const unsigned int* bfs_order; const unsigned int* level_start;
    // Scalars.
    int nodes; int rows; int nleaf; int nterm; int ncells; int ncfg;
    int nlevels; int nsnaps; int snap_t; int t; int traverser; int stage;
    // stage: 0 = iterate, 1 = value (fixed-policy passes), 2 = carry.
    int step;        // per-stage counter: value root index / carry snapshot index
    int mode;        // backward mode: 0 regret, 1 value, 2 best response
    int leaf;        // trip-2 exit leaf
    int first_query; // belief cache: rebuild both players on the first query
    int snapshots;   // keep per-iterate snapshots
    float alpha, beta, gamma, predict;
    int steps[2];    // traversals per player (for the CFR discount)
    int nroots;      // carried roots in the value stage
    int max_nc;      // max config count at the exit leaf (belief buffer)
    int strat_src;   // propagate/backprop strategy: 0 = cur, 1 = avg, 2 = snaps[step]
    // Per-tick group fields (set before each tick's desc upload).
    int row_off;     // packed row offset into the tick's xb/u buffers
    int nplayers;    // belief players this tick: 1 (traverser) or 2 (both)
    int p_player;    // readout/backprop player
} SolveDesc;

typedef struct {
    const float* w0; const float* b0; const float* ln0w; const float* ln0b;
    const float* w1; const float* b1; const float* ln1w; const float* ln1b;
    const float* wb; const float* wu; const float* bu;
    const float* wc; const float* bc; const float* wh1; const float* bh1;
    const float* wh2; const float* bh2; const float* wg; const float* bg;
    const float* wd0; const float* bd0; const float* wd1; const float* bd1;
    const float* wid; const float* wpile; const float* bpile;
    const float* wq; const float* bq; const float* wk; const float* bk;
    const float* wp; const float* bp;
    int hidden; int head; int dg; int rk; int de; int dc; int af; int xd; int hf;
    int cfeat;
} Weights;

// ------------------------------------------------------------- helpers

__device__ __forceinline__ int nc_of(const SolveDesc* d, int node, int player) {
    return (int)(d->cfg_off[2 * node + player + 1] - d->cfg_off[2 * node + player]);
}

__device__ __forceinline__ int na_of(const SolveDesc* d, int node) {
    // The node's obs segment's last boundary is its action count.
    return (int)d->obs_start[d->obs_off[node + 1] - 1];
}

__device__ __forceinline__ int reach_at(const SolveDesc* d, int node, int player, int c) {
    int off = (int)d->reach_off[node];
    if (player == 1) off += (int)(d->cfg_off[2 * node + 1] - d->cfg_off[2 * node]);
    return off + c;
}

__device__ __forceinline__ int voff_of(const SolveDesc* d, int node) {
    return (int)d->voff[node];
}

__device__ __forceinline__ bool legal_cell(const SolveDesc* d, int cell) {
    return (d->legal_bits[cell >> 3] >> (cell & 7)) & 1;
}

__device__ __forceinline__ float factor(float t, float p) {
    if (isinf(p)) return p > 0.0f ? 1.0f : 0.0f;
    float x = powf(t, p);
    return x / (x + 1.0f);
}

__device__ __forceinline__ float normalize_scale(const float* w, int n,
                                                 float* scale, float* flat) {
    float tot = 0.0f;
    for (int i = 0; i < n; i++) tot += w[i];
    *scale = tot > SMOOTH ? 1.0f / tot : 0.0f;
    *flat = tot > SMOOTH ? 0.0f : 1.0f / (float)(n > 0 ? n : 1);
    return tot;
}

// ------------------------------------------------------- phase 1: belief sums

// One thread per (leaf row, player). Writes xb[row][p] = sum_c w_c z[c] into
// the packed xb buffer at (packed_row_off + row). `nplayers` is 1 (traverser
// only) or 2 (both); the thread's player is rp % nplayers (row-major).
// Ports normalize_weights + accumulate of Solver::leaf_values.
extern "C" __global__ void belief_sums(
    const SolveDesc* descs, const int* slots, int nslots,
    float* xb_packed, const Weights* w) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int rp = blockIdx.x * blockDim.x + threadIdx.x;
    if (rp >= d->nleaf * d->nplayers) return;
    int row = rp / d->nplayers;
    int p = rp % d->nplayers;
    int leaf = (int)d->leaf_rows[row];
    int n = nc_of(d, leaf, p);
    const float* r = d->reach + reach_at(d, leaf, p, 0);
    float scale, flat;
    normalize_scale(r, n, &scale, &flat);
    int dg = w->dg;
    float* out = xb_packed + ((size_t)(d->row_off + row) * 2 * dg + p * dg);
    int c0 = (int)d->leaf_coff[2 * row + p];
    for (int c = 0; c < n; c++) {
        int ci = (int)d->leaf_cidx[c0 + c];
        float wc = r[c] * scale + flat;
        const float* z = d->cz + (size_t)ci * dg;
        for (int j = 0; j < dg; j++) out[j] += wc * z[j];
    }
}

// ------------------------------------------------------- LayerNorm + ReLU

// `row[j] = relu((row[j] - mean) * inv * g[j] + bt[j])` where
// sum = sum_j (row[j] + bias[j] + add[j]), mean = sum/n,
// var = sum_j (row[j] - mean)^2 / n, inv = 1/sqrt(var + LN_EPS).
// One thread per row. Ports net.rs::ln_relu.
extern "C" __global__ void ln_relu_kernel(float* out, const float* bias, const float* g,
                               const float* bt, const float* add, int has_add,
                               int rows, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float* row = out + (size_t)r * n;
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        float x = row[j] + bias[j];
        if (has_add) x += add[(size_t)r * n + j];
        row[j] = x;
        sum += x;
    }
    float mean = sum * (1.0f / n);
    float var = 0.0f;
    for (int j = 0; j < n; j++) {
        float d = row[j] - mean;
        var += d * d;
    }
    var *= 1.0f / n;
    float inv = 1.0f / sqrtf(var + LN_EPS);
    for (int j = 0; j < n; j++) {
        float x = (row[j] - mean) * inv * g[j] + bt[j];
        row[j] = x > 0.0f ? x : 0.0f;
    }
}

extern "C" __global__ void bias_add_kernel(float* u, const float* bu, int rows, int rk) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)rows * rk) return;
    int j = t % rk;
    u[t] += bu[j];
}

// ------------------------------------------------------- phase 3: readout

// One thread per non-terminal leaf row plus one per terminal leaf. Ports
// Solver::readout: v = <u, g[..rk]> + g[rk] times the opponent's reach;
// terminals use the game utility.
extern "C" __global__ void readout_kernel(
    const SolveDesc* descs, const int* slots, int nslots,
    const float* u_packed, const Weights* w) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int p = d->p_player;
    int opp = 1 - p;
    int rp = blockIdx.x * blockDim.x + threadIdx.x;
    if (rp >= d->nleaf + d->nterm) return;
    int rk = w->rk;
    if (rp < d->nleaf) {
        int row = rp;
        int leaf = (int)d->leaf_rows[row];
        int n = nc_of(d, leaf, p);
        float opp_reach = 0.0f;
        int nop = nc_of(d, leaf, opp);
        const float* ro = d->reach + reach_at(d, leaf, opp, 0);
        for (int c = 0; c < nop; c++) opp_reach += ro[c];
        const float* u = u_packed + (size_t)(d->row_off + row) * rk;
        float* v = d->vals + voff_of(d, leaf);
        int c0 = (int)d->leaf_coff[2 * row + p];
        for (int c = 0; c < n; c++) {
            int ci = (int)d->leaf_cidx[c0 + c];
            const float* g = d->cg + (size_t)ci * (rk + 1);
            float acc = 0.0f;
            for (int j = 0; j < rk; j++) acc += u[j] * g[j];
            v[c] = (acc + g[rk]) * opp_reach;
        }
    } else {
        int k = rp - d->nleaf;
        int leaf = (int)d->term_leaves[k];
        float u_term = d->terminal_utility[k];
        if (d->node_player[leaf] != p) u_term = -u_term;
        float opp_reach = 0.0f;
        int nop = nc_of(d, leaf, opp);
        const float* ro = d->reach + reach_at(d, leaf, opp, 0);
        for (int c = 0; c < nop; c++) opp_reach += ro[c];
        float v = u_term * opp_reach;
        int n = nc_of(d, leaf, p);
        float* out = d->vals + voff_of(d, leaf);
        for (int c = 0; c < n; c++) out[c] = v;
    }
}

// ------------------------------------------------------- phase 4: backward

// One block per solve; levels sequential inside the block; one thread per
// node within a level. Ports Solver::backprop. The traverser's own row is
// accumulated in place in the vals arena (children live in deeper levels).
extern "C" __global__ void backprop_kernel(const SolveDesc* descs, const int* slots,
                                int nslots, const Weights* w) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int mode = d->mode;
    int traverser = d->traverser;
    const float* strat = d->strat_src == 0 ? d->cur : (d->strat_src == 1 ? d->avg
                         : d->snaps + (size_t)d->step * d->ncells);
    const unsigned int* ls = d->level_start;
    int nlevels = d->nlevels;
    for (int lev = nlevels - 1; lev >= 0; lev--) {
        int lo = (int)ls[lev], hi = (int)ls[lev + 1];
        for (int i0 = lo + threadIdx.x; i0 < hi; i0 += blockDim.x) {
            int i = (int)d->bfs_order[i0];
            int kind = d->node_kind[i];
            if (kind == 2) continue;
            int me = d->node_player[i];
            int nc = nc_of(d, i, traverser);
            int vbase = voff_of(d, i);
            if (kind == 1) {  // chance
                int ch = (int)d->node_child[d->node_child_start[i]];
                const float* src = d->vals + voff_of(d, ch);
                if (me == traverser) {
                    int d0 = (int)d->draw_off[i];
                    const unsigned int* b = d->draw_row_start + d->draw_row_off[i];
                    for (int c = 0; c < nc; c++) {
                        float acc = 0.0f;
                        for (int k = (int)b[c]; k < (int)b[c + 1]; k++) {
                            acc += d->draw_p[d0 + k] * src[d->draw_to[d0 + k]];
                        }
                        d->vals[vbase + c] = acc;
                    }
                } else {
                    for (int c = 0; c < nc; c++) d->vals[vbase + c] = src[c];
                }
                continue;
            }
            // decision
            int na = na_of(d, i);
            int so = (int)d->soff[i];
            if (me == traverser) {
                if (mode == 0) {
                    for (int j = 0; j < nc * na; j++) d->inst[so + j] = 0.0f;
                }
                float neg_inf = -INFINITY_F;
                for (int c = 0; c < nc; c++) {
                    d->vals[vbase + c] = mode == 2 ? neg_inf : 0.0f;
                }
                for (int a = 0; a < na; a++) {
                    int ch = (int)d->node_child[d->node_child_start[i] +
                                                d->obs_child[d->act_off[i] + a]];
                    const float* cv = d->vals + voff_of(d, ch);
                    for (int c = 0; c < nc; c++) {
                        int cell = so + c * na + a;
                        if (!legal_cell(d, cell)) continue;
                        int t = d->trans[cell];
                        if (t < 0) continue;
                        float av = cv[t];
                        if (mode == 0) {
                            d->inst[cell] += av;
                            d->vals[vbase + c] += av * strat[cell];
                        } else if (mode == 1) {
                            d->vals[vbase + c] += av * strat[cell];
                        } else {
                            if (av > d->vals[vbase + c]) d->vals[vbase + c] = av;
                        }
                    }
                }
                if (mode == 0) {
                    for (int c = 0; c < nc; c++) {
                        float base = d->vals[vbase + c];
                        for (int a = 0; a < na; a++) {
                            int cell = so + c * na + a;
                            if (legal_cell(d, cell)) d->inst[cell] -= base;
                        }
                    }
                } else if (mode == 2) {
                    for (int c = 0; c < nc; c++) {
                        if (d->vals[vbase + c] == neg_inf) d->vals[vbase + c] = 0.0f;
                    }
                }
            } else {
                for (int c = 0; c < nc; c++) d->vals[vbase + c] = 0.0f;
                for (unsigned int ch_i = d->node_child_start[i];
                     ch_i < d->node_child_start[i + 1]; ch_i++) {
                    int ch = (int)d->node_child[ch_i];
                    const float* cv = d->vals + voff_of(d, ch);
                    for (int c = 0; c < nc; c++) d->vals[vbase + c] += cv[c];
                }
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------- phase 5: regret matching

// One thread per node of the group's solves (bfs order). Ports the RM block
// of Solver::step: discount + fold inst, clamp, normalize per config, and
// the sum_strat discount. Only the solve's traverser's nodes are touched.
extern "C" __global__ void rm_kernel(const SolveDesc* descs, const int* slots,
                          int nslots) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= d->nodes) return;
    int i = (int)d->bfs_order[t];
    int kind = d->node_kind[i];
    if (kind != 0 || d->node_player[i] != d->traverser) return;
    float m = (float)(d->steps[d->traverser] + 1);
    float da = factor(m, d->alpha);
    float db = factor(m, d->beta);
    float dg = powf(m / (m + 1.0f), d->gamma);
    int nc = nc_of(d, i, d->traverser);
    int na = na_of(d, i);
    int so = (int)d->soff[i];
    for (int c = 0; c < nc; c++) {
        float sum = 0.0f;
        for (int a = 0; a < na; a++) {
            int cell = so + c * na + a;
            if (!legal_cell(d, cell)) {
                d->cur[cell] = 0.0f;
                continue;
            }
            float r = d->regret[cell] * (d->regret[cell] > 0.0f ? da : db) +
                      d->inst[cell];
            d->regret[cell] = r;
            float v = fmaxf(r + d->predict * d->inst[cell], EPS);
            d->cur[cell] = v;
            sum += v;
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int a = 0; a < na; a++) d->cur[so + c * na + a] *= inv;
        }
    }
    for (int j = 0; j < nc * na; j++) d->sum_strat[so + j] *= dg;
}

// ------------------------------------------------------- phase 6: forward reach

// One block per solve; levels sequential; one thread per node. Ports
// Solver::propagate, chance CSR included.
extern "C" __global__ void propagate_kernel(const SolveDesc* descs, const int* slots,
                                 int nslots, const Weights* w) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    const float* strat = d->strat_src == 0 ? d->cur : (d->strat_src == 1 ? d->avg
                         : d->snaps + (size_t)d->step * d->ncells);
    if (threadIdx.x == 0) {
        int n0 = nc_of(d, 0, 0);
        int n1 = nc_of(d, 0, 1);
        for (int c = 0; c < n0; c++) d->reach[c] = d->root0[c];
        for (int c = 0; c < n1; c++) d->reach[n0 + c] = d->root1[c];
    }
    __syncthreads();
    const unsigned int* ls = d->level_start;
    int nlevels = d->nlevels;
    for (int lev = 0; lev < nlevels; lev++) {
        int lo = (int)ls[lev], hi = (int)ls[lev + 1];
        for (int i0 = lo + threadIdx.x; i0 < hi; i0 += blockDim.x) {
            int i = (int)d->bfs_order[i0];
            int kind = d->node_kind[i];
            if (kind == 2) continue;
            int me = d->node_player[i];
            int op = 1 - me;
            int n_me = nc_of(d, i, me);
            int n_op = nc_of(d, i, op);
            int base = (int)d->reach_off[i];
            int n0_i = nc_of(d, i, 0);
            int me_at = base + (me == 1 ? n0_i : 0);
            int op_at = base + (op == 1 ? n0_i : 0);
            const float* src_me = d->reach + me_at;
            if (kind == 1) {  // chance
                int ch = (int)d->node_child[d->node_child_start[i]];
                int cbase = (int)d->reach_off[ch];
                int n0_c = nc_of(d, ch, 0);
                int c_me_at = cbase + (me == 1 ? n0_c : 0);
                int c_op_at = cbase + (op == 1 ? n0_c : 0);
                for (int c = 0; c < n_op; c++) d->reach[c_op_at + c] = d->reach[op_at + c];
                int d0 = (int)d->draw_off[i];
                const unsigned int* b = d->draw_row_start + d->draw_row_off[i];
                for (int c = 0; c < n_me; c++) {
                    float wc = src_me[c];
                    if (wc == 0.0f) continue;
                    for (int k = (int)b[c]; k < (int)b[c + 1]; k++) {
                        d->reach[c_me_at + d->draw_to[d0 + k]] += wc * d->draw_p[d0 + k];
                    }
                }
                continue;
            }
            int na = na_of(d, i);
            int so = (int)d->soff[i];
            const float* cur = strat + so;
            int o0 = (int)d->obs_off[i];
            int act_base = (int)d->act_off[i];
            unsigned int first_ch = d->node_child_start[i];
            unsigned int nch = d->node_child_start[i + 1] - first_ch;
            for (unsigned int ci = 0; ci < nch; ci++) {
                int ch = (int)d->node_child[first_ch + ci];
                int cbase = (int)d->reach_off[ch];
                int n0_c = nc_of(d, ch, 0);
                int c_me_at = cbase + (me == 1 ? n0_c : 0);
                int c_op_at = cbase + (op == 1 ? n0_c : 0);
                for (int c = 0; c < n_op; c++) d->reach[c_op_at + c] = d->reach[op_at + c];
                unsigned int a0 = d->obs_start[o0 + ci];
                unsigned int a1 = d->obs_start[o0 + ci + 1];
                for (unsigned int ai = a0; ai < a1; ai++) {
                    int a = (int)d->obs_act[act_base + ai];
                    for (int c = 0; c < n_me; c++) {
                        int cell = so + c * na + a;
                        if (!legal_cell(d, cell)) continue;
                        int t = d->trans[cell];
                        if (t < 0) continue;
                        d->reach[c_me_at + t] += src_me[c] * cur[c * na + a];
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------- phase 7: average strategy

// One thread per node of the group's solves. Ports the AVG block of
// Solver::step (must run after the forward sweep: reads the fresh reaches).
extern "C" __global__ void avg_kernel(const SolveDesc* descs, const int* slots,
                           int nslots) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= d->nodes) return;
    int i = (int)d->bfs_order[t];
    int kind = d->node_kind[i];
    if (kind != 0 || d->node_player[i] != d->traverser) return;
    int nc = nc_of(d, i, d->traverser);
    int na = na_of(d, i);
    int so = (int)d->soff[i];
    const float* r = d->reach + reach_at(d, i, d->traverser, 0);
    for (int c = 0; c < nc; c++) {
        float sum = 0.0f;
        for (int a = 0; a < na; a++) {
            int cell = so + c * na + a;
            d->sum_strat[cell] += r[c] * d->cur[cell];
            sum += d->sum_strat[cell];
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int a = 0; a < na; a++) {
                d->avg[so + c * na + a] = d->sum_strat[so + c * na + a] * inv;
            }
        } else {
            int k = 0;
            for (int a = 0; a < na; a++) {
                if (legal_cell(d, so + c * na + a)) k++;
            }
            float u = 1.0f / (float)(k > 0 ? k : 1);
            for (int a = 0; a < na; a++) {
                int cell = so + c * na + a;
                d->avg[cell] = legal_cell(d, cell) ? u : 0.0f;
            }
        }
    }
}

// ------------------------------------------------------- carry stage: beliefs

// Normalise the reach at the exit leaf into the solve's belief buffer, one
// thread per (snapshot, player): out[(snap * 2 + p) * max_nc + c]. Ports
// Solver::carried_beliefs.
extern "C" __global__ void leaf_beliefs_kernel(const SolveDesc* descs, const int* slots,
                                    int nslots, float* out) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int rp = blockIdx.x * blockDim.x + threadIdx.x;
    if (rp >= 2 * (d->nsnaps - 1)) return;
    int snap = rp >> 1;
    int p = rp & 1;
    int leaf = d->leaf;
    int n = nc_of(d, leaf, p);
    const float* r = d->reach + reach_at(d, leaf, p, 0);
    float scale, flat;
    normalize_scale(r, n, &scale, &flat);
    float* o = out + ((size_t)snap * 2 + p) * d->max_nc;
    for (int c = 0; c < n; c++) o[c] = r[c] * scale + flat;
}

// ------------------------------------------------------- build kernels

// The card table finish: e = hid Wd1 + bd1 + wid[ids] (the Wd1 GEMM ran in
// the service). One thread per (type, de) element.
extern "C" __global__ void cards_finish(float* e, const float* hid, const float* bd1,
                             const float* wid, const unsigned char* ids,
                             int ntype, int de) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntype * de) return;
    int ty = t / de, j = t % de;
    e[t] = hid[t] + bd1[j] + wid[ids[ty] * de + j];
}

// Assemble the trunk input x[rows][xd] from xpub + e: the per-hex facts and
// occupant embedding, the pile summary (the service precomputed
// ph = cnt @ wpile[:4] and pe = bpile + e @ wpile[4:]), and the loose
// scalars. One thread per row. Ports net.rs::assemble.
extern "C" __global__ void assemble_kernel(
    float* x, const float* xpub, const float* e, const float* ph,
    const float* pe, int rows, int pubfeat, int de) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* src = xpub + (size_t)r * pubfeat;
    float* dst = x + (size_t)r * (N_HEXES * HEX_FACTS + N_HEXES * de + 2 * de + LOOSE);
    int hex_e = N_HEXES * HEX_FACTS;
    int piles = hex_e + N_HEXES * de;
    for (int h = 0; h < N_HEXES; h++) {
        const float* hx = src + h * HEX_CH;
        for (int j = 0; j < HEX_FACTS; j++) dst[h * HEX_FACTS + j] = hx[j];
        int t = -1;
        for (int k = 0; k < NTYPE; k++) {
            if (hx[HEX_FACTS + k] != 0.0f) { t = k; break; }
        }
        if (t >= 0) {
            const float* e_row = e + (size_t)t * de;
            for (int j = 0; j < de; j++) dst[hex_e + h * de + j] = e_row[j];
        }
    }
    for (int t = 0; t < NTYPE; t++) {
        const float* ph_row = ph + ((size_t)r * NTYPE + t) * de;
        const float* pe_row = pe + (size_t)t * de;
        float* acc = dst + piles + (t / NSLOT) * de;
        for (int j = 0; j < de; j++) {
            float v = ph_row[j] + pe_row[j];
            acc[j] += v > 0.0f ? v : 0.0f;
        }
    }
    for (int j = 0; j < LOOSE; j++) dst[piles + 2 * de + j] = src[OFF_LOOSE + j];
}

// Holding tower input: [n * NSLOT][hf] from cphi + e. One thread per
// (config, slot). Ports the input build of net.rs::embed.
extern "C" __global__ void holding_in_kernel(float* inp, const float* cphi, const float* e,
                                  int n, int cfeat, int de) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * NSLOT) return;
    int ci = t / NSLOT, k = t % NSLOT;
    const float* p = cphi + (size_t)ci * cfeat;
    float seat = p[cfeat - 1];
    float* row = inp + (size_t)t * (4 + de);
    row[0] = p[k];
    row[1] = p[NSLOT + k];
    row[2] = p[2 * NSLOT + k];
    row[3] = seat;
    int ty = (int)seat * NSLOT + k;
    const float* e_row = e + (size_t)ty * de;
    for (int j = 0; j < de; j++) row[4 + j] = e_row[j];
}

// Sum the per-slot rectified outputs into z. One thread per config. Ports
// the slot sum of net.rs::embed (the service GEMMs wc/bc before this).
extern "C" __global__ void slot_sum_kernel(float* z, const float* slot, const float* bc,
                                int n, int dg) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float* out = z + (size_t)t * dg;
    for (int j = 0; j < dg; j++) out[j] = 0.0f;
    for (int k = 0; k < NSLOT; k++) {
        const float* o = slot + ((size_t)t * NSLOT + k) * dg;
        for (int j = 0; j < dg; j++) {
            float v = o[j] + bc[j];
            out[j] += v > 0.0f ? v : 0.0f;
        }
    }
}

// z += res + bh2, one thread per element (the residual's second stage).
extern "C" __global__ void add2_kernel(float* z, const float* res, const float* bh2,
                            int n, int dg) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * dg) return;
    int j = t % dg;
    z[t] += res[t] + bh2[j];
}

// Action tower input: [na][af] = [psi | e of the paying card]. One thread
// per (action, de) column; the psi copy is done by the first thread of the
// action. Ports the gather of net.rs::embed_actions.
extern "C" __global__ void action_in_kernel(float* inp, const float* psi, const float* e,
                                 int na, int afeat, int de) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= na * de) return;
    int a = t / de, j = t % de;
    int af = afeat + de;
    const float* src = psi + (size_t)a * afeat;
    float* dst = inp + (size_t)a * af;
    if (j == 0) {
        for (int k = 0; k < afeat; k++) dst[k] = src[k];
    }
    int pays = -1;
    for (int k = AOFF_PAYS; k < AOFF_PAYS + NTYPE; k++) {
        if (src[k] != 0.0f) { pays = k - AOFF_PAYS; break; }
    }
    if (pays >= 0) dst[afeat + j] = e[(size_t)pays * de + j];
}

// Seeding after the warm start: regret = w * inst, sum_strat = w * r * cur,
// per traverser node. Grid (nodes, nslots). Ports Solver::warm_start's seed.
extern "C" __global__ void warm_seed_kernel(const SolveDesc* descs, const int* slots,
                                 int nslots, float weight) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d->nodes) return;
    int kind = d->node_kind[i];
    if (kind != 0 || d->node_player[i] != d->traverser) return;
    int nc = nc_of(d, i, d->traverser);
    int na = na_of(d, i);
    int so = (int)d->soff[i];
    const float* r = d->reach + reach_at(d, i, d->traverser, 0);
    for (int c = 0; c < nc; c++) {
        for (int a = 0; a < na; a++) {
            int cell = so + c * na + a;
            d->regret[cell] = weight * d->inst[cell];
            d->sum_strat[cell] = weight * r[c] * d->cur[cell];
        }
    }
}

// ------------------------------------------------------- build kernels (2)

// pe[t][j] = bpile[j] + sum_k e[t][k] * wpile[(4 + k)][j]: the card half of
// the pile summary, folded into the bias once per solve. One thread per
// (type, de) element. Ports the pe loop of net.rs::assemble.
extern "C" __global__ void pile_pe_kernel(float* pe, const float* e, const float* wpile,
                               const float* bpile, int de) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NTYPE * de) return;
    int ty = t / de, j = t % de;
    float acc = bpile[j];
    const float* er = e + (size_t)ty * de;
    for (int k = 0; k < de; k++) {
        acc += er[k] * wpile[((size_t)4 + k) * de + j];
    }
    pe[t] = acc;
}

// relu(x + bias), one thread per element.
extern "C" __global__ void relu_bias_kernel(float* x, const float* bias, int n, int width) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * width) return;
    int j = t % width;
    float v = x[t] + bias[j];
    x[t] = v > 0.0f ? v : 0.0f;
}

// Uniform strategy init at admission: cur/avg = uniform over the legal
// actions per config, at every decision node. Grid (nodes, nslots). Ports
// the strategy init of Solver::new.
extern "C" __global__ void init_strategy_kernel(const SolveDesc* descs, const int* slots,
                                     int nslots) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d->nodes) return;
    int kind = d->node_kind[i];
    if (kind != 0) return;
    int me = d->node_player[i];
    int nc = nc_of(d, i, me);
    int na = na_of(d, i);
    int so = (int)d->soff[i];
    for (int c = 0; c < nc; c++) {
        int k = 0;
        for (int a = 0; a < na; a++) {
            if (legal_cell(d, so + c * na + a)) k++;
        }
        float u = 1.0f / (float)(k > 0 ? k : 1);
        for (int a = 0; a < na; a++) {
            int cell = so + c * na + a;
            float v = legal_cell(d, cell) ? u : 0.0f;
            d->cur[cell] = v;
            d->avg[cell] = v;
        }
    }
}

// Seed the average strategy sum with one reach-weighted uniform strategy,
// as Solver::new does (the initial reach was uploaded). Grid (nodes, nslots).
extern "C" __global__ void seed_sum_kernel(const SolveDesc* descs, const int* slots,
                                int nslots) {
    int s = blockIdx.y;
    if (s >= nslots) return;
    const SolveDesc* d = descs + slots[s];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d->nodes) return;
    int kind = d->node_kind[i];
    if (kind != 0) return;
    int me = d->node_player[i];
    int nc = nc_of(d, i, me);
    int na = na_of(d, i);
    int so = (int)d->soff[i];
    const float* r = d->reach + reach_at(d, i, me, 0);
    for (int c = 0; c < nc; c++) {
        for (int a = 0; a < na; a++) {
            d->sum_strat[so + c * na + a] += r[c] * d->cur[so + c * na + a];
        }
    }
}
