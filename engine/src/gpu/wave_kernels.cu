// v5 contiguous-wave kernels. The generated preamble supplies the network
// shape and feature geometry. Every tree/config/cell address below is already
// wave-global; kernels never recover a solve owner in their hot loops.

#define EPS 1e-6f
#define LN_EPS 1e-5f
#define SMOOTH 1e-30f
#define NONE 0xffffffffu

#define DG_CH ((DG + 31) / 32)
#define RK_CH ((RK + 31) / 32)
#define HEAD_CH ((HEADW + 31) / 32)

typedef struct { unsigned int node, config; } Task;
typedef struct { unsigned int node, row; } ReadTask;

typedef struct {
    unsigned int node0, nodes;
    unsigned int row0, rows, nleaf;
    unsigned int config0, ncfg;
    unsigned int cell0, ncells;
    unsigned int reach0, reach_len;
    unsigned int vals0, vals_len;
    unsigned int root0, root_n0, root_n1;
    unsigned int carried0, nroots;
    unsigned int root_value0;
    unsigned int exit0, nexits, exit_cfg0, snapshot_configs;
} JobDev;

// Immutable table slots. Keep in lockstep with device.rs::Table.
#define T_NODE_KIND 0
#define T_NODE_PLAYER 1
#define T_NODE_NC 2
#define T_NODE_CHILD_START 3
#define T_NODE_CHILD 4
#define T_LEGAL_ROW_OF 5
#define T_LEGAL_OFF 6
#define T_LEGAL_ACTION 7
#define T_LEGAL_CHILD 8
#define T_LEGAL_TRANS 9
#define T_DRAW_OFF 10
#define T_DRAW_TO 11
#define T_DRAW_P 12
#define T_DRAW_ROW_OFF 13
#define T_DRAW_ROW_START 14
#define T_REACH_OFF 15
#define T_SOFF 16
#define T_VOFF 17
#define T_NODE_PARENT 18
#define T_REV_ROW_OF 19
#define T_REV_START 20
#define T_REV_SRC 21
#define T_REV_CELL 22
#define T_RVD_ROW_OF 23
#define T_RVD_START 24
#define T_RVD_SRC 25
#define T_RVD_P 26
#define T_ROW_NODE 27
#define T_ROW_JOB 28
#define T_ROW_CFG_OFF 29
#define T_ROW_CFG 30
#define T_RAW_ROWS 31
#define T_CARD_FEAT 32
#define T_IDS 33
#define T_CONFIG_JOB 34
#define T_CPHI 35
#define T_ROOTS 36
#define T_CARRIED 37
#define T_NODE_UTILITY 38
#define T_EXIT_NODES 39
#define T_EXIT_COFF 40
#define T_DECISION0 41
#define T_DECISION1 42
#define T_REACH_TASK0 43
#define T_REACH_TASK1 44
#define T_REACH_LEVEL0 45
#define T_REACH_LEVEL1 46
#define T_BACK_TASK0 47
#define T_BACK_TASK1 48
#define T_BACK_LEVEL0 49
#define T_BACK_LEVEL1 50
#define T_READOUT 51
#define T_JOBS 52
#define N_TABLES 53

// Mutable FP32 arena slots. Keep in lockstep with device.rs::Arena.
#define A_REACH 0
#define A_SNAP_REACH 1
#define A_VALS 2
#define A_REGRET 3
#define A_CUR 4
#define A_SUM 5
#define A_SNAP_STRAT 6
#define A_E 7
#define A_Z 8
#define A_G 9
#define A_H0 10
#define A_XB 11
#define A_H 12
#define A_H2 13
#define A_U 14
#define A_ROOT_VALUES 15
#define A_CARRY 16
#define A_BX 17
#define A_BH 18
#define A_BH2 19
#define A_BG 20
#define N_ARENAS 21

typedef struct {
    const unsigned char* table;
    float* arena;
    unsigned long long toff[N_TABLES];
    unsigned long long aoff[N_ARENAS];
    int jobs, nodes, rows, nleaf, ncfg, cells, reach_len, vals_len;
    int exits, snapshot_configs, carry_snapshots, nlevels;
    int decision_n[2], reach_task_n[2], back_task_n[2], readout_n;
} WaveDev;

typedef struct {
    const float *card_w[8], *card_b[8];
    const float *wid, *pile_w, *pile_b;
    const float *pub_w[8], *pub_b[8], *pub_lnw[8], *pub_lnb[8];
    const float *pub_out_w, *pub_out_b, *wb, *ln1w, *ln1b;
    const float *hmlp_w[8], *hmlp_b[8], *wu_w, *wu_b;
    const float *slot_w[8], *slot_b[8], *slot_out_w, *slot_out_b;
    const float *res_aw[4], *res_ab[4], *res_bw[4], *res_bb[4];
    const float *wg_w, *wg_b;
} WeightDev;

#define TP(w, ty, slot) ((const ty*)((w)->table + (w)->toff[(slot)]))
#define AP(w, slot) ((w)->arena + (w)->aoff[(slot)])

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ int nc_of(const WaveDev* w, int node, int p) {
    return (int)TP(w, unsigned int, T_NODE_NC)[2 * node + p];
}

__device__ __forceinline__ int reach_at(const WaveDev* w, int node, int p, int c) {
    int at = (int)TP(w, unsigned int, T_REACH_OFF)[node];
    if (p) at += nc_of(w, node, 0);
    return at + c;
}

__device__ __forceinline__ void norm_parts(const float* x, int n, int lane,
                                            float* scale, float* flat) {
    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) sum += x[i];
    sum = warp_sum(sum);
    *scale = sum > SMOOTH ? 1.0f / sum : 0.0f;
    *flat = sum > SMOOTH ? 0.0f : 1.0f / (float)(n > 0 ? n : 1);
}

__device__ __forceinline__ void ln_relu(float* row, const float* bias,
                                         const float* gain, const float* bt,
                                         const float* add, int n, int lane) {
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32) {
        float x = row[j] + bias[j] + (add ? add[j] : 0.0f);
        row[j] = x;
        sum += x;
    }
    float mean = warp_sum(sum) / (float)n;
    float var = 0.0f;
    for (int j = lane; j < n; j += 32) {
        float d = row[j] - mean;
        var += d * d;
    }
    float inv = rsqrtf(warp_sum(var) / (float)n + LN_EPS);
    for (int j = lane; j < n; j += 32) {
        float x = (row[j] - mean) * inv * gain[j] + bt[j];
        row[j] = x > 0.0f ? x : 0.0f;
    }
}

// ---------------------------------------------------------- one-time towers

extern "C" __global__ void pack_cards(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w->jobs * NTYPE * CARD_FEATS;
    if (i < n) AP(w, A_BG)[i] = TP(w, float, T_CARD_FEAT)[i];
}

// mode: 0 card, 1 plain public ReLU (unused), 2 slot, 3 residual A,
// 4 residual B (bias only), 5 head MLP.
extern "C" __global__ void bias_act(const WaveDev* w, const WeightDev* wt,
                                     int mode, int level, int rows, int which) {
    int width;
    const float* bias;
    int relu = 1;
    if (mode == 0) { width = CARDW[level]; bias = wt->card_b[level]; }
    else if (mode == 1) { width = PUBW[level]; bias = wt->pub_b[level]; }
    else if (mode == 2) { width = SLOTW[level]; bias = wt->slot_b[level]; }
    else if (mode == 3) { width = DG; bias = wt->res_ab[level]; }
    else if (mode == 4) { width = DG; bias = wt->res_bb[level]; relu = 0; }
    else { width = HMLPW[level]; bias = wt->hmlp_b[level]; }
    float* x = AP(w, which ? A_BH2 : A_BH);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    float v = x[i] + bias[i % width];
    x[i] = relu && v < 0.0f ? 0.0f : v;
}

extern "C" __global__ void cards_finish(const WaveDev* w, const WeightDev* wt,
                                         int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w->jobs * NTYPE * DE;
    if (i >= n) return;
    int job = i / (NTYPE * DE);
    int t = (i / DE) % NTYPE;
    int j = i % DE;
    int id = TP(w, unsigned char, T_IDS)[job * NTYPE + t];
    const float* x = AP(w, which ? A_BH2 : A_BH);
    AP(w, A_E)[i] = x[i] + wt->card_b[NCARD - 1][j] + wt->wid[id * DE + j];
}

extern "C" __global__ void pile_pe(const WaveDev* w, const WeightDev* wt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w->jobs * NTYPE * DE;
    if (i >= n) return;
    int j = i % DE;
    float acc = wt->pile_b[j];
    const float* e = AP(w, A_E) + (i / DE) * DE;
    for (int k = 0; k < DE; k++)
        acc += e[k] * wt->pile_w[(PILE_COUNTS + k) * DE + j];
    AP(w, A_BG)[(unsigned long long)w->rows * NTYPE * DE + i] = acc;
}

extern "C" __global__ void pack_piles(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = NTYPE * PILE_COUNTS;
    if (i >= w->rows * per) return;
    int row = i / per;
    AP(w, A_BG)[i] = (float)TP(w, unsigned char, T_RAW_ROWS)
        [(unsigned long long)row * GPU_ROW_BYTES + GR_PILES + i % per] / 5.0f;
}

extern "C" __global__ void assemble(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int row = blockIdx.x;
    if (row >= w->rows) return;
    int job = TP(w, unsigned int, T_ROW_JOB)[row];
    const unsigned char* src = TP(w, unsigned char, T_RAW_ROWS)
        + (unsigned long long)row * GPU_ROW_BYTES;
    float* dst = AP(w, A_BX) + (unsigned long long)row * XD;
    int hex_e = N_HEXES * HEX_FACTS;
    int piles = hex_e + N_HEXES * DE;
    for (int j = threadIdx.x; j < XD; j += blockDim.x) dst[j] = 0.0f;
    __syncthreads();
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
            const float* e = AP(w, A_E) + ((unsigned long long)job * NTYPE + t) * DE;
            for (int j = lane; j < DE; j += 32) dst[hex_e + h * DE + j] = e[j];
        }
    }
    __syncthreads();
    const float* ph = AP(w, A_BH) + (unsigned long long)row * NTYPE * DE;
    const float* pe = AP(w, A_BG) + (unsigned long long)w->rows * NTYPE * DE
        + (unsigned long long)job * NTYPE * DE;
    for (int t = 0; t < NTYPE; t++) {
        float* out = dst + piles + (t / NSLOT) * DE;
        for (int j = threadIdx.x; j < DE; j += blockDim.x) {
            float v = ph[t * DE + j] + pe[t * DE + j];
            if (v > 0.0f) out[j] += v;
        }
        __syncthreads();
    }
    float* loose = dst + piles + 2 * DE;
    for (int j = threadIdx.x; j < LOOSE; j += blockDim.x) {
        float v;
        if (j < 12) {
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
        } else if (j == 13) v = src[GR_INIT_MOVED] ? 1.0f : 0.0f;
        else v = src[GR_TO_ACT] == 0 ? 1.0f : 0.0f;
        loose[j] = v;
    }
}

extern "C" __global__ void trunk_norm(const WaveDev* w, const WeightDev* wt,
                                       int level, int rows, int which) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    int width = PUBW[level];
    float* x = AP(w, which ? A_BH2 : A_BH) + (unsigned long long)row * width;
    ln_relu(x, wt->pub_b[level], wt->pub_lnw[level], wt->pub_lnb[level], 0, width, lane);
}

extern "C" __global__ void holding_in(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->ncfg * NSLOT) return;
    int cfg = i / NSLOT, k = i % NSLOT;
    int job = TP(w, unsigned int, T_CONFIG_JOB)[cfg];
    const float* p = TP(w, float, T_CPHI) + (unsigned long long)cfg * CFEAT;
    float seat = p[CFEAT - 1];
    float* out = AP(w, A_BG) + (unsigned long long)i * HF;
    out[0] = p[k]; out[1] = p[NSLOT + k]; out[2] = p[2 * NSLOT + k]; out[3] = seat;
    const float* e = AP(w, A_E) + ((unsigned long long)job * NTYPE + (int)seat * NSLOT + k) * DE;
    for (int j = 0; j < DE; j++) out[4 + j] = e[j];
}

extern "C" __global__ void slot_sum(const WaveDev* w, const WeightDev* wt,
                                     int which) {
    int lane = threadIdx.x & 31;
    int cfg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (cfg >= w->ncfg) return;
    const float* src = AP(w, which ? A_BH2 : A_BH);
    float* out = AP(w, which ? A_BH : A_BH2) + (unsigned long long)cfg * DG;
    for (int j = lane; j < DG; j += 32) {
        float v = 0.0f;
        for (int k = 0; k < NSLOT; k++)
            v += fmaxf(src[((unsigned long long)cfg * NSLOT + k) * DG + j]
                       + wt->slot_out_b[j], 0.0f);
        out[j] = v;
    }
}

extern "C" __global__ void finish_zg(const WaveDev* w, const WeightDev* wt,
                                      int zwhich) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = DG + RK + 1;
    if (i >= w->ncfg * stride) return;
    int cfg = i / stride, j = i % stride;
    if (j < DG) {
        const float* z = AP(w, zwhich ? A_BH2 : A_BH);
        AP(w, A_Z)[(unsigned long long)cfg * DG + j] = z[(unsigned long long)cfg * DG + j];
    } else {
        int k = j - DG;
        AP(w, A_G)[(unsigned long long)cfg * (RK + 1) + k] += wt->wg_b[k];
    }
}

// --------------------------------------------------------------- CFR state

extern "C" __global__ void init_strategy(const WaveDev* w, const WeightDev* wt,
                                          int player) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w->decision_n[player];
    if (i >= n) return;
    const Task* tasks = TP(w, Task, player ? T_DECISION1 : T_DECISION0);
    Task q = tasks[i];
    unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[q.node] + q.config;
    unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
    unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
    float u = 1.0f / (float)(hi > lo ? hi - lo : 1);
    for (unsigned int c = lo; c < hi; c++) AP(w, A_CUR)[c] = u;
}

extern "C" __global__ void seed_reach(const WaveDev* w, const WeightDev* wt,
                                       int snap, int root_index) {
    (void)wt;
    int j = blockIdx.x;
    if (j >= w->jobs) return;
    const JobDev d = TP(w, JobDev, T_JOBS)[j];
    float* reach = AP(w, snap ? A_SNAP_REACH : A_REACH);
    for (int i = threadIdx.x; i < (int)d.reach_len; i += blockDim.x)
        reach[d.reach0 + i] = 0.0f;
    __syncthreads();
    int nr = d.root_n0 + d.root_n1;
    if (root_index >= (int)d.nroots) return;
    const float* root = root_index < 0
        ? TP(w, float, T_ROOTS) + d.root0
        : TP(w, float, T_CARRIED) + d.carried0 + (unsigned long long)root_index * nr;
    for (int i = threadIdx.x; i < nr; i += blockDim.x) {
        int p = i >= (int)d.root_n0;
        int c = p ? i - d.root_n0 : i;
        reach[reach_at(w, d.node0, p, c)] = root[i];
    }
}

extern "C" __global__ void reach_level(const WaveDev* w, const WeightDev* wt,
                                        int player, int begin, int end,
                                        int snap, int strat_snap, int accumulate) {
    (void)wt;
    int k = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= end) return;
    const Task* tasks = TP(w, Task, player ? T_REACH_TASK1 : T_REACH_TASK0);
    Task q = tasks[k];
    int node = q.node, c = q.config;
    int parent = TP(w, unsigned int, T_NODE_PARENT)[node];
    int actor = TP(w, unsigned char, T_NODE_PLAYER)[parent];
    float* reach = AP(w, snap ? A_SNAP_REACH : A_REACH);
    const float* strat = AP(w, strat_snap ? A_SNAP_STRAT : A_CUR);
    float value = 0.0f;
    if (actor != player) {
        value = reach[reach_at(w, parent, player, c)];
    } else {
        unsigned int rr = TP(w, unsigned int, T_REV_ROW_OF)[node];
        if (rr != NONE) {
            unsigned int row = rr + c;
            unsigned int lo = TP(w, unsigned int, T_REV_START)[row];
            unsigned int hi = TP(w, unsigned int, T_REV_START)[row + 1];
            int base = reach_at(w, parent, player, 0);
            for (unsigned int x = lo; x < hi; x++)
                value += reach[base + TP(w, unsigned int, T_REV_SRC)[x]]
                       * strat[TP(w, unsigned int, T_REV_CELL)[x]];
        } else {
            unsigned int row = TP(w, unsigned int, T_RVD_ROW_OF)[node] + c;
            unsigned int lo = TP(w, unsigned int, T_RVD_START)[row];
            unsigned int hi = TP(w, unsigned int, T_RVD_START)[row + 1];
            int base = reach_at(w, parent, player, 0);
            for (unsigned int x = lo; x < hi; x++)
                value += reach[base + TP(w, unsigned int, T_RVD_SRC)[x]]
                       * TP(w, float, T_RVD_P)[x];
        }
    }
    reach[reach_at(w, node, player, c)] = value;
    if (accumulate && TP(w, unsigned char, T_NODE_KIND)[node] == 0
        && TP(w, unsigned char, T_NODE_PLAYER)[node] == player) {
        unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[node] + c;
        unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
        unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
        for (unsigned int x = lo; x < hi; x++)
            AP(w, A_SUM)[x] += value * AP(w, A_CUR)[x];
    }
}

extern "C" __global__ void seed_sum(const WaveDev* w, const WeightDev* wt,
                                     int player) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->decision_n[player]) return;
    const Task q = TP(w, Task, player ? T_DECISION1 : T_DECISION0)[i];
    unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[q.node] + q.config;
    unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
    unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
    float r = AP(w, A_REACH)[reach_at(w, q.node, player, q.config)];
    for (unsigned int x = lo; x < hi; x++) AP(w, A_SUM)[x] = r * AP(w, A_CUR)[x];
}

extern "C" __global__ void root_average(const WaveDev* w, const WeightDev* wt,
                                         int player) {
    (void)wt;
    int j = blockIdx.x;
    if (j >= w->jobs) return;
    JobDev d = TP(w, JobDev, T_JOBS)[j];
    int node = d.node0;
    if (TP(w, unsigned char, T_NODE_KIND)[node] != 0
        || TP(w, unsigned char, T_NODE_PLAYER)[node] != player) return;
    int n = nc_of(w, node, player);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[node] + c;
        unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
        unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
        float r = AP(w, A_REACH)[reach_at(w, node, player, c)];
        for (unsigned int x = lo; x < hi; x++) AP(w, A_SUM)[x] += r * AP(w, A_CUR)[x];
    }
}

// ------------------------------------------------------------ value network

extern "C" __global__ void belief_sums(const WaveDev* w, const WeightDev* wt,
                                        int traverser, int both) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= w->nleaf) return;
    ReadTask q = TP(w, ReadTask, T_READOUT)[task];
    for (int side = 0; side < (both ? 2 : 1); side++) {
        int p = both ? side : 1 - traverser;
        int n = nc_of(w, q.node, p);
        const float* r = AP(w, A_REACH) + reach_at(w, q.node, p, 0);
        float scale, flat;
        norm_parts(r, n, lane, &scale, &flat);
        float acc[DG_CH];
        #pragma unroll
        for (int z = 0; z < DG_CH; z++) acc[z] = 0.0f;
        unsigned int c0 = TP(w, unsigned int, T_ROW_CFG_OFF)[2 * q.row + p];
        for (int c = 0; c < n; c++) {
            unsigned int cfg = TP(w, unsigned int, T_ROW_CFG)[c0 + c];
            float wc = r[c] * scale + flat;
            const float* z = AP(w, A_Z) + (unsigned long long)cfg * DG;
            #pragma unroll
            for (int k = 0; k < DG_CH; k++) {
                int x = (k << 5) + lane;
                if (x < DG) acc[k] += wc * z[x];
            }
        }
        float* out = AP(w, A_XB) + ((unsigned long long)q.row * 2 + p) * DG;
        #pragma unroll
        for (int k = 0; k < DG_CH; k++) {
            int x = (k << 5) + lane;
            if (x < DG) out[x] = acc[k];
        }
    }
}

extern "C" __global__ void head_entry(const WaveDev* w, const WeightDev* wt) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= w->rows) return;
    float* dst = AP(w, A_H) + (unsigned long long)row * H_STRIDE;
    const float* add = AP(w, A_H0) + (unsigned long long)row * HEADW;
    float x[HEAD_CH], sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < HEAD_CH; k++) {
        int j = (k << 5) + lane;
        float v = j < HEADW ? dst[j] + wt->pub_out_b[j] + add[j] : 0.0f;
        x[k] = v;
        if (j < HEADW) sum += v;
    }
    float mean = warp_sum(sum) / (float)HEADW, var = 0.0f;
    #pragma unroll
    for (int k = 0; k < HEAD_CH; k++) {
        int j = (k << 5) + lane;
        if (j < HEADW) { float d = x[k] - mean; var += d * d; }
    }
    float inv = rsqrtf(warp_sum(var) / (float)HEADW + LN_EPS);
    #pragma unroll
    for (int k = 0; k < HEAD_CH; k++) {
        int j = (k << 5) + lane;
        if (j < HEADW) {
            float v = (x[k] - mean) * inv * wt->ln1w[j] + wt->ln1b[j];
            dst[j] = v > 0.0f ? v : 0.0f;
        }
    }
}

extern "C" __global__ void head_act(const WaveDev* w, const WeightDev* wt,
                                     int level, int rows, int which) {
    int width = HMLPW[level];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    int row = i / width, col = i % width;
    float* x = AP(w, which ? A_H2 : A_H)
        + (unsigned long long)row * H_STRIDE + col;
    *x = fmaxf(*x + wt->hmlp_b[level][col], 0.0f);
}

extern "C" __global__ void wu_bias(const WaveDev* w, const WeightDev* wt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w->rows * RK) AP(w, A_U)[i] += wt->wu_b[i % RK];
}

extern "C" __global__ void readout(const WaveDev* w, const WeightDev* wt,
                                    int player) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= w->readout_n) return;
    ReadTask q = TP(w, ReadTask, T_READOUT)[task];
    int opp = 1 - player;
    int nop = nc_of(w, q.node, opp);
    const float* ro = AP(w, A_REACH) + reach_at(w, q.node, opp, 0);
    float orc = 0.0f;
    for (int c = lane; c < nop; c += 32) orc += ro[c];
    orc = warp_sum(orc);
    int n = nc_of(w, q.node, player);
    float* out = AP(w, A_VALS) + TP(w, unsigned int, T_VOFF)[q.node];
    if (q.row == NONE) {
        float u = TP(w, float, T_NODE_UTILITY)[q.node];
        if (TP(w, unsigned char, T_NODE_PLAYER)[q.node] != player) u = -u;
        for (int c = lane; c < n; c += 32) out[c] = u * orc;
        return;
    }
    const float* ur = AP(w, A_U) + (unsigned long long)q.row * RK;
    unsigned int c0 = TP(w, unsigned int, T_ROW_CFG_OFF)[2 * q.row + player];
    for (int c = 0; c < n; c++) {
        unsigned int cfg = TP(w, unsigned int, T_ROW_CFG)[c0 + c];
        const float* g = AP(w, A_G) + (unsigned long long)cfg * (RK + 1);
        float part = 0.0f;
        for (int j = lane; j < RK; j += 32) part += ur[j] * g[j];
        part = warp_sum(part);
        if (lane == 0) out[c] = (part + g[RK]) * orc;
    }
}

// mode 0: CFR update using current strategy; mode 1: fixed final strategy.
extern "C" __global__ void backprop(const WaveDev* w, const WeightDev* wt,
                                     int player, int begin, int end, int mode,
                                     float da, float db, float ds, float predict) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int k = begin + ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (k >= end) return;
    const Task* tasks = TP(w, Task, player ? T_BACK_TASK1 : T_BACK_TASK0);
    Task q = tasks[k];
    int node = q.node, c = q.config;
    int kind = TP(w, unsigned char, T_NODE_KIND)[node];
    int actor = TP(w, unsigned char, T_NODE_PLAYER)[node];
    float* vals = AP(w, A_VALS);
    unsigned int vbase = TP(w, unsigned int, T_VOFF)[node];
    if (kind == 1) {
        int child = TP(w, unsigned int, T_NODE_CHILD)
            [TP(w, unsigned int, T_NODE_CHILD_START)[node]];
        if (actor == player) {
            unsigned int d0 = TP(w, unsigned int, T_DRAW_OFF)[node];
            unsigned int rb = TP(w, unsigned int, T_DRAW_ROW_OFF)[node];
            unsigned int lo = TP(w, unsigned int, T_DRAW_ROW_START)[rb + c];
            unsigned int hi = TP(w, unsigned int, T_DRAW_ROW_START)[rb + c + 1];
            float v = 0.0f;
            for (unsigned int x = lo + lane; x < hi; x += 32)
                v += TP(w, float, T_DRAW_P)[d0 + x]
                   * vals[TP(w, unsigned int, T_VOFF)[child]
                          + TP(w, unsigned int, T_DRAW_TO)[d0 + x]];
            v = warp_sum(v);
            if (lane == 0) vals[vbase + c] = v;
        } else if (lane == 0) {
            vals[vbase + c] = vals[TP(w, unsigned int, T_VOFF)[child] + c];
        }
        return;
    }
    if (actor != player) {
        float v = 0.0f;
        const unsigned int* cs = TP(w, unsigned int, T_NODE_CHILD_START);
        const unsigned int* ch = TP(w, unsigned int, T_NODE_CHILD);
        for (unsigned int x = cs[node] + lane; x < cs[node + 1]; x += 32)
            v += vals[TP(w, unsigned int, T_VOFF)[ch[x]] + c];
        v = warp_sum(v);
        if (lane == 0) vals[vbase + c] = v;
        return;
    }
    unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[node] + c;
    unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
    unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
    const float* strat = AP(w, mode ? A_SNAP_STRAT : A_CUR);
    float base = 0.0f;
    for (unsigned int x = lo + lane; x < hi; x += 32) {
        unsigned int tr = TP(w, unsigned int, T_LEGAL_TRANS)[x];
        if (tr != NONE)
            base += vals[TP(w, unsigned int, T_VOFF)
                         [TP(w, unsigned int, T_LEGAL_CHILD)[x]] + tr] * strat[x];
    }
    base = warp_sum(base);
    if (lane == 0) vals[vbase + c] = base;
    if (mode) return;
    float total = 0.0f;
    for (unsigned int x = lo + lane; x < hi; x += 32) {
        float delta = 0.0f;
        unsigned int tr = TP(w, unsigned int, T_LEGAL_TRANS)[x];
        if (tr != NONE)
            delta += vals[TP(w, unsigned int, T_VOFF)
                          [TP(w, unsigned int, T_LEGAL_CHILD)[x]] + tr];
        delta -= base;
        float old = AP(w, A_REGRET)[x];
        float r = old * (old > 0.0f ? da : db) + delta;
        AP(w, A_REGRET)[x] = r;
        float v = fmaxf(r + predict * delta, EPS);
        AP(w, A_CUR)[x] = v;
        total += v;
        AP(w, A_SUM)[x] *= ds;
    }
    total = warp_sum(total);
    if (total > 0.0f) {
        float inv = 1.0f / total;
        for (unsigned int x = lo + lane; x < hi; x += 32) AP(w, A_CUR)[x] *= inv;
    }
}

extern "C" __global__ void normalize_strategy(const WaveDev* w, const WeightDev* wt,
                                               int player, int touched) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->decision_n[player]) return;
    Task q = TP(w, Task, player ? T_DECISION1 : T_DECISION0)[i];
    unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[q.node] + q.config;
    unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
    unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
    if (!touched) {
        for (unsigned int x = lo; x < hi; x++) AP(w, A_SNAP_STRAT)[x] = AP(w, A_CUR)[x];
        return;
    }
    float sum = 0.0f;
    for (unsigned int x = lo; x < hi; x++) sum += AP(w, A_SUM)[x];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (unsigned int x = lo; x < hi; x++) AP(w, A_SNAP_STRAT)[x] = AP(w, A_SUM)[x] * inv;
    } else {
        float u = 1.0f / (float)(hi > lo ? hi - lo : 1);
        for (unsigned int x = lo; x < hi; x++) AP(w, A_SNAP_STRAT)[x] = u;
    }
}

extern "C" __global__ void gather_carry(const WaveDev* w, const WeightDev* wt,
                                         int slot, int snap) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int exit = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (exit >= w->exits || slot >= w->carry_snapshots) return;
    int node = TP(w, unsigned int, T_EXIT_NODES)[exit];
    const float* reach = AP(w, snap ? A_SNAP_REACH : A_REACH);
    for (int p = 0; p < 2; p++) {
        int n = nc_of(w, node, p);
        const float* src = reach + reach_at(w, node, p, 0);
        float scale, flat;
        norm_parts(src, n, lane, &scale, &flat);
        unsigned int off = TP(w, unsigned int, T_EXIT_COFF)[2 * exit + p];
        float* dst = AP(w, A_CARRY) + (unsigned long long)slot * w->snapshot_configs + off;
        for (int c = lane; c < n; c += 32) dst[c] = src[c] * scale + flat;
    }
}

extern "C" __global__ void collect_root(const WaveDev* w, const WeightDev* wt,
                                         int root_index, int player) {
    (void)wt;
    int j = blockIdx.x;
    if (j >= w->jobs) return;
    JobDev d = TP(w, JobDev, T_JOBS)[j];
    if (root_index >= (int)d.nroots) return;
    int n = player ? d.root_n1 : d.root_n0;
    int base = player ? d.root_n0 : 0;
    const float* src = AP(w, A_VALS) + TP(w, unsigned int, T_VOFF)[d.node0];
    float* dst = AP(w, A_ROOT_VALUES) + d.root_value0
        + (unsigned long long)root_index * (d.root_n0 + d.root_n1) + base;
    for (int c = threadIdx.x; c < n; c += blockDim.x) dst[c] = src[c];
}
