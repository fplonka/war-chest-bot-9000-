// v5 contiguous-wave kernels. The generated preamble supplies the network
// shape and feature geometry. Every tree/config/cell address below is already
// wave-global; kernels never recover a solve owner in their hot loops.

#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

#define EPS 1e-6f
#define LN_EPS 1e-5f
#define SMOOTH 1e-30f
#define NONE 0xffffffffu

#define CONFIG_CH ((CONFIG_DIM + 31) / 32)

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
#define T_LEGAL_VALUE 7
#define T_DRAW_OFF 8
#define T_DRAW_TO 9
#define T_DRAW_P 10
#define T_DRAW_ROW_OFF 11
#define T_DRAW_ROW_START 12
#define T_REACH_BASE 13
#define T_SOFF 14
#define T_VALS_BASE 15
#define T_NODE_PARENT 16
#define T_REV_ROW_OF 17
#define T_REV_START 18
#define T_REV_SRC 19
#define T_REV_CELL 20
#define T_RVD_ROW_OF 21
#define T_RVD_START 22
#define T_RVD_SRC 23
#define T_RVD_P 24
#define T_ROW_NODE 25
#define T_ROW_JOB 26
#define T_ROW_CFG_OFF 27
#define T_ROW_CFG 28
#define T_RAW_ROWS 29
#define T_CARD_FEAT 30
#define T_IDS 31
#define T_CONFIG_JOB 32
#define T_CPHI 33
#define T_ROOTS 34
#define T_CARRIED 35
#define T_NODE_UTILITY 36
#define T_EXIT_NODES 37
#define T_EXIT_COFF 38
#define T_DECISION0 39
#define T_DECISION1 40
#define T_REACH_TASK0 41
#define T_REACH_TASK1 42
#define T_REACH_LEVEL0 43
#define T_REACH_LEVEL1 44
#define T_BACK_TASK0 45
#define T_BACK_TASK1 46
#define T_BACK_LEVEL0 47
#define T_BACK_LEVEL1 48
#define T_READOUT 49
#define T_JOBS 50
#define T_CONFIG_PLAYER 51
#define N_TABLES 52

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
    const float *rule_w[2], *rule_b[2], *unit_id;
    const float *public_w[3], *public_b[3], *public_lnw[3], *public_lnb[3];
    const float *belief_slot_w, *belief_slot_b;
    const float *belief_config_w, *belief_config_b;
    const float *belief_config_lnw, *belief_config_lnb;
    const float *belief_item_w, *belief_item_b;
    const float *candidate_slot_w, *candidate_slot_b;
    const float *candidate_w, *candidate_b, *candidate_lnw, *candidate_lnb;
    const float *context_w[2], *context_b[2], *context_lnw[2], *context_lnb[2];
    const float *context_joint_w, *candidate_joint_w, *joint_bias;
    const float *value_w, *value_b;
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
    return (int)TP(w, unsigned int, T_REACH_BASE)[2 * node + p] + c;
}

__device__ __forceinline__ int value_at(const WaveDev* w, int node, int p, int c) {
    return (int)TP(w, unsigned int, T_VALS_BASE)[2 * node + p] + c;
}

__device__ __forceinline__ void norm_parts(const float* x, int n, int lane,
                                            float* scale, float* flat,
                                            float* total) {
    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) sum += x[i];
    sum = warp_sum(sum);
    *scale = sum > SMOOTH ? 1.0f / sum : 0.0f;
    *flat = sum > SMOOTH ? 0.0f : 1.0f / (float)(n > 0 ? n : 1);
    if (total) *total = sum;
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ void ln_gelu(float* row, const float* bias,
                                         const float* gain, const float* bt,
                                         int n, int lane) {
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32) {
        float x = row[j] + bias[j];
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
        row[j] = gelu((row[j] - mean) * inv * gain[j] + bt[j]);
    }
}

// ---------------------------------------------------------- one-time towers

extern "C" __global__ void pack_cards(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = 2 * w->jobs * NTYPE * CARD_FEATS;
    if (i >= n) return;
    int row = i / CARD_FEATS;
    int feature = i % CARD_FEATS;
    int t = row % NTYPE;
    int view = (row / NTYPE) & 1;
    int job = row / (2 * NTYPE);
    int physical_t = (view ? 1 - t / NSLOT : t / NSLOT) * NSLOT + t % NSLOT;
    AP(w, A_BG)[i] = TP(w, float, T_CARD_FEAT)
        [((unsigned long long)job * NTYPE + physical_t) * CARD_FEATS + feature];
}

extern "C" __global__ void bias_gelu(const WaveDev* w, const WeightDev* wt,
                                      int mode, int rows, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = mode == 0 ? UNIT_DIM : SLOT_DIM;
    if (i >= rows * width) return;
    const float* bias = mode == 0 ? wt->rule_b[0]
        : mode == 1 ? wt->belief_slot_b : wt->candidate_slot_b;
    float* x = AP(w, which ? A_BH2 : A_BH);
    x[i] = gelu(x[i] + bias[i % width]);
}

extern "C" __global__ void cards_finish(const WaveDev* w, const WeightDev* wt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = 2 * w->jobs * NTYPE * UNIT_DIM;
    if (i >= n) return;
    int row = i / UNIT_DIM;
    int j = i % UNIT_DIM;
    int t = row % NTYPE;
    int view = (row / NTYPE) & 1;
    int job = row / (2 * NTYPE);
    int physical_t = (view ? 1 - t / NSLOT : t / NSLOT) * NSLOT + t % NSLOT;
    int id = TP(w, unsigned char, T_IDS)[job * NTYPE + physical_t];
    AP(w, A_E)[i] = AP(w, A_BH2)[i] + wt->rule_b[1][j]
        + wt->unit_id[id * UNIT_DIM + j];
}

extern "C" __global__ void assemble(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int qrow = blockIdx.x;
    if (qrow >= 2 * w->rows) return;
    int row = qrow / 2, view = qrow & 1;
    int job = TP(w, unsigned int, T_ROW_JOB)[row];
    const unsigned char* src = TP(w, unsigned char, T_RAW_ROWS)
        + (unsigned long long)row * GPU_ROW_BYTES;
    float* dst = AP(w, A_BX) + (unsigned long long)qrow * PUBLIC_IN;
    int hex_e = N_HEXES * HEX_FACTS;
    int piles = hex_e + N_HEXES * UNIT_DIM;
    for (int j = threadIdx.x; j < PUBLIC_IN; j += blockDim.x) dst[j] = 0.0f;
    __syncthreads();
    for (int h = threadIdx.x / 32; h < N_HEXES; h += blockDim.x / 32) {
        int lane = threadIdx.x & 31;
        int physical_h = view ? HEX_MIRROR[h] : h;
        int owner = src[GR_HEX_OWNER + physical_h];
        if (view && owner < 2) owner = 1 - owner;
        int slot = src[GR_HEX_SLOT + physical_h];
        int marker = src[GR_HEX_MARKER + physical_h];
        if (view && marker < 2) marker = 1 - marker;
        if (lane < HEX_FACTS) {
            float v = 0.0f;
            if (lane < 2) v = owner == lane ? 1.0f : 0.0f;
            else if (lane == 2) v = (float)src[GR_HEX_HEIGHT + physical_h] / 5.0f;
            else if (lane < 5) v = marker == lane - 3 ? 1.0f : 0.0f;
            else v = (float)HEX_LOCATION[h];
            dst[h * HEX_FACTS + lane] = v;
        }
        int t = owner < 2 && slot < NSLOT ? owner * NSLOT + slot : -1;
        if (t >= 0) {
            const float* e = AP(w, A_E)
                + (((unsigned long long)job * 2 + view) * NTYPE + t) * UNIT_DIM;
            for (int j = lane; j < UNIT_DIM; j += 32)
                dst[hex_e + h * UNIT_DIM + j] = e[j];
        }
    }
    __syncthreads();
    for (int t = 0; t < NTYPE; t++) {
        int physical_t = (view ? 1 - t / NSLOT : t / NSLOT) * NSLOT + t % NSLOT;
        float* out = dst + piles + t * (PILE_COUNTS + UNIT_DIM);
        for (int j = threadIdx.x; j < PILE_COUNTS; j += blockDim.x)
            out[j] = (float)src[GR_PILES + physical_t * PILE_COUNTS + j] / 5.0f;
        const float* e = AP(w, A_E)
            + (((unsigned long long)job * 2 + view) * NTYPE + t) * UNIT_DIM;
        for (int j = threadIdx.x; j < UNIT_DIM; j += blockDim.x)
            out[PILE_COUNTS + j] = e[j];
    }
    float* loose = dst + piles + NTYPE * (PILE_COUNTS + UNIT_DIM);
    for (int j = threadIdx.x; j < LOOSE; j += blockDim.x) {
        float v;
        if (j < 12) {
            int p = j / 6, k = j % 6;
            int physical_p = view ? 1 - p : p;
            int markers = src[GR_MARKERS + physical_p];
            if (k == 0) v = (float)markers / 6.0f;
            else if (k == 1) v = (float)(6 - markers) / 6.0f;
            else if (k == 2) v = (float)src[GR_HAND + physical_p] / 3.0f;
            else if (k == 3) v = (float)src[GR_FD + physical_p] / MAX_COINS;
            else if (k == 4) v = (float)src[GR_BAG + physical_p] / MAX_COINS;
            else v = src[GR_INITIATIVE] == physical_p ? 1.0f : 0.0f;
        } else if (j == 12) {
            int plies = src[GR_PLIES] | ((int)src[GR_PLIES + 1] << 8);
            v = (float)plies / MAX_PLIES;
        } else if (j == 13) v = src[GR_INIT_MOVED] ? 1.0f : 0.0f;
        else v = src[GR_TO_ACT] == (view ? 1 : 0) ? 1.0f : 0.0f;
        loose[j] = v;
    }
}

extern "C" __global__ void norm_gelu(const WaveDev* w, const WeightDev* wt,
                                      int mode, int level, int rows, int arena) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    int width = mode == 0 ? PUBLIC_DIM : mode < 3 ? CONFIG_DIM : CONTEXT_DIM;
    const float *bias, *gain, *bt;
    if (mode == 0) { bias = wt->public_b[level]; gain = wt->public_lnw[level]; bt = wt->public_lnb[level]; }
    else if (mode == 1) { bias = wt->belief_config_b; gain = wt->belief_config_lnw; bt = wt->belief_config_lnb; }
    else if (mode == 2) { bias = wt->candidate_b; gain = wt->candidate_lnw; bt = wt->candidate_lnb; }
    else { bias = wt->context_b[level]; gain = wt->context_lnw[level]; bt = wt->context_lnb[level]; }
    float* x = AP(w, arena) + (unsigned long long)row * width;
    ln_gelu(x, bias, gain, bt, width, lane);
}

extern "C" __global__ void holding_in(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->ncfg * NSLOT) return;
    int cfg = i / NSLOT, k = i % NSLOT;
    int job = TP(w, unsigned int, T_CONFIG_JOB)[cfg];
    const float* p = TP(w, float, T_CPHI) + (unsigned long long)cfg * CFEAT;
    int player = TP(w, unsigned char, T_CONFIG_PLAYER)[cfg];
    float* out = AP(w, A_BG) + (unsigned long long)i * (3 + UNIT_DIM);
    out[0] = p[k]; out[1] = p[NSLOT + k]; out[2] = p[2 * NSLOT + k];
    const float* e = AP(w, A_E)
        + (((unsigned long long)job * 2 + player) * NTYPE + k) * UNIT_DIM;
    for (int j = 0; j < UNIT_DIM; j++) out[3 + j] = e[j];
}

extern "C" __global__ void slot_sum(const WaveDev* w, const WeightDev* wt,
                                     int which) {
    int lane = threadIdx.x & 31;
    int cfg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (cfg >= w->ncfg) return;
    const float* src = AP(w, which ? A_BH2 : A_BH);
    float* out = AP(w, which ? A_BH : A_BH2) + (unsigned long long)cfg * SLOT_DIM;
    for (int j = lane; j < SLOT_DIM; j += 32) {
        float v = 0.0f;
        for (int k = 0; k < NSLOT; k++)
            v += src[((unsigned long long)cfg * NSLOT + k) * SLOT_DIM + j];
        out[j] = v;
    }
}

extern "C" __global__ void copy_bias(const WaveDev* w, const WeightDev* wt,
                                      int mode, int rows, int arena) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * CONFIG_DIM) return;
    const float* bias = mode == 0 ? wt->belief_item_b : wt->candidate_b;
    AP(w, arena)[i] += bias[i % CONFIG_DIM];
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

__device__ __forceinline__ void reach_task(
    const WaveDev* w, int player, int k,
    int snap, int strat_snap) {
    const Task* tasks = TP(w, Task, player ? T_REACH_TASK1 : T_REACH_TASK0);
    Task q = tasks[k];
    int node = q.node, c = q.config;
    int parent = TP(w, unsigned int, T_NODE_PARENT)[node];
    float* reach = AP(w, snap ? A_SNAP_REACH : A_REACH);
    const float* strat = AP(w, strat_snap ? A_SNAP_STRAT : A_CUR);
    float value = 0.0f;
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
    reach[reach_at(w, node, player, c)] = value;
}

extern "C" __global__ void reach_sweep(
    const WaveDev* w, const WeightDev* wt, int player,
    int snap, int strat_snap, int accumulate) {
    (void)wt;
    cg::grid_group grid = cg::this_grid();
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Snapshot and Phase-2 evaluation need both independent players under one
    // fixed strategy. Share one cooperative level schedule instead of
    // launching and globally synchronising two otherwise identical sweeps.
    if (player < 0) {
        const unsigned int* level0 = TP(w, unsigned int, T_REACH_LEVEL0);
        const unsigned int* level1 = TP(w, unsigned int, T_REACH_LEVEL1);
        for (int l = 0; l < w->nlevels; l++) {
            int begin0 = level0[l], n0 = level0[l + 1] - begin0;
            int begin1 = level1[l], n1 = level1[l + 1] - begin1;
            for (int k = thread; k < n0 + n1; k += stride) {
                if (k < n0)
                    reach_task(w, 0, begin0 + k, snap, strat_snap);
                else
                    reach_task(w, 1, begin1 + k - n0, snap, strat_snap);
            }
            grid.sync();
        }
        return;
    }
    const unsigned int* level = TP(
        w, unsigned int, player ? T_REACH_LEVEL1 : T_REACH_LEVEL0);
    for (int l = 0; l < w->nlevels; l++) {
        int begin = level[l], end = level[l + 1];
        for (int k = begin + thread; k < end; k += stride)
            reach_task(w, player, k, snap, strat_snap);
        grid.sync();
    }
    if (!accumulate) return;

    // Unchanged opponent beliefs alias their parent and therefore have no
    // forward task. Accumulate every decision row in one flat pass after the
    // final reach barrier, including the root and aliased decision nodes.
    const Task* decisions = TP(w, Task, player ? T_DECISION1 : T_DECISION0);
    for (int i = thread; i < w->decision_n[player]; i += stride) {
        Task q = decisions[i];
        unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[q.node] + q.config;
        unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
        unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
        float r = AP(w, A_REACH)[reach_at(w, q.node, player, q.config)];
        for (unsigned int x = lo; x < hi; x++)
            AP(w, A_SUM)[x] += r * AP(w, A_CUR)[x];
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

// ------------------------------------------------------------ value network

extern "C" __global__ void belief_sums(const WaveDev* w, const WeightDev* wt,
                                        int traverser) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= w->nleaf) return;
    ReadTask q = TP(w, ReadTask, T_READOUT)[task];
    for (int p = 0; p < 2; p++) {
        int n = nc_of(w, q.node, p);
        const float* r = AP(w, A_REACH) + reach_at(w, q.node, p, 0);
        float scale, flat, total;
        norm_parts(r, n, lane, &scale, &flat, &total);
        if (lane == 0) {
            // Readout needs the same opponent reach mass. Cache it in the
            // value slot that readout is about to overwrite (player 0), or in
            // dead snapshot-reach scratch (player 1).
            AP(w, p ? A_VALS : A_SNAP_REACH)
                [value_at(w, q.node, 1 - p, 0)] = total;
        }
        float acc[CONFIG_CH];
        #pragma unroll
        for (int z = 0; z < CONFIG_CH; z++) acc[z] = 0.0f;
        unsigned int c0 = TP(w, unsigned int, T_ROW_CFG_OFF)[2 * q.row + p];
        for (int c = 0; c < n; c++) {
            unsigned int cfg = TP(w, unsigned int, T_ROW_CFG)[c0 + c];
            float wc = r[c] * scale + flat;
            const float* z = AP(w, A_Z) + (unsigned long long)cfg * CONFIG_DIM;
            #pragma unroll
            for (int k = 0; k < CONFIG_CH; k++) {
                int x = (k << 5) + lane;
                if (x < CONFIG_DIM) acc[k] += wc * z[x];
            }
        }
        int side = p == traverser ? 0 : 1;
        float* out = AP(w, A_XB)
            + ((unsigned long long)q.row * 2 + side) * CONFIG_DIM;
        #pragma unroll
        for (int k = 0; k < CONFIG_CH; k++) {
            int x = (k << 5) + lane;
            if (x < CONFIG_DIM) out[x] = acc[k];
        }
    }
}

extern "C" __global__ void readout(const WaveDev* w, const WeightDev* wt,
                                    int player) {
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= w->readout_n) return;
    ReadTask q = TP(w, ReadTask, T_READOUT)[task];
    int opp = 1 - player;
    int n = nc_of(w, q.node, player);
    int vbase = value_at(w, q.node, player, 0);
    float* out = AP(w, A_VALS) + vbase;
    float orc;
    if (q.row != NONE) {
        orc = AP(w, player ? A_SNAP_REACH : A_VALS)[vbase];
    } else {
        int nop = nc_of(w, q.node, opp);
        const float* ro = AP(w, A_REACH) + reach_at(w, q.node, opp, 0);
        orc = 0.0f;
        for (int c = lane; c < nop; c += 32) orc += ro[c];
        orc = warp_sum(orc);
    }
    if (q.row == NONE) {
        float u = TP(w, float, T_NODE_UTILITY)[q.node];
        if (TP(w, unsigned char, T_NODE_PLAYER)[q.node] != player) u = -u;
        for (int c = lane; c < n; c += 32) out[c] = u * orc;
        return;
    }
    __shared__ float readout_smem[(WAVE_BLOCK / 32) * JOINT_DIM];
    float* common = readout_smem + (threadIdx.x >> 5) * JOINT_DIM;
    const float* u = AP(w, A_U) + (unsigned long long)q.row * JOINT_DIM;
    for (int j = lane; j < JOINT_DIM; j += 32) common[j] = u[j];
    __syncwarp();
    unsigned int c0 = TP(w, unsigned int, T_ROW_CFG_OFF)[2 * q.row + player];
    for (int base = 0; base < n; base += 32) {
        int c = base + lane;
        if (c >= n) break;
        unsigned int cfg = TP(w, unsigned int, T_ROW_CFG)[c0 + c];
        const float* candidate = AP(w, A_G) + (unsigned long long)cfg * JOINT_DIM;
        float value = *wt->value_b;
        #pragma unroll 8
        for (int j = 0; j < JOINT_DIM; j++)
            value += gelu(common[j] + candidate[j] + wt->joint_bias[j]) * wt->value_w[j];
        out[c] = value * orc;
    }
}

// mode 0: CFR update using current strategy; mode 1: fixed final strategy.
__device__ __forceinline__ void backprop_task(
    const WaveDev* w, int player, int k, int lane, int mode,
    float da, float db, float ds, float predict) {
    const Task* tasks = TP(w, Task, player ? T_BACK_TASK1 : T_BACK_TASK0);
    Task q = tasks[k];
    int node = q.node, c = q.config;
    int kind = TP(w, unsigned char, T_NODE_KIND)[node];
    int actor = TP(w, unsigned char, T_NODE_PLAYER)[node];
    float* vals = AP(w, A_VALS);
    unsigned int vbase = value_at(w, node, player, 0);
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
                   * vals[value_at(w, child, player,
                                   TP(w, unsigned int, T_DRAW_TO)[d0 + x])];
            v = warp_sum(v);
            if (lane == 0) vals[vbase + c] = v;
        }
        return;
    }
    if (actor != player) {
        float v = 0.0f;
        const unsigned int* cs = TP(w, unsigned int, T_NODE_CHILD_START);
        const unsigned int* ch = TP(w, unsigned int, T_NODE_CHILD);
        for (unsigned int x = cs[node] + lane; x < cs[node + 1]; x += 32)
            v += vals[value_at(w, ch[x], player, c)];
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
        unsigned int value = TP(w, unsigned int, T_LEGAL_VALUE)[x];
        if (value != NONE) base += vals[value] * strat[x];
    }
    base = warp_sum(base);
    if (lane == 0) vals[vbase + c] = base;
    if (mode) return;
    float total = 0.0f;
    for (unsigned int x = lo + lane; x < hi; x += 32) {
        float delta = 0.0f;
        unsigned int value = TP(w, unsigned int, T_LEGAL_VALUE)[x];
        if (value != NONE) delta += vals[value];
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

extern "C" __global__ void backprop_sweep(
    const WaveDev* w, const WeightDev* wt, int player, int mode,
    float da, float db, float ds, float predict) {
    (void)wt;
    cg::grid_group grid = cg::this_grid();
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int stride = (gridDim.x * blockDim.x) >> 5;
    const unsigned int* level = TP(
        w, unsigned int, player ? T_BACK_LEVEL1 : T_BACK_LEVEL0);
    for (int l = w->nlevels - 1; l >= 0; l--) {
        int begin = level[l], end = level[l + 1];
        for (int k = begin + warp; k < end; k += stride)
            backprop_task(w, player, k, lane, mode, da, db, ds, predict);
        grid.sync();
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
        norm_parts(src, n, lane, &scale, &flat, nullptr);
        unsigned int off = TP(w, unsigned int, T_EXIT_COFF)[2 * exit + p];
        half* dst = reinterpret_cast<half*>(AP(w, A_CARRY))
            + (unsigned long long)slot * w->snapshot_configs + off;
        for (int c = lane; c < n; c += 32)
            dst[c] = __float2half_rn(src[c] * scale + flat);
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
    const float* src = AP(w, A_VALS) + value_at(w, d.node0, player, 0);
    float* dst = AP(w, A_ROOT_VALUES) + d.root_value0
        + (unsigned long long)root_index * (d.root_n0 + d.root_n1) + base;
    for (int c = threadIdx.x; c < n; c += blockDim.x) dst[c] = src[c];
}
