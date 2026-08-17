// Contiguous-wave CUDA kernels. The generated preamble supplies the network
// shape and the board's geometry. Every tree/config/cell address below is
// already wave-global; kernels never recover a solve owner in their hot loops.

#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

#define EPS 1e-6f
#define LN_EPS 1e-5f
#define SMOOTH 1e-30f
#define NONE 0xffffffffu

/// The trunk activation tensor carries one zero hex per row so the neighbour
/// gather can address a missing neighbour as hex `N_HEXES` and stay
/// branch-free.
#define HEX_STRIDE (N_HEXES + 1)

static_assert(JOIN_IN == 2 * POOL + 1, "join input is two beliefs and one seat");
static_assert(HEX_FACTS == 6, "hex_facts writes the six frozen per-hex facts");
static_assert(LOOSE == 15, "loose_feature writes two player blocks and three globals");

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
#define T_CONFIG_JOB 31
#define T_CPHI 32
#define T_ROOTS 33
#define T_CARRIED 34
#define T_NODE_UTILITY 35
#define T_EXIT_NODES 36
#define T_EXIT_COFF 37
#define T_DECISION0 38
#define T_DECISION1 39
#define T_REACH_TASK0 40
#define T_REACH_TASK1 41
#define T_REACH_LEVEL0 42
#define T_REACH_LEVEL1 43
#define T_BACK_TASK0 44
#define T_BACK_TASK1 45
#define T_BACK_LEVEL0 46
#define T_BACK_LEVEL1 47
#define T_READOUT 48
#define T_JOBS 49
#define T_CONFIG_PLAYER 50
#define N_TABLES 51

// Mutable FP32 arena slots. Keep in lockstep with device.rs::Arena.
#define A_REACH 0
#define A_SNAP_REACH 1
#define A_VALS 2
#define A_REGRET 3
#define A_CUR 4
#define A_SUM 5
#define A_SNAP_STRAT 6
#define A_CARDS 7
#define A_BAG 8
#define A_F 9
#define A_G 10
#define A_P 11
#define A_JP 12
#define A_POOLED 13
#define A_JIN 14
#define A_Z 15
#define A_JT 16
#define A_H 17
#define A_ROOT_VALUES 18
#define A_CARRY 19
#define A_TOK 20
#define A_TS 21
#define A_X 22
#define A_A 23
#define A_MIX 24
#define A_POOL 25
#define A_GB 26
#define A_BOARD 27
#define A_PACK 28
#define A_HIDDEN 29
#define A_CFG 30
#define N_ARENAS 31

typedef struct {
    const unsigned char* table;
    float* arena;
    unsigned long long toff[N_TABLES];
    unsigned long long aoff[N_ARENAS];
    int jobs, nodes, rows, nleaf, ncfg, cells, reach_len, vals_len;
    int exits, snapshot_configs, carry_snapshots, nlevels;
    int decision_n[2], reach_task_n[2], back_task_n[2], readout_n;
} WaveDev;

/// Only what device code dereferences: biases, LayerNorm pairs, the two
/// embeddings, and the three matrices small enough that a kernel applies them
/// itself instead of paying for a pack buffer and a tiny-K GEMM. Every other
/// matrix reaches cuBLAS as a raw offset from the host.
typedef struct {
    const float* card_b[2];
    const float* pile_w;
    const float* seat;
    const float* hex_stem_w;
    const float* hex_stem_b;
    const float* pos;
    const float* glob_stem_w;
    const float* mix_b[BLOCKS];
    const float* pool_b[BLOCKS];
    const float* out_b[BLOCKS];
    // `pre` runs on the residual stream (index `BLOCKS` is the trunk's final
    // normalisation); `mid` runs between a block's two halves.
    const float* pre_lnw[BLOCKS + 1];
    const float* pre_lnb[BLOCKS + 1];
    const float* mid_lnw[BLOCKS];
    const float* mid_lnb[BLOCKS];
    const float* board_b;
    const float* cfg1_b;
    const float* cfg_lnw;
    const float* cfg_lnb;
    const float* cfg_f_b;
    const float* cfg_g_b;
    const float* join_b_b;
    const float* join_w_b[JBLOCKS];
    const float* join_lnw[JBLOCKS];
    const float* join_lnb[JBLOCKS];
    const float* jout_lnw;
    const float* jout_lnb;
    const float* join_out_b;
    const float* h_lnw;
    const float* h_lnb;
    const float* value_bias;
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

__device__ __forceinline__ const unsigned char* raw_row(const WaveDev* w, int row) {
    return TP(w, unsigned char, T_RAW_ROWS) + (unsigned long long)row * GPU_ROW_BYTES;
}

/// Physical coin-type index of view `view`'s token `t`: a view's first `NSLOT`
/// tokens are always its own seat's coins.
__device__ __forceinline__ int physical_type(int view, int t) {
    return (view ? 1 - t / NSLOT : t / NSLOT) * NSLOT + t % NSLOT;
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

/// LayerNorm over one `N`-channel row owned by one warp. `load(j)` supplies the
/// pre-norm value, so a caller can fold in a bias or a broadcast term without
/// a separate pass; the row stays in registers across both reductions because
/// reloading it from the arena costs more than the whole arithmetic. `ACT`
/// appends the GELU every pre-activation block wants -- the readout
/// normalisation is the one place that does not.
template <int N, bool ACT, class Load>
__device__ __forceinline__ void norm_row(Load load, float* dst,
                                          const float* gain, const float* bt,
                                          int lane) {
    constexpr int CH = N / 32;
    float x[CH];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < CH; k++) {
        x[k] = load((k << 5) + lane);
        sum += x[k];
    }
    float mean = warp_sum(sum) / (float)N;
    float var = 0.0f;
    #pragma unroll
    for (int k = 0; k < CH; k++) var += (x[k] - mean) * (x[k] - mean);
    float inv = rsqrtf(warp_sum(var) / (float)N + LN_EPS);
    #pragma unroll
    for (int k = 0; k < CH; k++) {
        int j = (k << 5) + lane;
        float v = (x[k] - mean) * inv * gain[j] + bt[j];
        dst[j] = ACT ? gelu(v) : v;
    }
}

// ------------------------------------------------------ public row expansion

/// The `HEX_FACTS` raw facts of one hex, in `write_public_features` order:
/// occupant owner one-hot, stack height, location-marker owner one-hot, and
/// whether the hex is a location. `view` mirrors the board and swaps seats.
/// `occupant` is the coin-type token standing there, or -1 for an empty hex.
__device__ __forceinline__ void hex_facts(const unsigned char* src, int view,
                                           int h, float* out, int* occupant) {
    int ph = view ? HEX_MIRROR[h] : h;
    int owner = src[GR_HEX_OWNER + ph];
    if (view && owner < 2) owner = 1 - owner;
    int marker = src[GR_HEX_MARKER + ph];
    if (view && marker < 2) marker = 1 - marker;
    int slot = src[GR_HEX_SLOT + ph];
    out[0] = owner == 0 ? 1.0f : 0.0f;
    out[1] = owner == 1 ? 1.0f : 0.0f;
    out[2] = (float)src[GR_HEX_HEIGHT + ph] / CNORM;
    out[3] = marker == 0 ? 1.0f : 0.0f;
    out[4] = marker == 1 ? 1.0f : 0.0f;
    out[5] = (float)HEX_LOCATION[h];
    *occupant = owner < 2 && slot < NSLOT ? owner * NSLOT + slot : -1;
}

/// Scalar `j` of the `LOOSE` block: two six-wide player blocks, then plies,
/// whether initiative has moved, and whether this view is to act.
__device__ __forceinline__ float loose_feature(const unsigned char* src,
                                                int view, int j) {
    if (j < 12) {
        int p = j / 6, k = j % 6;
        int pp = view ? 1 - p : p;
        int markers = src[GR_MARKERS + pp];
        if (k == 0) return (float)markers / 6.0f;
        if (k == 1) return (float)(6 - markers) / 6.0f;
        if (k == 2) return (float)src[GR_HAND + pp] / 3.0f;
        if (k == 3) return (float)src[GR_FD + pp] / MAX_COINS;
        if (k == 4) return (float)src[GR_BAG + pp] / MAX_COINS;
        return src[GR_INITIATIVE] == pp ? 1.0f : 0.0f;
    }
    if (j == 12) {
        int plies = src[GR_PLIES] | ((int)src[GR_PLIES + 1] << 8);
        return (float)plies / MAX_PLIES;
    }
    if (j == 13) return src[GR_INIT_MOVED] ? 1.0f : 0.0f;
    return src[GR_TO_ACT] == (view ? 1 : 0) ? 1.0f : 0.0f;
}

// ------------------------------------------------------------- card describer

extern "C" __global__ void pack_cards(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * w->jobs * NTYPE * CARD_FEATS) return;
    int feature = i % CARD_FEATS;
    int row = i / CARD_FEATS;
    int t = row % NTYPE;
    int view = (row / NTYPE) & 1;
    int job = row / (2 * NTYPE);
    AP(w, A_PACK)[i] = TP(w, float, T_CARD_FEAT)
        [((unsigned long long)job * NTYPE + physical_type(view, t)) * CARD_FEATS
         + feature];
}

/// mode 0: the card describer's hidden layer; mode 1: the config encoder's.
extern "C" __global__ void bias_gelu(const WaveDev* w, const WeightDev* wt,
                                      int mode, int rows) {
    int width = mode == 0 ? TYPE : CFGH;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const float* bias = mode == 0 ? wt->card_b[0] : wt->cfg1_b;
    float* x = AP(w, A_HIDDEN);
    x[i] = gelu(x[i] + bias[i % width]);
}

extern "C" __global__ void cards_finish(const WaveDev* w, const WeightDev* wt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * w->jobs * NTYPE * TYPE) return;
    AP(w, A_CARDS)[i] += wt->card_b[1][i % TYPE];
}

// -------------------------------------------------------------------- trunk

/// One coin-type token per physical row: printed card, pile counts and seat.
extern "C" __global__ void tokens(const WaveDev* w, const WeightDev* wt,
                                  int row0, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * NTYPE * TYPE) return;
    int j = i % TYPE;
    int t = (i / TYPE) % NTYPE;
    int row = row0 + i / (NTYPE * TYPE);
    int job = TP(w, unsigned int, T_ROW_JOB)[row];
    const unsigned char* pile =
        raw_row(w, row) + GR_PILES + physical_type(0, t) * PILE_COUNTS;
    float v = AP(w, A_CARDS)
            [((unsigned long long)job * 2 * NTYPE + t) * TYPE + j]
        + wt->seat[(t / NSLOT) * TYPE + j];
    #pragma unroll
    for (int f = 0; f < PILE_COUNTS; f++)
        v += wt->pile_w[f * TYPE + j] * ((float)pile[f] / CNORM);
    AP(w, A_TOK)[i] = v;
}

/// The trunk stem, one block per physical row. The global term pools every
/// projected type token, so piles remain visible before a unit reaches board.
extern "C" __global__ void stem(const WaveDev* w, const WeightDev* wt,
                                 int row0) {
    int local = blockIdx.x;
    int row = row0 + local;
    const unsigned char* src = raw_row(w, row);
    __shared__ float facts[N_HEXES * HEX_FACTS];
    __shared__ int occupant[N_HEXES];
    __shared__ float glob[C];
    __shared__ float loose[LOOSE];
    for (int h = threadIdx.x; h < N_HEXES; h += blockDim.x)
        hex_facts(src, 0, h, &facts[h * HEX_FACTS], &occupant[h]);
    for (int f = threadIdx.x; f < LOOSE; f += blockDim.x)
        loose[f] = loose_feature(src, 0, f);
    __syncthreads();
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float v = 0.0f;
        for (int f = 0; f < LOOSE; f++) v += wt->glob_stem_w[f * C + j] * loose[f];
        for (int t = 0; t < NTYPE; t++)
            v += gelu(AP(w, A_TS)[((unsigned long long)local * NTYPE + t) * C + j])
               / (float)NTYPE;
        glob[j] = v;
    }
    __syncthreads();
    float* x = AP(w, A_X) + (unsigned long long)local * N_HEXES * C;
    for (int i = threadIdx.x; i < N_HEXES * C; i += blockDim.x) {
        int h = i / C, j = i % C;
        float v = wt->hex_stem_b[j] + wt->pos[h * C + j] + glob[j];
        #pragma unroll
        for (int f = 0; f < HEX_FACTS; f++)
            v += wt->hex_stem_w[f * C + j] * facts[h * HEX_FACTS + f];
        int t = occupant[h];
        if (t >= 0)
            v += AP(w, A_TS)[((unsigned long long)local * NTYPE + t) * C + j];
        x[i] = v;
    }
}

/// Normalise one board row into shared memory, then build both inputs consumed
/// by the block. Keeping the normalised tokens on chip avoids writing them to
/// global memory and reading each token again for six neighbours and pooling.
extern "C" __global__ void trunk_gather_pool(const WaveDev* w,
                                              const WeightDev* wt,
                                              int block, int n) {
    __shared__ float a[HEX_STRIDE * C];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int r = blockIdx.x;
    if (r >= n) return;

    const float* x = AP(w, A_X) + (unsigned long long)r * N_HEXES * C;
    for (int h = warp; h < N_HEXES; h += warps) {
        norm_row<C, true>([&](int j) { return x[h * C + j]; }, a + h * C,
                          wt->pre_lnw[block], wt->pre_lnb[block], lane);
    }
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        a[N_HEXES * C + j] = 0.0f;
    }
    __syncthreads();

    float* mix = AP(w, A_MIX) + (unsigned long long)r * N_HEXES * 2 * C;
    for (int i = threadIdx.x; i < N_HEXES * C; i += blockDim.x) {
        int h = i / C, j = i - h * C;
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < 6; d++) {
            sum += a[HEX_NEIGHBOUR[h * 6 + d] * C + j];
        }
        mix[(unsigned long long)h * 2 * C + j] = a[h * C + j];
        mix[(unsigned long long)h * 2 * C + C + j] = sum;
    }

    float* pool = AP(w, A_POOL) + (unsigned long long)r * 2 * C;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float sum = a[j], top = a[j];
        for (int h = 1; h < N_HEXES; h++) {
            float v = a[h * C + j];
            sum += v;
            top = fmaxf(top, v);
        }
        pool[j] = sum / (float)N_HEXES;
        pool[C + j] = top;
    }
}

/// Broadcast the pooling bias over the row's hexes, then normalise and
/// activate. Both linear biases land here because cuBLAS left them out.
extern "C" __global__ void block_mid(const WaveDev* w, const WeightDev* wt,
                                     int block, int cells) {
    int lane = threadIdx.x & 31;
    int cell = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (cell >= cells) return;
    float* y = AP(w, A_A) + (unsigned long long)cell * C;
    const float* gb = AP(w, A_GB) + (unsigned long long)(cell / N_HEXES) * C;
    const float* mb = wt->mix_b[block];
    const float* pb = wt->pool_b[block];
    norm_row<C, true>([&](int j) { return y[j] + gb[j] + mb[j] + pb[j]; }, y,
                      wt->mid_lnw[block], wt->mid_lnb[block], lane);
}

extern "C" __global__ void block_out(const WaveDev* w, const WeightDev* wt,
                                     int block, int cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells * C) return;
    AP(w, A_X)[i] += AP(w, A_MIX)[i] + wt->out_b[block][i % C];
}

/// Normalise the final trunk output in shared memory and pool it directly.
extern "C" __global__ void board_pool(const WaveDev* w, const WeightDev* wt,
                                      int row0, int n) {
    __shared__ float a[N_HEXES * C];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int r = blockIdx.x;
    if (r >= n) return;

    const float* x = AP(w, A_X) + (unsigned long long)r * N_HEXES * C;
    for (int h = warp; h < N_HEXES; h += warps) {
        norm_row<C, true>([&](int j) { return x[h * C + j]; }, a + h * C,
                          wt->pre_lnw[BLOCKS], wt->pre_lnb[BLOCKS], lane);
    }
    __syncthreads();

    int width = 2 * C + LOOSE;
    float* out = AP(w, A_BOARD) + (unsigned long long)r * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        if (j < 2 * C) {
            int c = j % C;
            float sum = a[c], top = a[c];
            for (int h = 1; h < N_HEXES; h++) {
                float v = a[h * C + c];
                sum += v;
                top = fmaxf(top, v);
            }
            out[j] = j < C ? sum / (float)N_HEXES : top;
        } else {
            int row = row0 + r;
            out[j] = loose_feature(raw_row(w, row), 0, j - 2 * C);
        }
    }
}

extern "C" __global__ void board_bias(const WaveDev* w, const WeightDev* wt,
                                       int row0, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * D) return;
    AP(w, A_P)[(unsigned long long)row0 * D + i] += wt->board_b[i % D];
}

// ----------------------------------------------------------- config encoder

/// One slot token per config: its three zone counts and the printed card of
/// that slot, read from the owning seat's view of the card table.
extern "C" __global__ void config_pack(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->ncfg * NSLOT) return;
    int cfg = i / NSLOT, k = i % NSLOT;
    int job = TP(w, unsigned int, T_CONFIG_JOB)[cfg];
    int owner = TP(w, unsigned char, T_CONFIG_PLAYER)[cfg];
    const float* phi = TP(w, float, T_CPHI) + (unsigned long long)cfg * CFEAT;
    float* out = AP(w, A_PACK) + (unsigned long long)i * (3 + TYPE);
    out[0] = phi[k];
    out[1] = phi[NSLOT + k];
    out[2] = phi[2 * NSLOT + k];
    const float* card = AP(w, A_CARDS)
        + (((unsigned long long)job * 2 + owner) * NTYPE + k) * TYPE;
    for (int j = 0; j < TYPE; j++) out[3 + j] = card[j];
}

/// The config encoder is a sum over slot tokens, which is what makes it
/// invariant to the order the draft put the coins in.
extern "C" __global__ void slot_sum(const WaveDev* w, const WeightDev* wt) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int cfg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (cfg >= w->ncfg) return;
    const float* src = AP(w, A_HIDDEN) + (unsigned long long)cfg * NSLOT * CFGH;
    float* out = AP(w, A_CFG) + (unsigned long long)cfg * CFGH;
    for (int j = lane; j < CFGH; j += 32) {
        float v = 0.0f;
        for (int k = 0; k < NSLOT; k++) v += src[k * CFGH + j];
        out[j] = v;
    }
}

/// Config encoder hidden normalisation.
extern "C" __global__ void config_norm(const WaveDev* w, const WeightDev* wt,
                                       int rows) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    float* x = AP(w, A_CFG) + (unsigned long long)row * CFGH;
    norm_row<CFGH, true>([&](int j) { return x[j]; }, x, wt->cfg_lnw,
                         wt->cfg_lnb, lane);
}

/// Finish both config outputs. `f` only wants its bias; `g` also gets the
/// count-weighted sum of per-zone card embeddings, which is linear in the
/// counts and therefore survives the belief pooling as the expected holding of
/// every card, bound to that card rather than to a slot position.
extern "C" __global__ void config_finish(const WaveDev* w, const WeightDev* wt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->ncfg * (D + POOL)) return;
    int j = i % (D + POOL), cfg = i / (D + POOL);
    if (j < D) {
        AP(w, A_F)[(unsigned long long)cfg * D + j] += wt->cfg_f_b[j];
        return;
    }
    j -= D;
    int job = TP(w, unsigned int, T_CONFIG_JOB)[cfg];
    int owner = TP(w, unsigned char, T_CONFIG_PLAYER)[cfg];
    const float* phi = TP(w, float, T_CPHI) + (unsigned long long)cfg * CFEAT;
    const float* bag = AP(w, A_BAG)
        + ((unsigned long long)job * 2 + owner) * NTYPE * 3 * POOL + j;
    float v = wt->cfg_g_b[j];
    for (int k = 0; k < NSLOT; k++) {
        const float* zone = bag + (unsigned long long)k * 3 * POOL;
        #pragma unroll
        for (int z = 0; z < 3; z++) v += phi[z * NSLOT + k] * zone[z * POOL];
    }
    AP(w, A_G)[(unsigned long long)cfg * POOL + j] += v;
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
    // Both table bases are loop-invariant, but a store through the arena can
    // alias `w` as far as the compiler knows, so it reloads them on every
    // iteration unless they are named here.
    unsigned int rr = TP(w, unsigned int, T_REV_ROW_OF)[node];
    int base = reach_at(w, parent, player, 0);
    if (rr != NONE) {
        unsigned int row = rr + c;
        unsigned int lo = TP(w, unsigned int, T_REV_START)[row];
        unsigned int hi = TP(w, unsigned int, T_REV_START)[row + 1];
        const unsigned int* src = TP(w, unsigned int, T_REV_SRC);
        const unsigned int* cell = TP(w, unsigned int, T_REV_CELL);
        for (unsigned int x = lo; x < hi; x++)
            value += reach[base + src[x]] * strat[cell[x]];
    } else {
        unsigned int row = TP(w, unsigned int, T_RVD_ROW_OF)[node] + c;
        unsigned int lo = TP(w, unsigned int, T_RVD_START)[row];
        unsigned int hi = TP(w, unsigned int, T_RVD_START)[row + 1];
        const unsigned int* src = TP(w, unsigned int, T_RVD_SRC);
        const float* p = TP(w, float, T_RVD_P);
        for (unsigned int x = lo; x < hi; x++)
            value += reach[base + src[x]] * p[x];
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
    const unsigned int* row_of = TP(w, unsigned int, T_LEGAL_ROW_OF);
    const unsigned int* legal = TP(w, unsigned int, T_LEGAL_OFF);
    const float* reach = AP(w, A_REACH);
    const float* cur = AP(w, A_CUR);
    float* sum = AP(w, A_SUM);
    int decisions_n = w->decision_n[player];
    for (int i = thread; i < decisions_n; i += stride) {
        Task q = decisions[i];
        unsigned int row = row_of[q.node] + q.config;
        float r = reach[reach_at(w, q.node, player, q.config)];
        for (unsigned int x = legal[row]; x < legal[row + 1]; x++)
            sum[x] += r * cur[x];
    }
}

extern "C" __global__ void seed_sum(const WaveDev* w, const WeightDev* wt,
                                     int player) {
    (void)wt;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w->decision_n[player]) return;
    const Task q = TP(w, Task, player ? T_DECISION1 : T_DECISION0)[i];
    unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[q.node] + q.config;
    const unsigned int* legal = TP(w, unsigned int, T_LEGAL_OFF);
    const float* cur = AP(w, A_CUR);
    float* sum = AP(w, A_SUM);
    float r = AP(w, A_REACH)[reach_at(w, q.node, player, q.config)];
    for (unsigned int x = legal[row]; x < legal[row + 1]; x++)
        sum[x] = r * cur[x];
}

// ---------------------------------------------------------------- the join

/// The belief-weighted pooled config embedding of every canonical query,
/// indexed by `2 * row + player`. This and the reach mass are all the network
/// reads that move between CFR iterations.
extern "C" __global__ void belief_sums(const WaveDev* w, const WeightDev* wt,
                                        int traverser, int both) {
    (void)wt;
    int lane = threadIdx.x & 31;
    int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= w->nleaf) return;
    ReadTask q = TP(w, ReadTask, T_READOUT)[task];
    // Query indexing means a side's block is still exactly what it was, so
    // between alternating iterations only the player whose strategy just moved
    // has to be summed again.
    for (int side = 0; side < (both ? 2 : 1); side++) {
        int p = both ? side : 1 - traverser;
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
        float acc[POOL / 32];
        #pragma unroll
        for (int z = 0; z < POOL / 32; z++) acc[z] = 0.0f;
        // The config list and the embedding table are loop-invariant; the
        // reduction below writes through the arena, so the compiler will not
        // hoist them on its own.
        const unsigned int* cfgs = TP(w, unsigned int, T_ROW_CFG)
            + TP(w, unsigned int, T_ROW_CFG_OFF)[2 * q.row + p];
        const float* embed = AP(w, A_G);
        for (int c = 0; c < n; c++) {
            const float* g = embed + (unsigned long long)cfgs[c] * POOL;
            float wc = r[c] * scale + flat;
            #pragma unroll
            for (int k = 0; k < POOL / 32; k++) acc[k] += wc * g[(k << 5) + lane];
        }
        float* out = AP(w, A_POOLED) + (unsigned long long)(2 * q.row + p) * POOL;
        #pragma unroll
        for (int k = 0; k < POOL / 32; k++) out[(k << 5) + lane] = acc[k];
    }
}

/// Both operands of the join's first layer: the moving input, and the seed
/// `z` = cached `join_p(P)` plus the layer's bias, which the GEMM then
/// accumulates onto.
extern "C" __global__ void join_input(const WaveDev* w, const WeightDev* wt,
                                       int traverser, int rows) {
    int width = JOIN_IN + JW;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    int j = i % width, r = i / width;
    int q = 2 * r + traverser;
    if (j < 2 * POOL) {
        int side = j < POOL ? q : q ^ 1;
        AP(w, A_JIN)[(unsigned long long)r * JOIN_IN + j] =
            AP(w, A_POOLED)[(unsigned long long)side * POOL + j % POOL];
        return;
    }
    if (j < JOIN_IN) {
        AP(w, A_JIN)[(unsigned long long)r * JOIN_IN + j] =
            traverser ? 1.0f : -1.0f;
        return;
    }
    j -= JOIN_IN;
    AP(w, A_Z)[(unsigned long long)r * JW + j] =
        AP(w, A_JP)[(unsigned long long)r * JW + j] + wt->join_b_b[j];
}

/// One join residual block's pre-activation. The block's bias does not depend
/// on the GEMM, so it goes onto the residual stream here and the GEMM
/// accumulates straight onto `z`.
extern "C" __global__ void join_block(const WaveDev* w, const WeightDev* wt,
                                       int block, int rows) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    float* z = AP(w, A_Z) + (unsigned long long)row * JW;
    float* t = AP(w, A_JT) + (unsigned long long)row * JW;
    norm_row<JW, true>([&](int j) { return z[j]; }, t, wt->join_lnw[block],
                       wt->join_lnb[block], lane);
    const float* bias = wt->join_w_b[block];
    for (int j = lane; j < JW; j += 32) z[j] += bias[j];
}

/// The last join normalisation, and the seed of `h` = `P[q]` plus the output
/// layer's bias. Both sit between the final residual block and the output
/// GEMM, and both are one row per query.
extern "C" __global__ void join_finish(const WaveDev* w, const WeightDev* wt,
                                        int traverser, int rows) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    float* z = AP(w, A_Z) + (unsigned long long)row * JW;
    norm_row<JW, true>([&](int j) { return z[j]; }, z, wt->jout_lnw,
                       wt->jout_lnb, lane);
    const float* p = AP(w, A_P) + (unsigned long long)row * D;
    float* h = AP(w, A_H) + (unsigned long long)row * D;
    for (int j = lane; j < D; j += 32) h[j] = p[j] + wt->join_out_b[j];
}

/// One block values `READOUT_TILE` consecutive leaf tasks. Consecutive tasks
/// are neighbouring leaves of the same subgame and draw their configs from the
/// same interned pool, so tiling them turns most of the readout stream into L1
/// hits. The value itself is one dot product: `<f(c), h> + bias`.
extern "C" __global__ void readout(const WaveDev* w, const WeightDev* wt,
                                    int player) {
    constexpr int WPT = (WAVE_BLOCK / 32) / READOUT_TILE;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int slot = warp / WPT, sub = warp % WPT;
    int task = blockIdx.x * READOUT_TILE + slot;
    // The readout is H's only consumer, so its final LayerNorm lands directly
    // in block-resident storage instead of taking a separate global round trip.
    __shared__ float common[READOUT_TILE][D];
    __shared__ float opp_reach[READOUT_TILE];

    // Every thread reaches the one barrier below, live task or not.
    bool live = task < w->readout_n;
    ReadTask q = live ? TP(w, ReadTask, T_READOUT)[task] : ReadTask{0, NONE};
    int n = live ? nc_of(w, q.node, player) : 0;
    if (live && q.row == NONE) {
        if (sub == 0) {
            int opp = 1 - player;
            const float* ro = AP(w, A_REACH) + reach_at(w, q.node, opp, 0);
            int nop = nc_of(w, q.node, opp);
            float orc = 0.0f;
            for (int c = lane; c < nop; c += 32) orc += ro[c];
            orc = warp_sum(orc);
            if (lane == 0) opp_reach[slot] = orc;
        }
    } else if (live) {
        const float* h = AP(w, A_H) + (unsigned long long)q.row * D;
        if (sub == 0)
            norm_row<D, false>([&](int j) { return h[j]; }, common[slot],
                               wt->h_lnw, wt->h_lnb, lane);
        // `belief_sums` parked the opponent reach mass in the value slot this
        // kernel is about to overwrite, so it has to be read before any warp
        // starts writing. The barrier below is that ordering.
        if (sub == 0 && lane == 0)
            opp_reach[slot] = AP(w, player ? A_SNAP_REACH : A_VALS)
                [value_at(w, q.node, player, 0)];
    }
    __syncthreads();
    if (!live) return;

    float* out = AP(w, A_VALS) + value_at(w, q.node, player, 0);
    float orc = opp_reach[slot];
    if (q.row == NONE) {
        float u = TP(w, float, T_NODE_UTILITY)[q.node];
        if (TP(w, unsigned char, T_NODE_PLAYER)[q.node] != player) u = -u;
        for (int c = sub * 32 + lane; c < n; c += WPT * 32) out[c] = u * orc;
        return;
    }

    float bias = *wt->value_bias;
    const unsigned int* cfgs = TP(w, unsigned int, T_ROW_CFG)
        + TP(w, unsigned int, T_ROW_CFG_OFF)[2 * q.row + player];
    const float* readouts = AP(w, A_F);
    const float* h = common[slot];
    for (int c = sub; c < n; c += WPT) {
        const float4* f = reinterpret_cast<const float4*>(
            readouts + (unsigned long long)cfgs[c] * D);
        float value = 0.0f;
        #pragma unroll
        for (int j = lane; j < D / 4; j += 32) {
            float4 row = f[j];
            int b = 4 * j;
            value += row.x * h[b] + row.y * h[b + 1]
                   + row.z * h[b + 2] + row.w * h[b + 3];
        }
        value = warp_sum(value);
        if (lane == 0) out[c] = (value + bias) * orc;
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
            const float* draw_p = TP(w, float, T_DRAW_P) + d0;
            const unsigned int* draw_to = TP(w, unsigned int, T_DRAW_TO) + d0;
            unsigned int cbase = value_at(w, child, player, 0);
            float v = 0.0f;
            for (unsigned int x = lo + lane; x < hi; x += 32)
                v += draw_p[x] * vals[cbase + draw_to[x]];
            v = warp_sum(v);
            if (lane == 0) vals[vbase + c] = v;
        }
        return;
    }
    if (actor != player) {
        float v = 0.0f;
        const unsigned int* cs = TP(w, unsigned int, T_NODE_CHILD_START);
        const unsigned int* ch = TP(w, unsigned int, T_NODE_CHILD);
        const unsigned int* vbases = TP(w, unsigned int, T_VALS_BASE);
        for (unsigned int x = cs[node] + lane; x < cs[node + 1]; x += 32)
            v += vals[vbases[2 * ch[x] + player] + c];
        v = warp_sum(v);
        if (lane == 0) vals[vbase + c] = v;
        return;
    }
    unsigned int row = TP(w, unsigned int, T_LEGAL_ROW_OF)[node] + c;
    unsigned int lo = TP(w, unsigned int, T_LEGAL_OFF)[row];
    unsigned int hi = TP(w, unsigned int, T_LEGAL_OFF)[row + 1];
    const unsigned int* cell = TP(w, unsigned int, T_LEGAL_VALUE);
    const float* strat = AP(w, mode ? A_SNAP_STRAT : A_CUR);
    float base = 0.0f;
    for (unsigned int x = lo + lane; x < hi; x += 32) {
        unsigned int value = cell[x];
        if (value != NONE) base += vals[value] * strat[x];
    }
    base = warp_sum(base);
    if (lane == 0) vals[vbase + c] = base;
    if (mode) return;
    // The three strategy arenas are indexed by the same cell and written back
    // in this loop; naming them keeps their bases out of the load stream.
    float* regret = AP(w, A_REGRET);
    float* cur = AP(w, A_CUR);
    float* sum = AP(w, A_SUM);
    float total = 0.0f;
    for (unsigned int x = lo + lane; x < hi; x += 32) {
        unsigned int value = cell[x];
        float delta = (value != NONE ? vals[value] : 0.0f) - base;
        float old = regret[x];
        float r = old * (old > 0.0f ? da : db) + delta;
        regret[x] = r;
        float v = fmaxf(r + predict * delta, EPS);
        cur[x] = v;
        total += v;
        sum[x] *= ds;
    }
    total = warp_sum(total);
    if (total > 0.0f) {
        float inv = 1.0f / total;
        for (unsigned int x = lo + lane; x < hi; x += 32) cur[x] *= inv;
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
