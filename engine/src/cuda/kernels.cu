#include <cuda_fp16.h>
#include <mma.h>

// The trunk's matrix multiplies. `wmma` is a C++ interface, so the alias
// lives out here rather than inside the `extern "C"` block below.
namespace wm = nvcuda::wmma;

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

__device__ __forceinline__ float warp_sum(float v) {
    for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
    return v;
}

__device__ __forceinline__ float gelu1(float x) {
    // tanh approximation, matching `net::gelu`.
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__global__ void k_gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu1(x[i]);
}

// LayerNorm, `src` to `dst`; the two may be the same buffer.
//
// One row per **warp**, not per block. The rows here are 96 to 256 wide, so a
// block-wide reduction spends most of its time in `__syncthreads` for a sum a
// warp can shuffle in five steps -- and at 96 wide a 128-thread block leaves a
// quarter of its lanes idle throughout. Measured at a third of all device
// time, it was the largest kernel in the profile.
//
// `act` folds in the GELU, which is `Norm::apply`; without it this is
// `Norm::plain`.
// `add` is the bias of whichever matrix multiply produced `src`, folded in
// here rather than paid for with a pass of its own. A residual stream is added
// to, so the bias a block owes it is never actually stored: `add` carries the
// running sum of every bias the stream has been owed so far, and the norm is
// the only thing that reads the stream.
//
// Rows are ninety-six to two hundred and fifty-six wide, so one row is one
// **warp**, not one block: a block-wide reduction spends its time in
// `__syncthreads` for a sum a warp shuffles in five steps.
__device__ __forceinline__ void norm_row(const float* src, float* dst,
                                         const float* gamma, const float* beta,
                                         const float* add, int has, int rows,
                                         int width, int act) {
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    const float* in = src + (size_t)r * width;
    float* out = dst + (size_t)r * width;
    // At most eight values a lane, so the row is read once and kept.
    float v[8];
    int n = 0;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += 32) {
        float x = in[j] + (has ? add[j] : 0.0f);
        v[n++] = x;
        sum += x;
    }
    for (int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    float mean = sum / width;
    float var = 0.0f;
    for (int k = 0; k < n; ++k) {
        float d = v[k] - mean;
        var += d * d;
    }
    for (int s = 16; s > 0; s >>= 1) var += __shfl_xor_sync(0xffffffff, var, s);
    float inv = rsqrtf(var / width + 1e-5f);
    n = 0;
    for (int j = threadIdx.x; j < width; j += 32) {
        float y = (v[n++] - mean) * inv * gamma[j] + beta[j];
        out[j] = act ? gelu1(y) : y;
    }
}

__global__ void k_norm(const float* src, float* dst, const float* gamma,
                       const float* beta, const float* add, int has, int rows,
                       int width, int act) {
    norm_row(src, dst, gamma, beta, add, has, rows, width, act);
}

// The same, in place. A separate entry only because one buffer cannot be
// handed to a launch as both an argument to read and an argument to write.
__global__ void k_norm_ip(float* x, const float* gamma, const float* beta,
                          const float* add, int has, int rows, int width, int act) {
    norm_row(x, x, gamma, beta, add, has, rows, width, act);
}

// `out[r, j] += b[j]` -- the per-column bias a GEMM does not carry.
//
// Row down `blockIdx.x`, column across the block. A flat index would need
// `i / width` and `i % width`, and an integer division is twenty-odd cycles
// against the one add this kernel exists to do: over a round's four hundred
// thousand join rows that was a hundred and thirty million divisions.
__global__ void k_bias(float* out, const float* b, int rows, int width) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float* row = out + (size_t)r * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) row[j] += b[j];
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


// The trunk's eight residual blocks, with the board resident in shared memory.
//
// This is the kernel that decides whether the throughput target is reachable.
// Written as separate launches a block is eighteen passes over `[37 hexes, 96
// channels]` of global memory -- two norms, a neighbour mix, two matrix
// multiplies, a pool, a group bias and an accumulate, each reading and writing
// the whole board. That is about two megabytes per leaf over eight blocks, and
// at a hundred and fifty solves a second with ~5,700 leaves apiece it comes to
// 1.75 TB/s against the 1.87 the two cards have.
//
// A whole board is 37 x 96 = 3,552 numbers. It fits in shared memory, so the
// board is read once, worked on eight times where it sits, and handed back as
// the pooled row the head wants -- about eight hundred bytes. The weights are
// the same for every leaf and come to 442 KB in half over the eight blocks,
// which L2 holds.
//
// One CUDA block is one leaf. `threadIdx.x` is the channel and there are
// exactly `TRUNK_C` of them over three lanes' worth, so a thread owns three
// numbers of every row it touches; `threadIdx.y` sweeps the hexes.
//
// **The residual stream lives in registers.** A thread owns the channels
// `lane, lane + 32, lane + 64` of each of its hexes, and that is exactly how
// every elementwise step of the trunk reads it: the two norms are a warp
// shuffle over a row a warp owns, and the accumulate at the foot of a block is
// a register add. Only the matrix multiplies want the board laid out in
// memory, and they want it in half, so shared memory holds the half copy they
// read and the single-precision result they write, and nothing else.
//
// `off` is the weight plan, `TRUNK_OFF` entries a block and two after them:
//   0 mix.w  1 mix.b  2 pool.w  3 pool.b  4 out.w  5 out.b
//   6 ln0.g  7 ln0.b  8 ln1.g  9 ln1.b     then trunk.ln.g, trunk.ln.b
// Entries ten and eleven are where that block's mix and out matrices sit in
// `wh`, the half copy `k_trunk_half` packs.
#define TRUNK_OFF 12
// A thread owns this many channels, and this many hexes at most.
//
// One channel a thread meant one shared-memory load for every fused multiply,
// and the SM issues one instruction per clock per partition against thirty-two
// lanes of arithmetic -- so half the machine went on addressing. Three channels
// a thread turns that into three loads for twelve multiplies. Thirty-two
// threads a hex also makes a row exactly one warp wide, so a LayerNorm is a
// shuffle with no barrier in it.
#define TRUNK_Q 3
#define TRUNK_C (32 * TRUNK_Q)
// Hexes a thread owns, and warps in a block. Both are compile-time, and that
// is the whole point: a loop whose trip count the compiler cannot see forces
// the residual stream out of registers, and a trunk whose stream lives in
// memory runs at a few per cent of the card. `37 = 3*12 + 1`, so one warp
// holds four hexes and the rest hold three; the dead ones are masked rather
// than skipped, because a runtime bound is the thing being avoided.
#define TRUNK_SPAN 12
#define TRUNK_MAXH 4
// Thirty-seven hexes padded to forty-eight, which is what the warps cover
// anyway and is three tiles of sixteen rows. The eleven dead rows are zeroed
// once and multiply to nothing.
#define TRUNK_M (TRUNK_SPAN * TRUNK_MAXH)
// A row of the half board carries the hex's own channels and then its
// neighbours' sum, because that is the mix's input. The eight of padding is
// what spreads consecutive rows across the shared memory banks.
#define TRUNK_LDA (2 * TRUNK_C + 8)
// A row of a multiply's result. Four floats of padding, for the same reason.
#define TRUNK_LDO (TRUNK_C + 4)
// The multiply's tile, which is the shape the tensor cores take.
#define TRUNK_TILE 16
#define TRUNK_TILES ((TRUNK_M / TRUNK_TILE) * (TRUNK_C / TRUNK_TILE))

// Mean and inverse standard deviation of one hex's row, held `TRUNK_Q` values
// to a lane across one warp. Two passes, as `norm_row` does, because the
// one-pass form loses the difference between two nearly equal sums.
__device__ __forceinline__ void row_stats(const float* v, float* mean, float* inv) {
    float s = 0.0f;
    for (int q = 0; q < TRUNK_Q; ++q) s += v[q];
    s = warp_sum(s);
    float m = s / TRUNK_C, t = 0.0f;
    for (int q = 0; q < TRUNK_Q; ++q) {
        float d = v[q] - m;
        t += d * d;
    }
    t = warp_sum(t);
    *mean = m;
    *inv = rsqrtf(t / TRUNK_C + 1e-5f);
}

/// One residual block's multiply, `[TRUNK_M, k] x [k, TRUNK_C]`, in half with
/// single-precision accumulation.
///
/// `a` is the half board in shared memory, `TRUNK_LDA` to a row; `b` is a
/// weight matrix in global memory, `TRUNK_C` to a row, which L2 serves because
/// every leaf on the card reads the same one; `o` takes the result in single
/// precision, `TRUNK_LDO` to a row.
///
/// Eighteen output tiles over twelve warps, so six warps take a second one.
/// In single precision this loop was three global loads and four shared reads
/// for twelve fused multiplies, and the trunk ran at a sixth of the card;
/// a tile is one `ldmatrix` pair for two hundred and fifty-six.
///
/// The summation order is the tile order and the tile order is the shape, so
/// this multiply gives the same answer whatever else the round carries. That
/// is what cuBLAS cannot promise -- it picks an algorithm by shape and a
/// round's shape moves with its batch -- and it is the reason this is written
/// out rather than called.
__device__ __forceinline__ void tiles(const __half* a, const __half* b, float* o,
                                      int k, int warp) {
    for (int t = warp; t < TRUNK_TILES; t += TRUNK_SPAN) {
        int m = t / (TRUNK_C / TRUNK_TILE), n = t % (TRUNK_C / TRUNK_TILE);
        wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
        wm::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk < k; kk += TRUNK_TILE) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> av;
            wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::row_major> bv;
            wm::load_matrix_sync(av, a + (size_t)m * TRUNK_TILE * TRUNK_LDA + kk, TRUNK_LDA);
            wm::load_matrix_sync(bv, b + (size_t)kk * TRUNK_C + n * TRUNK_TILE, TRUNK_C);
            wm::mma_sync(acc, av, bv, acc);
        }
        wm::store_matrix_sync(o + (size_t)m * TRUNK_TILE * TRUNK_LDO + n * TRUNK_TILE, acc,
                              TRUNK_LDO, wm::mem_row_major);
    }
}

// Two blocks an SM, said out loud. Left to itself the compiler spends its
// registers freely here and fits one block, giving back in warps what the
// tensor cores bought; naming the block count fixes the budget instead.
__global__ __launch_bounds__(32 * TRUNK_SPAN, 2)
void k_trunk(const float* x0, const int* nb, const float* __restrict__ w,
             const __half* __restrict__ wh, const float* __restrict__ bias,
             const float* __restrict__ ln,
             const int* __restrict__ off, const float* xpub, float* out,
             int rows, int nhex, int blocks, int stride, int off_loose,
             int loose) {
    int row = blockIdx.x;
    if (row >= rows) return;
    // A fragment load wants a thirty-two byte aligned address, and every row
    // of both boards is a whole number of them from the base.
    extern __shared__ __align__(32) char smem[];
    __half* ha = (__half*)smem;                   // [TRUNK_M, 2 * TRUNK_C]
    float* ob = (float*)(smem + TRUNK_M * TRUNK_LDA * sizeof(__half));
    float* pooled = ob + TRUNK_M * TRUNK_LDO;     // [2 * TRUNK_C]
    float* gb = pooled + 2 * TRUNK_C;             // [TRUNK_C]

    const int lane = threadIdx.x, slot = threadIdx.y;
    int hex[TRUNK_MAXH];
    bool live[TRUNK_MAXH];
#pragma unroll
    for (int t = 0; t < TRUNK_MAXH; ++t) {
        hex[t] = slot + t * TRUNK_SPAN;
        live[t] = hex[t] < nhex;
    }
    // The residual stream, and the padded rows of the half board, which are
    // zero for good: nothing reads what a dead row multiplies to.
    float x[TRUNK_MAXH][TRUNK_Q];
    float cur[TRUNK_Q];
#pragma unroll
    for (int t = 0; t < TRUNK_MAXH; ++t)
        for (int q = 0; q < TRUNK_Q; ++q) {
            int j = lane + 32 * q;
            x[t][q] = live[t] ? x0[((size_t)row * nhex + hex[t]) * TRUNK_C + j] : 0.0f;
            if (!live[t]) {
                ha[hex[t] * TRUNK_LDA + j] = __float2half(0.0f);
                ha[hex[t] * TRUNK_LDA + TRUNK_C + j] = __float2half(0.0f);
            }
        }

    for (int blk = 0; blk < blocks; ++blk) {
        const int* o = off + blk * TRUNK_OFF;
        // a = gelu(norm(x)), written where the multiply wants it.
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t) {
            float mean, inv;
            row_stats(x[t], &mean, &inv);
            if (live[t])
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    ha[hex[t] * TRUNK_LDA + j] = __float2half(
                        gelu1((x[t][q] - mean) * inv * ln[o[6] + j] + ln[o[7] + j]));
                }
        }
        __syncthreads();
        // The pooled global bias -- mean and max over the hexes, projected
        // once for the whole board -- and each hex's neighbour sum, which is
        // the second half of the mix's input. Both read the first half only,
        // so they run together.
        if (slot == 0) {
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                float sum = 0.0f, mx = neg_inf();
                for (int h = 0; h < nhex; ++h) {
                    float v = __half2float(ha[h * TRUNK_LDA + j]);
                    sum += v;
                    mx = fmaxf(mx, v);
                }
                pooled[j] = sum / nhex;
                pooled[TRUNK_C + j] = mx;
            }
        }
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t)
            if (live[t]) {
                for (int q = 0; q < TRUNK_Q; ++q) cur[q] = 0.0f;
                for (int k = 0; k < 6; ++k) {
                    int n = nb[hex[t] * 6 + k];
                    if (n >= 0)
                        for (int q = 0; q < TRUNK_Q; ++q)
                            cur[q] += __half2float(ha[n * TRUNK_LDA + lane + 32 * q]);
                }
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    ha[hex[t] * TRUNK_LDA + TRUNK_C + j] = __float2half(cur[q]);
                }
            }
        __syncthreads();
        if (slot == 0) {
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                float sv = bias[o[3] + j];
                for (int k = 0; k < 2 * TRUNK_C; ++k)
                    sv += pooled[k] * w[o[2] + (size_t)k * TRUNK_C + j];
                gb[j] = sv;
            }
        }
        __syncthreads();
        tiles(ha, wh + o[10], ob, 2 * TRUNK_C, slot);
        __syncthreads();
        // gelu(norm(mix + bias + the pooled bias)), back into half.
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t) {
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                cur[q] = ob[hex[t] * TRUNK_LDO + j] + bias[o[1] + j] + gb[j];
            }
            float mean, inv;
            row_stats(cur, &mean, &inv);
            if (live[t])
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    ha[hex[t] * TRUNK_LDA + j] = __float2half(
                        gelu1((cur[q] - mean) * inv * ln[o[8] + j] + ln[o[9] + j]));
                }
        }
        __syncthreads();
        tiles(ha, wh + o[11], ob, TRUNK_C, slot);
        __syncthreads();
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t)
            if (live[t])
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    x[t][q] += ob[hex[t] * TRUNK_LDO + j] + bias[o[5] + j];
                }
    }

    // The trunk norm, and the head's input: pooled mean and max, then the
    // loose scalars straight off the public encoding. The stream goes back to
    // memory here only because the pool is across hexes and every warp holds
    // its own.
    const int* tn = off + blocks * TRUNK_OFF;
#pragma unroll
    for (int t = 0; t < TRUNK_MAXH; ++t) {
        float mean, inv;
        row_stats(x[t], &mean, &inv);
        if (live[t])
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                ob[hex[t] * TRUNK_LDO + j] =
                    gelu1((x[t][q] - mean) * inv * ln[tn[0] + j] + ln[tn[1] + j]);
            }
    }
    __syncthreads();
    int width = 2 * TRUNK_C + loose;
    if (slot == 0) {
        for (int q = 0; q < TRUNK_Q; ++q) {
            int j = lane + 32 * q;
            float sum = 0.0f, mx = neg_inf();
            for (int h = 0; h < nhex; ++h) {
                float v = ob[h * TRUNK_LDO + j];
                sum += v;
                mx = fmaxf(mx, v);
            }
            out[(size_t)row * width + j] = sum / nhex;
            out[(size_t)row * width + TRUNK_C + j] = mx;
        }
    }
    for (int k = lane + 32 * slot; k < loose; k += 32 * TRUNK_SPAN)
        out[(size_t)row * width + 2 * TRUNK_C + k] =
            xpub[(size_t)row * stride + off_loose + k];
}

// The half copy of the trunk's matrices, packed a block at a time: the mix
// `[2c, c]` then the out `[c, c]`, both row-major exactly as the net stores
// them. Both sizes are a whole number of tiles, so every matrix starts where a
// fragment load can begin.
//
// It is built on the card from the weights already there rather than on the
// host, so a publish is one launch and the rounding is the card's own.
__global__ void k_trunk_half(const float* __restrict__ w, const int* __restrict__ off,
                             __half* wh, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int mix = 2 * TRUNK_C * TRUNK_C, per = mix + TRUNK_C * TRUNK_C;
    const int* o = off + (i / per) * TRUNK_OFF;
    int j = i % per;
    wh[i] = __float2half(j < mix ? w[o[0] + j] : w[o[4] + j - mix]);
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
//
// The batch is both traversers back to back -- rows `0..stride` ask about
// player zero and `stride..2*stride` about player one, over the same leaves.
// The join is the only part of the pass that depends on which seat is asking,
// so running it once over twice the rows costs the same arithmetic and half
// the launches.
__global__ void k_join_input(const float* pooled, float* out, int rows, int pool,
                             int tile) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int width = 2 * pool + 1;
    // The tile holds `tile` leaves and both of their seats, seat-major, so a
    // leaf's two pooled rows are adjacent whichever seat is asking.
    int p = r / tile, q = r % tile;
    const float* mine = pooled + ((size_t)2 * q + p) * pool;
    const float* theirs = pooled + ((size_t)2 * q + 1 - p) * pool;
    float* dst = out + (size_t)r * width;
    for (int j = threadIdx.x; j < pool; j += blockDim.x) {
        dst[j] = mine[j];
        dst[pool + j] = theirs[j];
    }
    if (threadIdx.x == 0) dst[2 * pool] = p == 0 ? -1.0f : 1.0f;
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


// One packed upload, scattered.
//
// A growth touches a tail of each of thirty arrays describing a tree, and a
// round holds thirty-odd solves. Issued one at a time that is a thousand stream
// operations a round -- more host time than every kernel of the iteration put
// together. They travel concatenated instead: one buffer, and this to put each
// piece where it belongs. `start` is the prefix sum of the pieces, so a thread
// finds its own by bisection.
// `sum` is the prefix sum of the piece lengths, so a thread can find which
// piece it belongs to; `src` says where that piece's words are in the blob.
// The two are separate because one run of words can land in two arrays -- the
// tail a growth appends is both the starting strategy and the prior -- and
// then the same words are read twice.
__global__ void k_scatter(const unsigned int* blob, unsigned int* const* dst,
                          const unsigned int* at, const unsigned int* src,
                          const unsigned int* sum, int pieces, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int lo = 0, hi = pieces;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((int)sum[mid] <= i) lo = mid; else hi = mid;
    }
    int k = i - (int)sum[lo];
    dst[lo][at[lo] + k] = blob[src[lo] + k];
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
    const unsigned int* cell_row;
    const unsigned int* cell_val;
    const unsigned int* rev_base;
    const unsigned int* rev_start;
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
    /// PUCT's counts. Small integers -- a cell is passed at most once per
    /// trajectory and a solve runs a few hundred -- so two bytes hold them
    /// with room to spare, over the largest arrays a solve keeps.
    unsigned short* visits;
    float* prior;
    float* avg;
    const float* rootb;
    // The network state that outlives an iteration: board vectors, the join
    // cache, the config readout and pooling rows, and the belief index.
    const float* p;
    const float* jp;
    // The config readout and pooling rows, in half precision.
    //
    // Both kernels are gathers of half a flop to the byte, so what they cost
    // is what they read, and together they are a third of the device. The
    // numbers are a learned embedding of order one and the dot products still
    // accumulate in single precision, so the width is all that changes -- and
    // twice as many of them fit in L2, where a solve's configs are read again
    // at every one of its iterations.
    //
    // This needs the encoder's multiplies to be shaped the same whoever shares
    // the round -- see `CFG_STEP`. Without that, two orderings that differ in
    // the last bit round to different halves, and regret matching turns that
    // into 1.4e-1 of root policy.
    const __half* f;
    const __half* g;
    const unsigned int* cidx;
    const unsigned int* coff;
    // Batch row -> node, for the leaves the network answers for.
    const unsigned int* leaf_node;
    const unsigned int* term;
    unsigned long long* seed;
    unsigned long long levels;
    unsigned long long nterm;
    /// One value arena per traverser, so both can be backpropagated at once.
    unsigned long long nvals;
    // A round holds solves at different points of their own sixty-four
    // iterations, and a CFR iterate is weighted by how many came before it. So
    // the decay factors are a function of *this* solve's step count and are
    // computed here rather than passed in: one number for the whole batch
    // would be some other solve's weights.
    unsigned long long step;
    /// Iterations this call asks of this solve, and expansion trajectories
    /// after them. Both differ across a round once a solve's tree is full.
    unsigned long long todo;
    unsigned long long nexpand;
};

// `Cfr::factor` in search.rs: an infinite exponent is the CFR+ limit, which
// keeps positive regrets whole and drops negative ones outright.
__device__ __forceinline__ float cfr_factor(float t, float p) {
    if (isinf(p)) return p > 0.0f ? 1.0f : 0.0f;
    float x = powf(t, p);
    return x / (x + 1.0f);
}

#define NO_ROW 0xffffffffu
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

// The counterfactual value of one cell: the slot its child holds for the
// acting player, or zero where the cell reaches no information state of theirs.
__device__ __forceinline__ float action_value(const Tree& t, const float* vals,
                                              unsigned int cell) {
    unsigned int vc = t.cell_val[cell];
    return vc == NO_ROW ? 0.0f : vals[vc];
}

// Where player `p`'s block starts inside node `i`'s reach region.
__device__ __forceinline__ unsigned int rbase(const Tree& t, unsigned int i, int p) {
    return t.roff[i] + (p == 1 ? t.nc[2 * i] : 0);
}

// The root beliefs, before the first level of the sweep reads them.
__global__ void k_seed_reach(const Tree* trees, int iter) {
    const Tree& t = trees[blockIdx.y];
    if ((unsigned long long)iter >= t.todo) return;
    unsigned int n = t.nc[0] + t.nc[1];
    for (unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; c < n;
         c += gridDim.x * blockDim.x)
        t.reach[t.roff[0] + c] = t.rootb[c];
}

// Reach probabilities for one level, both players. `avg` picks the reference
// strategy over the regret-matching iterate, which is what the value pass that
// produces a solve's targets propagates under.
__global__ void k_reach_sweep(const Tree* trees, int level, int avg, int also_sum, int iter) {
    const Tree& t = trees[blockIdx.y];
    if ((unsigned long long)iter >= t.todo) return;
    const float* strat = avg ? t.avg : t.cur;
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
                // The parent cell is the whole entry: which of its configs
                // the cell belongs to is what `cell_row` says. Repeating that
                // in an array of its own was four bytes a cell to save a load
                // at an index the strategy gather gets from the same place.
                for (unsigned int k = a; k < b; ++k) {
                    unsigned int pc = t.rev_cell[k];
                    v += t.reach[src + t.cell_row[pc]] * strat[pc];
                }
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
    if (!also_sum || t.kind[node] != 0) return;
    // The accumulation below reads the reach this sweep has just written, and
    // it reads it by *cell* where the sweep wrote it by *config* -- so a lane
    // reads an address another warp of this block owns. Both early returns
    // above are block-uniform, so every thread reaches this.
    __syncthreads();
    // The reach-weighted iterate, added to the running strategy sum. The reach
    // it needs is the one the loop above has just made current, and the thread
    // that owns a config there owns it here, so this costs a level of launches
    // less than a pass of its own would.
    unsigned int actor = t.player[node];
    unsigned int an = t.nc[2 * node + actor], so = t.soff[node], lb = t.legal_base[node];
    unsigned int ra = rbase(t, node, actor);
    // Flat over the node's cells, not over its configs. A config owns a
    // contiguous run of them, so a thread that walked its own config's run made
    // one memory transaction per lane per step; over `sum` and `cur`, which are
    // the largest arrays a solve holds, that is the difference between a
    // coalesced read and thirty-two scattered ones. `cell_row` says which
    // config a cell belongs to, and the reach it needs is a small gather.
    unsigned int lo = t.legal_off[lb], hi = t.legal_off[lb + an];
    for (unsigned int cell = lo + threadIdx.x; cell < hi; cell += blockDim.x)
        t.sum[so + cell] += t.reach[ra + t.cell_row[so + cell]] * t.cur[so + cell];
}

// Value backpropagation for one level, one traverser. `avg` averages under the
// reference strategy and leaves regrets alone -- the value pass a solve's
// targets come from; otherwise this is the regret update itself.
__global__ void k_backprop_sweep(const Tree* trees, int level, int avg, int iter,
                                 float alpha, float beta, float gamma, float predict) {
    const float EPS = 1e-6f;
    const Tree& t = trees[blockIdx.y];
    if ((unsigned long long)iter >= t.todo) return;
    // This solve's own iterate index, not the round's.
    float m = (float)(t.step + (unsigned long long)iter) + 1.0f;
    float da = cfr_factor(m, alpha), db = cfr_factor(m, beta);
    float dg = powf(m / (m + 1.0f), gamma);
    // The two traversers write disjoint cells and read disjoint value arenas,
    // so they are one launch rather than two.
    int traverser = blockIdx.z;
    float* vals = t.vals + traverser * t.nvals;
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
                    v += t.draw_p[k] * vals[cv + t.draw_to[k]];
                vals[vi + c] = v;
            }
        } else {
            for (unsigned int c = threadIdx.x; c < n; c += blockDim.x)
                vals[vi + c] = vals[cv + c];
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
            for (unsigned int j = a; j < a + k; ++j) v += vals[t.voff[t.child[j]] + c];
            vals[vi + c] = v;
        }
        return;
    }

    unsigned int so = t.soff[node], lb = t.legal_base[node];
    // A warp to a config, lanes across its cells. A thread that owned a config
    // walked that config's own contiguous run, so thirty-two lanes made
    // thirty-two separate transactions per step over `cur`, `regret`
    // and `sum` -- the largest arrays a solve holds. The reductions a config
    // needs are warp shuffles rather than serial sums.
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (avg) {
        for (unsigned int c = warp; c < n; c += warps) {
            unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
            float base = 0.0f;
            for (unsigned int cell = a + lane; cell < b; cell += 32)
                base += action_value(t, vals, so + cell) * t.avg[so + cell];
            base = warp_sum(base);
            if (lane == 0) vals[vi + c] = base;
        }
        return;
    }
    // A cell's action value is `cell_val`'s slot in this traverser's value
    // arena, and both loops below want it. It is re-read rather than kept: an
    // arena of them was four bytes a cell -- one of the largest arrays a solve
    // holds -- to save a load that the first loop has just brought into L1.
    unsigned int ncells = t.legal_off[lb + n];
    for (unsigned int c = warp; c < n; c += warps) {
        unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
        float base = 0.0f;
        for (unsigned int cell = a + lane; cell < b; cell += 32)
            base += action_value(t, vals, so + cell) * t.cur[so + cell];
        base = warp_sum(base);
        if (lane == 0) vals[vi + c] = base;
        float total = 0.0f;
        for (unsigned int cell = a + lane; cell < b; cell += 32) {
            float delta = action_value(t, vals, so + cell) - base;
            float old = t.regret[so + cell];
            float r = old * (old > 0.0f ? da : db) + delta;
            t.regret[so + cell] = r;
            float v = fmaxf(r + predict * delta, EPS);
            t.cur[so + cell] = v;
            total += v;
        }
        total = warp_sum(total);
        if (total > 0.0f) {
            float inv = 1.0f / total;
            for (unsigned int cell = a + lane; cell < b; cell += 32) t.cur[so + cell] *= inv;
        }
    }
    __syncthreads();
    for (unsigned int k = threadIdx.x; k < ncells; k += blockDim.x) t.sum[so + k] *= dg;
}

// The reach-weighted iterate, added to the running strategy sum. Both players
// in one pass: a decision node belongs to exactly one of them.
__global__ void k_avg_block(const Tree* trees, int level, int iter) {
    const Tree& t = trees[blockIdx.y];
    if ((unsigned long long)iter >= t.todo) return;
    unsigned int node;
    if (!level_task(t, level, blockIdx.x, &node)) return;
    if (t.kind[node] != 0) return;
    unsigned int actor = t.player[node];
    unsigned int an = t.nc[2 * node + actor], so = t.soff[node], lb = t.legal_base[node];
    unsigned int ra = rbase(t, node, actor);
    // Flat over the node's cells, not over its configs. A config owns a
    // contiguous run of them, so a thread that walked its own config's run made
    // one memory transaction per lane per step; over `sum` and `cur`, which are
    // the largest arrays a solve holds, that is the difference between a
    // coalesced read and thirty-two scattered ones. `cell_row` says which
    // config a cell belongs to, and the reach it needs is a small gather.
    unsigned int lo = t.legal_off[lb], hi = t.legal_off[lb + an];
    for (unsigned int cell = lo + threadIdx.x; cell < hi; cell += blockDim.x)
        t.sum[so + cell] += t.reach[ra + t.cell_row[so + cell]] * t.cur[so + cell];
}

// Every leaf's normalised belief, and the opponent's reach mass there. One
// block per (row, player); `w` and `oppmass` are what the network reads.
__global__ void k_beliefs(const Tree* trees, const int* part_of_row,
                          const int* local_row, const unsigned int* coff, float* w,
                          float* mass, int rows) {
    // A warp to a query, eight warps to a block. One warp per block left an SM
    // holding sixteen of them and a quarter of its lanes busy, on a kernel that
    // is nothing but dependent gathers and wants every warp it can get.
    int r = blockIdx.x * blockDim.y + threadIdx.y;
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
    if (threadIdx.x == 0) mass[(size_t)p * rows + r] = acc;
}

// Gather each solve's resident board vectors into the round's layout, so the
// join is one chain of large GEMMs over every solve in flight.
__global__ void k_gather(const Tree* trees, const int* part_of_row,
                         const int* local_row, int which, float* out,
                         int rows, int width, int q0, int tile) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int r = q0 + row % tile;
    const Tree& t = trees[part_of_row[r]];
    const float* src = (which == 0 ? t.p : t.jp) + (size_t)local_row[r] * width;
    float* dst = out + (size_t)row * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) dst[j] = src[j];
}

// The belief block the join reads: `sum_c beta(c) g(c)` over one query's
// support. `coff` bounds a query's cells in the round's `w`, `cidx` names each
// cell's row in its own solve's `g`.
// The belief-weighted pooling of a query's configs.
//
// `threadIdx.y` splits the support, because that is where the parallelism is:
// the row is only `pool` wide and the sum runs over a hundred-odd configs, each
// a dependent gather. One thread per output channel walked the whole support
// serially and left the block waiting on memory.
// Narrow a run of floats into a half-precision arena.
__global__ void k_narrow(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void k_belief_pool(const Tree* trees, const int* part_of_row,
                              const int* base_of_part, const unsigned int* coff,
                              const float* w, float* out, int q0, int queries,
                              int pool) {
    extern __shared__ float part_acc[];
    int mine = blockIdx.x;
    if (mine >= queries) return;
    int q = q0 + mine;
    int part = part_of_row[q >> 1];
    const Tree& t = trees[part];
    unsigned int base = base_of_part[part], lo = coff[q], hi = coff[q + 1];
    int j = threadIdx.x, y = threadIdx.y, ny = blockDim.y;
    float acc = 0.0f;
    for (unsigned int k = lo + y; k < hi; k += ny)
        acc += w[k] * __half2float(t.g[(size_t)t.cidx[k - base] * pool + j]);
    part_acc[y * pool + j] = acc;
    __syncthreads();
    if (y == 0) {
        float total = 0.0f;
        for (int i = 0; i < ny; ++i) total += part_acc[i * pool + j];
        out[(size_t)mine * pool + j] = total;
    }
}

// `v(c) = (<f(c), h_row> + bias) * opp_reach[row]`, for every config of the
// queried player at every leaf of the batch, written straight into that
// solve's own value arena.
//
// One block per row, one config per warp, and the row's head vector staged in
// shared memory so it is read once for the row rather than once for each of its
// hundred-odd configs.
__global__ void k_readout(const Tree* trees, const int* part_of_row,
                          const int* local_row, const unsigned int* coff,
                          const float* h, const float* cf_bias, const float* mass,
                          const float* add, const float* gamma, const float* beta,
                          int rows, int stride, int d, int q0, int tile) {
    extern __shared__ float hs[];
    __shared__ float red[8];
    int row = blockIdx.x;
    if (row >= rows) return;
    // `row` indexes this tile's join rows; `traverser` and `r` are the round's.
    int traverser = row / tile, r = q0 + row % tile;
    int tid = threadIdx.x + 32 * threadIdx.y, nt = 32 * blockDim.y;
    // The head's own LayerNorm, done here rather than in a pass of its own.
    // This is the only reader of `h`, and it stages the row in shared memory
    // anyway, so normalising it costs one reduction instead of a read and a
    // write of every head vector in the round.
    // The head's residual seed is this leaf's board vector, added here rather
    // than copied into `h` by a gather of its own -- `h` is written once by the
    // join's last multiply and read once, here.
    const float* seed = trees[part_of_row[r]].p + (size_t)local_row[r] * d;
    float sum = 0.0f;
    for (int j = tid; j < d; j += nt) {
        float x = h[(size_t)row * d + j] + seed[j] + add[j];
        hs[j] = x;
        sum += x;
    }
    for (int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    if (threadIdx.x == 0) red[threadIdx.y] = sum;
    __syncthreads();
    float total = 0.0f;
    for (int k = 0; k < blockDim.y; ++k) total += red[k];
    float mean = total / d, var = 0.0f;
    for (int j = tid; j < d; j += nt) {
        float dv = hs[j] - mean;
        var += dv * dv;
    }
    for (int s = 16; s > 0; s >>= 1) var += __shfl_xor_sync(0xffffffff, var, s);
    __syncthreads();
    if (threadIdx.x == 0) red[threadIdx.y] = var;
    __syncthreads();
    total = 0.0f;
    for (int k = 0; k < blockDim.y; ++k) total += red[k];
    float inv = rsqrtf(total / d + 1e-5f);
    __syncthreads();
    for (int j = tid; j < d; j += nt) hs[j] = (hs[j] - mean) * inv * gamma[j] + beta[j];
    __syncthreads();
    const Tree& t = trees[part_of_row[r]];
    unsigned int node = t.leaf_node[local_row[r]];
    unsigned int lo = coff[2 * r + traverser], hi = coff[2 * r + traverser + 1];
    unsigned int cs = t.coff[2 * local_row[r] + traverser];
    float bias = *cf_bias, scale = mass[(size_t)(1 - traverser) * stride + r];
    float* vals = t.vals + traverser * t.nvals;
    unsigned int vo = t.voff[node];
    for (unsigned int k = lo + threadIdx.y; k < hi; k += blockDim.y) {
        const __half* fr = t.f + (size_t)t.cidx[cs + (k - lo)] * d;
        float acc = 0.0f;
        for (int j = threadIdx.x; j < d; j += 32) acc += __half2float(fr[j]) * hs[j];
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
        if (threadIdx.x == 0) vals[vo + (k - lo)] = (acc + bias) * scale;
    }
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
//
// A warp does this together: the total is a shuffle reduction, and the scan
// that follows is the same in every lane, so it broadcasts one address at a
// time rather than gathering. Every lane draws from the same stream and takes
// the same turn, which is what keeps the walk below coherent.
__device__ int pick(const float* w, int n, unsigned long long* s) {
    float total = 0.0f;
    for (int i = threadIdx.x; i < n; i += 32) total += fmaxf(w[i], 0.0f);
    total = warp_sum(total);
    if (!(total > 0.0f)) return n > 0 ? (int)(rng_next(s) % (unsigned long long)n) : 0;
    double needle = rng_unit(s) * (double)total;
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
//
// Ties go to the lowest cell, which is what a serial scan keeping the first
// strictly greater score does.
__device__ int puct_choice(const Tree& t, unsigned int node, unsigned int a,
                           unsigned int b, int me, float c_puct) {
    int opp = 1 - me;
    unsigned int so = t.soff[node], ra = rbase(t, node, opp);
    unsigned int nc = t.nc[2 * node + opp];
    float mass = 0.0f;
    for (unsigned int i = threadIdx.x; i < nc; i += 32) mass += t.reach[ra + i];
    mass = warp_sum(mass);
    float scale = mass > 1e-30f ? 1.0f / mass : 0.0f;
    float total = 0.0f;
    for (unsigned int cell = a + threadIdx.x; cell < b; cell += 32)
        total += (float)t.visits[so + cell];
    total = warp_sum(total);
    float explore = c_puct * sqrtf(total);
    // The Q the last backward sweep formed. This node's actor is the traverser
    // whose arena holds it, and nothing has written that arena since.
    const float* vals = t.vals + (unsigned)me * t.nvals;
    int best = (int)a;
    float score = neg_inf();
    for (unsigned int cell = a + threadIdx.x; cell < b; cell += 32) {
        float v = action_value(t, vals, so + cell) * scale
                + explore * t.prior[so + cell] / (1.0f + (float)t.visits[so + cell]);
        if (v > score) { score = v; best = (int)cell; }
    }
    for (int k = 16; k > 0; k >>= 1) {
        float os = __shfl_xor_sync(0xffffffff, score, k);
        int oc = __shfl_xor_sync(0xffffffff, best, k);
        if (os > score || (os == score && oc < best)) { score = os; best = oc; }
    }
    return best;
}

__global__ void k_expand(const Tree* trees, unsigned int* out, int parts,
                         int sims, float c_puct, int iter) {
    int part = blockIdx.x;
    if (part >= parts) return;
    const Tree& t = trees[part];
    // A round runs as many iterations as its longest member asked for, and a
    // solve that has had its share must not be given another phase.
    if ((unsigned long long)iter >= t.todo) {
        for (int sim = threadIdx.x; sim < sims; sim += 32)
            out[part * sims + sim] = NO_ROW;
        return;
    }
    // A solve whose tree has spent its budget asks for no trajectories, and a
    // round holds both kinds. The row is `sims` wide either way so the host can
    // slice it; what this solve did not sample reads as nothing.
    for (int sim = (int)t.nexpand + threadIdx.x; sim < sims; sim += 32)
        out[part * sims + sim] = NO_ROW;
    if (t.nexpand == 0) return;
    unsigned long long s = *t.seed;
    unsigned int n0 = t.nc[0], n1 = t.nc[1];
    for (int sim = 0; sim < (int)t.nexpand; ++sim) {
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
                unsigned int a = t.draw_start[base + c[me]];
                unsigned int b = t.draw_start[base + c[me] + 1];
                int j = pick(t.draw_p + a, (int)(b - a), &s);
                c[me] = (int)t.draw_to[a + j];
                node = t.child[t.child_at[node]];
                continue;
            }
            unsigned int lb = t.legal_base[node], so = t.soff[node];
            unsigned int a = t.legal_off[lb + c[me]], b = t.legal_off[lb + c[me] + 1];
            if (a == b) break;
            int cell;
            // Student of Games selects by half PUCT and half the search's own
            // average: `pi_select = 1/2 pi_PUCT + 1/2 pi_CFR`. PUCT is a
            // maximisation, so its half is a point mass on the argmax, and
            // sampling the mixture is a coin flip between the two.
            if (rng_unit(&s) < 0.5) {
                cell = puct_choice(t, node, a, b, (int)me, c_puct);
            } else {
                bool mine = false;
                for (unsigned int q = a + threadIdx.x; q < b; q += 32)
                    mine |= t.sum[so + q] > 0.0f;
                bool any = __any_sync(0xffffffff, mine);
                const float* row = any ? t.sum + so + a : t.cur + so + a;
                cell = (int)a + pick(row, (int)(b - a), &s);
            }
            // Counted as the trajectory passes, which is also the virtual loss
            // Student of Games adds across the simulations of one iteration:
            // a later simulation of the same phase sees this one's visit.
            if (threadIdx.x == 0) t.visits[so + cell] += 1;
            __syncwarp();
            // `cell_val` is `voff[child] + trans`, so the transition it was
            // built from comes back out of it and needs no array of its own.
            unsigned int vc = t.cell_val[so + cell];
            if (vc == NO_ROW) break;
            node = t.legal_child[so + cell];
            c[me] = (int)(vc - t.voff[node]);
        }
        if (threadIdx.x == 0) out[part * sims + sim] = found;
    }
    if (threadIdx.x == 0) *t.seed = s;
}

// Terminal leaves are scored from the game, not the network, so the batch never
// carried them. One warp per terminal.
__global__ void k_terminals(const Tree* trees) {
    const Tree& t = trees[blockIdx.y];
    unsigned int k = blockIdx.x * blockDim.y + threadIdx.y;
    if (k >= t.nterm) return;
    unsigned int node = t.term[k];
    int traverser = blockIdx.z, opp = 1 - traverser;
    unsigned int n = t.nc[2 * node + opp], ra = rbase(t, node, opp);
    float acc = 0.0f;
    for (unsigned int c = threadIdx.x; c < n; c += 32) acc += t.reach[ra + c];
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, s);
    // Zero-sum by construction, so one stored utility serves both seats.
    float u = t.player[node] == (unsigned)traverser ? t.util[node] : -t.util[node];
    unsigned int m = t.nc[2 * node + traverser], vo = t.voff[node];
    float* vals = t.vals + traverser * t.nvals;
    for (unsigned int c = threadIdx.x; c < m; c += 32) vals[vo + c] = u * acc;
}

// The reference strategy: the normalised running sum, laid out exactly like
// `cur`. Built once, when the tree has stopped growing.
//
// `cur` still holds the literal initial policy for a player that has not
// traversed yet, so a row whose sum has not moved keeps it rather than being
// reconstructed by a multiply and a divide that need not round back.
__global__ void k_finish(const Tree* trees, int level, const int* touched) {
    int mask = touched[blockIdx.y];
    if (mask < 0) return;
    const Tree& t = trees[blockIdx.y];
    unsigned int node;
    if (!level_task(t, level, blockIdx.x, &node)) return;
    if (t.kind[node] != 0) return;
    unsigned int me = t.player[node], so = t.soff[node], lb = t.legal_base[node];
    unsigned int n = t.nc[2 * node + me];
    bool moved = (mask >> me) & 1;
    for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
        unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
        if (!moved) {
            for (unsigned int cell = a; cell < b; ++cell) t.avg[so + cell] = t.cur[so + cell];
            continue;
        }
        float sum = 0.0f;
        for (unsigned int cell = a; cell < b; ++cell) sum += t.sum[so + cell];
        float k = (float)max(b - a, 1u);
        for (unsigned int cell = a; cell < b; ++cell)
            t.avg[so + cell] = sum > 0.0f ? t.sum[so + cell] / sum : 1.0f / k;
    }
}

}  // extern "C"
