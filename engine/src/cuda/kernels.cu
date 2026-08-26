#include <cuda_fp16.h>

// Elementwise and gather work for the value network, and the join, which is
// fused into one kernel of its own. The trunk and the join multiply on the
// tensor cores; every other matrix multiply is cuBLAS.
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

// One packed public row to the exact feature layout `pbs::expand_row` writes.
// Every layout number is supplied by Rust as an NVRTC define.
__global__ void k_expand_rows(const unsigned char* rows, const float* cards,
                              const unsigned char* locations, float* out, int n) {
    int r = blockIdx.x;
    if (r >= n) return;
    const unsigned char* row = rows + (size_t)r * ROW_BYTES;
    float* dst = out + (size_t)r * PUBFEAT;
    for (int j = threadIdx.x; j < PUBFEAT; j += blockDim.x) {
        float v = 0.0f;
        if (j < HEX_BLOCK) {
            int h = j / HEX_CH, ch = j % HEX_CH;
            int owner = row[ROW_HEX_OWNER + h];
            int slot = row[ROW_HEX_SLOT + h];
            int marker = row[ROW_HEX_MARKER + h];
            if (ch == 0) v = owner == 0;
            else if (ch == 1) v = owner == 1;
            else if (ch == 2) v = owner < 2 ? row[ROW_HEX_HEIGHT + h] / 5.0f : 0.0f;
            else if (ch == 3) v = marker == 0;
            else if (ch == 4) v = marker == 1;
            else if (ch == 5) v = locations[h];
            else if (ch < HEX_FACTS) {
                int d = ch - 6;
                int byte = row[ROW_STACK_OWED + 8 * d + h / 8];
                v = (byte >> (h % 8)) & 1;
            } else {
                int type = ch - HEX_FACTS;
                v = owner < 2 && slot < NSLOT && type == owner * NSLOT + slot;
            }
        } else if (j < OFF_CARDS) {
            v = row[ROW_PILES + j - OFF_PILES] / 5.0f;
        } else if (j < OFF_LOOSE) {
            int k = j - OFF_CARDS;
            int type = k / CARD_FEATS, fact = k % CARD_FEATS;
            v = cards[(size_t)row[ROW_IDS + type] * CARD_FEATS + fact];
        } else if (j < OFF_LOOSE + 2 * PLAYER_SCALARS) {
            int k = j - OFF_LOOSE, player = k / PLAYER_SCALARS;
            int field = k % PLAYER_SCALARS, on_board = 0;
            for (int h = 0; h < N_HEXES; ++h)
                on_board += row[ROW_HEX_MARKER + h] == player;
            if (field == 0) v = (6 - on_board) / 6.0f;
            else if (field == 1) v = on_board / 6.0f;
            else if (field == 2) v = row[ROW_HAND_SIZE + player] / 3.0f;
            else if (field == 3) v = row[ROW_FD_SIZE + player] / MAX_COINS;
            else if (field == 4) v = row[ROW_BAG_SIZE + player] / MAX_COINS;
            else {
                int initiative = row[ROW_INITIATIVE];
                v = (initiative < 2 ? initiative : 0) == player;
            }
        } else {
            int g = j - OFF_LOOSE - 2 * PLAYER_SCALARS;
            if (g == 0)
                v = (row[ROW_PLIES] | (row[ROW_PLIES + 1] << 8)) / (float)MAX_MAIN_PLAYS;
            else if (g == 1) v = row[ROW_INIT_MOVED];
            else if (g == 2) v = row[ROW_TO_ACT] == 0;
            else {
                int k = g - 3, d = k / PENDING_KINDS;
                v = d < CONT_CAP && row[ROW_STACK_KIND + d] == k % PENDING_KINDS;
            }
        }
        dst[j] = v;
    }
}

__global__ void k_gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu1(x[i]);
}

// LayerNorm in place, with the GELU folded in when `act` -- `Norm::apply`
// against `Norm::plain`.
//
// One row per **warp**, not per block. The rows here are 96 to 256 wide, so a
// block-wide reduction spends most of its time in `__syncthreads` for a sum a
// warp can shuffle in five steps -- and at 96 wide a 128-thread block leaves a
// quarter of its lanes idle throughout. Measured at a third of all device
// time, it was the largest kernel in the profile.
//
// The join's four norms are not here: they are fused into `k_leaf`, which
// keeps its residual stream in registers and never writes a row out to be
// normalised.
__global__ void k_norm_ip(float* x, const float* gamma, const float* beta,
                          int rows, int width, int act) {
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    float* row = x + (size_t)r * width;
    // At most eight values a lane, so the row is read once and kept.
    float v[8];
    int n = 0;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += 32) {
        v[n++] = row[j];
        sum += row[j];
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
        row[j] = act ? gelu1(y) : y;
    }
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




// The trunk's eight residual blocks, with the board resident in shared memory
// and the three matrix multiplies on the tensor cores.
//
// This is the kernel that decides whether the throughput target is reachable.
// Three boards share one 112-row matrix, so the tensor cores pay for one
// padding row instead of eleven per board. The float residual stream stays in
// shared memory; normalized rows and neighbour projections stay half-sized.
// The eight residual blocks run in one fixed order, and a board's answer is
// independent of which other boards share its block.
//
// A warp is the unit of a multiply. `mma.sync.m16n8k16` takes a 16x8 slab of
// the combined board and a 16x8 slab of the weight, accumulating in f32. Twelve
// warps cover the ninety-six output channels. The same warps then read rows in
// the LayerNorm shape, so reductions remain warp shuffles.
//
// The neighbour half of the mix is folded rather than gathered: the mix is
// linear, so summing the neighbours' projections is the same as projecting
// their sum. Both halves share each input fragment.
//
// `off` is the weight plan, `TRUNK_OFF` entries a block and two after them:
//   0 mix.w  1 mix.b  2 pool.w  3 pool.b  4 out.w  5 out.b
//   6 ln0.g  7 ln0.b  8 ln1.g  9 ln1.b     then trunk.ln.g, trunk.ln.b
#define TRUNK_OFF 12
// `TRUNK_C`, `TRUNK_BOARDS` and `TRUNK_ROWS` arrive from Rust. Three boards
// share one M dimension, and the rows beyond the real batch are zero, so only
// one of the 112 tensor-core rows is padding.
// Row tiles a warp accumulates, and steps of a sixteen-wide `k` tile.
#define TRUNK_MT (TRUNK_ROWS / 16)
#define TRUNK_KS (TRUNK_C / 16)
// Warps in a block: one eight-channel output tile each. Row-wise that is four
// hexes a warp and three channels a lane, and a hex row is exactly one warp
// wide, so a LayerNorm is a shuffle with no barrier in it.
#define TRUNK_SPAN (TRUNK_C / 8)
#define TRUNK_MAXH ((N_HEXES + TRUNK_SPAN - 1) / TRUNK_SPAN)
#define TRUNK_Q (TRUNK_C / 32)
// The float residual stride and the half operand stride. Both are padded so a
// fragment starts on a four-byte boundary and neighboring rows do not alias
// the same shared-memory banks.
#define TRUNK_LDS (TRUNK_C + 4)
#define TRUNK_HALF_LDS (TRUNK_C + 8)

// A `.tf32` operand is an f32 register whose low thirteen mantissa bits are
// zero. Rounding here, rather than letting the tensor core truncate, is what
// makes the error unbiased: eleven significand bits into every product, single
// precision out of every accumulate. The weights are rounded once on the host,
// so this is only ever applied to an activation as it is stored.
__device__ __forceinline__ float tf32(float v) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return __uint_as_float(r);
}

/// `d += a * b` over one 16x8x8 tile: `a` row-major, `b` column-major, `d` an
/// f32 accumulator. The register-to-element map is the one the PTX manual
/// gives for `mma.m16n8k8` with `.f32` operands; `frag_a`, `frag_b`,
/// `frag_row` and `frag_col` are the four corners of it.
__device__ __forceinline__ void mma_tile(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/// The board's 16x8 fragment at row tile `m` and depth `k`. A lane holds the
/// rows `lane / 4` and `lane / 4 + 8` at the columns `lane % 4` and
/// `lane % 4 + 4`, and the register order is row first: the low bit of the
/// register index is the row and the next one is the column.
__device__ __forceinline__ void frag_a(const float* a, int m, int k, int lane,
                                       int lds, unsigned (&f)[4]) {
    const float* p = a + (size_t)(16 * m + (lane >> 2)) * lds + k + (lane & 3);
    f[0] = __float_as_uint(p[0]);
    f[1] = __float_as_uint(p[8 * lds]);
    f[2] = __float_as_uint(p[4]);
    f[3] = __float_as_uint(p[8 * lds + 4]);
}

/// The weight's 8x8 fragment at depth `k` for the warp's own output tile. The
/// matrix arrives in fragment order, so a lane's two values are one eight-byte
/// load and the warp reads two hundred and fifty-six contiguous bytes.
__device__ __forceinline__ void frag_b(const float* w, int k, int slot, int lane,
                                       int ntiles, unsigned (&f)[2]) {
    float2 v = *(const float2*)(w + (((size_t)k * ntiles + slot) * 32 + lane) * 2);
    f[0] = __float_as_uint(v.x);
    f[1] = __float_as_uint(v.y);
}

/// `d += a * b` over one 16x8x16 half tile. The operands are four and two
/// packed half2 registers; accumulation stays in f32.
__device__ __forceinline__ void mma_half_tile(float (&d)[4], const unsigned (&a)[4],
                                              const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/// The `m16n8k16` A fragment: two rows eight apart, and two pairs of columns
/// eight apart. The four registers contain the eight half values in PTX order.
__device__ __forceinline__ void frag_a_half(const __half* a, int m, int k, int lane,
                                            int lds, unsigned (&f)[4]) {
    const __half* p = a + (size_t)(16 * m + (lane >> 2)) * lds + k + 2 * (lane & 3);
    f[0] = *(const unsigned*)p;
    f[1] = *(const unsigned*)(p + 8 * lds);
    f[2] = *(const unsigned*)(p + 8);
    f[3] = *(const unsigned*)(p + 8 * lds + 8);
}

/// The `m16n8k16` B fragment is packed by k tile, output tile and lane. A
/// lane's four values are two consecutive rows, then the pair eight rows on.
__device__ __forceinline__ void frag_b_half(const __half* w, int k, int slot, int lane,
                                            int ntiles, unsigned (&f)[2]) {
    const unsigned* p = (const unsigned*)w
        + (((size_t)k * ntiles + slot) * 32 + lane) * 2;
    f[0] = p[0];
    f[1] = p[1];
}

/// Pack the trunk's f32 matrices into the half fragments its tensor cores read.
__global__ void k_trunk_half(const float* w, const int* off, unsigned short* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int mix = 2 * TRUNK_C * TRUNK_C;
    const int matrix = i % (3 * TRUNK_C * TRUNK_C) < mix ? 0 : 1;
    const int block = i / (3 * TRUNK_C * TRUNK_C);
    const int local = i % (3 * TRUNK_C * TRUNK_C) - matrix * mix;
    const int base = off[block * TRUNK_OFF + (matrix == 0 ? 10 : 11)];
    const int tiles = TRUNK_C / 8;
    const int tile = local / (tiles * 128);
    const int in_tile = local % (tiles * 128);
    const int nt = in_tile / 128;
    const int lane = (in_tile % 128) / 4;
    const int element = in_tile % 4;
    const int row = 16 * tile + 2 * (lane & 3) + (element >= 2 ? 8 : 0) + (element & 1);
    const int col = 8 * nt + (lane >> 2);
    out[i] = __half_as_ushort(__float2half_rn(w[base + row * TRUNK_C + col]));
}

/// Which element of the board an accumulator register holds: register `i` of
/// row tile `m` is the row below, and the channel beside it.
__device__ __forceinline__ int frag_row(int m, int i, int lane) {
    return 16 * m + (lane >> 2) + ((i & 2) ? 8 : 0);
}
__device__ __forceinline__ int frag_col(int i, int slot, int lane) {
    return 8 * slot + 2 * (lane & 3) + (i & 1);
}

// Mean and inverse standard deviation of one hex's row, held `TRUNK_Q` values
// to a lane across one warp. Two passes, as `k_norm_ip` does, because the
// one-pass form loses the difference between two nearly equal sums.
__device__ __forceinline__ void row_stats(const float* v, int c, float* mean,
                                          float* inv) {
    float s = 0.0f;
    for (int q = 0; q < TRUNK_Q; ++q) s += v[q];
    s = warp_sum(s);
    float m = s / c, t = 0.0f;
    for (int q = 0; q < TRUNK_Q; ++q) {
        float d = v[q] - m;
        t += d * d;
    }
    t = warp_sum(t);
    *mean = m;
    *inv = rsqrtf(t / c + 1e-5f);
}

// Say the occupancy that this width and device can fit. Left to itself the
// compiler spends a one-block register budget on nothing the accumulators need.
__global__ __launch_bounds__(32 * TRUNK_SPAN, TRUNK_MIN_BLOCKS)
void k_trunk(const float* x0, const int* nb, const float* __restrict__ w,
             const __half* __restrict__ wh, const float* __restrict__ bias,
             const float* __restrict__ ln,
             const int* __restrict__ off, const float* xpub, float* out,
             int rows, int nhex, int c, int blocks, int stride, int off_loose,
             int loose) {
    int first = blockIdx.x * TRUNK_BOARDS;
    if (first >= rows) return;
    int count = min(TRUNK_BOARDS, rows - first);
    int valid_rows = count * nhex;
    extern __shared__ __align__(16) unsigned char raw[];
    float* x = reinterpret_cast<float*>(raw);
    __half* a = reinterpret_cast<__half*>(x + TRUNK_BOARDS * N_HEXES * TRUNK_LDS);
    __half* u = a + TRUNK_ROWS * TRUNK_HALF_LDS;
    float* pooled = reinterpret_cast<float*>(u + TRUNK_BOARDS * N_HEXES * TRUNK_HALF_LDS);
    float* gb = pooled + TRUNK_BOARDS * 2 * c;

    const int lane = threadIdx.x, slot = threadIdx.y;
    float cur[TRUNK_Q];

#pragma unroll
    for (int board = 0; board < TRUNK_BOARDS; ++board)
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t) {
            int hex = slot + t * TRUNK_SPAN;
            if (board < count && hex < nhex)
                for (int q = 0; q < TRUNK_Q; ++q)
                    x[(board * nhex + hex) * TRUNK_LDS + lane + 32 * q] =
                        x0[((size_t)(first + board) * nhex + hex) * TRUNK_C + lane + 32 * q];
        }
    __syncthreads();

    for (int blk = 0; blk < blocks; ++blk) {
        const int* o = off + blk * TRUNK_OFF;
        // Normalize every real board row into the half operand. The combined
        // matrix has one final zero row, which removes the per-board padding.
#pragma unroll
        for (int board = 0; board < TRUNK_BOARDS; ++board)
#pragma unroll
            for (int t = 0; t < TRUNK_MAXH; ++t) {
                int hex = slot + t * TRUNK_SPAN;
                bool live = board < count && hex < nhex;
                int r = board * nhex + hex;
                for (int q = 0; q < TRUNK_Q; ++q)
                    cur[q] = live ? x[r * TRUNK_LDS + lane + 32 * q] : 0.0f;
                float mean, inv;
                row_stats(cur, c, &mean, &inv);
                if (live)
                    for (int q = 0; q < TRUNK_Q; ++q) {
                        int j = lane + 32 * q;
                        a[(size_t)r * TRUNK_HALF_LDS + j] = __float2half_rn(
                            gelu1((cur[q] - mean) * inv * ln[o[6] + j] + ln[o[7] + j]));
                    }
            }
        if (slot == 0)
            for (int r = valid_rows; r < TRUNK_ROWS; ++r)
                for (int j = lane; j < c; j += 32)
                    a[(size_t)r * TRUNK_HALF_LDS + j] = __float2half(0.0f);
        __syncthreads();

        // Pool each board's normalized rows and form its global bias.
        if (slot == 0) {
            for (int board = 0; board < count; ++board)
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    float sum = 0.0f, mx = neg_inf();
                    for (int hex = 0; hex < nhex; ++hex) {
                        float v = __half2float(a[(size_t)(board * nhex + hex) * TRUNK_HALF_LDS + j]);
                        sum += v;
                        mx = fmaxf(mx, v);
                    }
                    pooled[(size_t)board * 2 * c + j] = sum / nhex;
                    pooled[(size_t)board * 2 * c + c + j] = mx;
                }
            for (int board = 0; board < count; ++board)
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    float sv = bias[o[3] + j];
                    for (int k = 0; k < 2 * c; ++k)
                        sv += pooled[(size_t)board * 2 * c + k] * w[o[2] + (size_t)k * c + j];
                    gb[(size_t)board * c + j] = sv;
                }
        }
        __syncthreads();

        // The three boards are one M dimension. Both halves of the mix share
        // each A fragment, and their f32 accumulators are written back as
        // half rows for the neighbour gather and the next norm.
        float an[TRUNK_MT][4], as[TRUNK_MT][4];
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) an[m][i] = as[m][i] = 0.0f;
        for (int k = 0; k < TRUNK_KS; ++k) {
            unsigned bs[2], bn[2];
            frag_b_half(wh + o[10], k, slot, lane, TRUNK_SPAN, bs);
            frag_b_half(wh + o[10] + (size_t)TRUNK_C * TRUNK_C, k, slot, lane, TRUNK_SPAN, bn);
#pragma unroll
            for (int m = 0; m < TRUNK_MT; ++m) {
                unsigned af[4];
                frag_a_half(a, m, 16 * k, lane, TRUNK_HALF_LDS, af);
                mma_half_tile(as[m], af, bs);
                mma_half_tile(an[m], af, bn);
            }
        }
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = frag_row(m, i, lane), j = frag_col(i, slot, lane);
                if (r < valid_rows) {
                    int board = r / nhex;
                    a[(size_t)r * TRUNK_HALF_LDS + j] =
                        __float2half_rn(as[m][i] + bias[o[1] + j] + gb[(size_t)board * c + j]);
                    u[(size_t)r * TRUNK_HALF_LDS + j] = __float2half_rn(an[m][i]);
                }
            }
        __syncthreads();

        // Gather the six neighbours from the half projection, then normalize
        // the mixed row for the next half-precision matrix multiply.
#pragma unroll
        for (int board = 0; board < TRUNK_BOARDS; ++board)
#pragma unroll
            for (int t = 0; t < TRUNK_MAXH; ++t) {
                int hex = slot + t * TRUNK_SPAN;
                bool live = board < count && hex < nhex;
                int r = board * nhex + hex;
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    float v = live ? __half2float(a[(size_t)r * TRUNK_HALF_LDS + j]) : 0.0f;
                    if (live)
                        for (int k = 0; k < 6; ++k) {
                            int n = nb[hex * 6 + k];
                            if (n >= 0)
                                v += __half2float(u[(size_t)(board * nhex + n) * TRUNK_HALF_LDS + j]);
                        }
                    cur[q] = v;
                }
                float mean, inv;
                row_stats(cur, c, &mean, &inv);
                if (live)
                    for (int q = 0; q < TRUNK_Q; ++q) {
                        int j = lane + 32 * q;
                        a[(size_t)r * TRUNK_HALF_LDS + j] = __float2half_rn(
                            gelu1((cur[q] - mean) * inv * ln[o[8] + j] + ln[o[9] + j]));
                    }
            }
        __syncthreads();

        // The output projection stays in f32 until it is added to the residual.
        float ao[TRUNK_MT][4];
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) ao[m][i] = 0.0f;
        for (int k = 0; k < TRUNK_KS; ++k) {
            unsigned bo[2];
            frag_b_half(wh + o[11], k, slot, lane, TRUNK_SPAN, bo);
#pragma unroll
            for (int m = 0; m < TRUNK_MT; ++m) {
                unsigned af[4];
                frag_a_half(a, m, 16 * k, lane, TRUNK_HALF_LDS, af);
                mma_half_tile(ao[m], af, bo);
            }
        }
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = frag_row(m, i, lane), j = frag_col(i, slot, lane);
                if (r < valid_rows) x[r * TRUNK_LDS + j] += ao[m][i] + bias[o[5] + j];
            }
        __syncthreads();
    }

    const int* tn = off + blocks * TRUNK_OFF;
#pragma unroll
    for (int board = 0; board < TRUNK_BOARDS; ++board)
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t) {
            int hex = slot + t * TRUNK_SPAN;
            bool live = board < count && hex < nhex;
            int r = board * nhex + hex;
            for (int q = 0; q < TRUNK_Q; ++q)
                cur[q] = live ? x[r * TRUNK_LDS + lane + 32 * q] : 0.0f;
            float mean, inv;
            row_stats(cur, c, &mean, &inv);
            if (live)
                for (int q = 0; q < TRUNK_Q; ++q) {
                    int j = lane + 32 * q;
                    x[r * TRUNK_LDS + j] =
                        gelu1((cur[q] - mean) * inv * ln[tn[0] + j] + ln[tn[1] + j]);
                }
        }
    __syncthreads();
    int width = 2 * c + loose;
    if (slot == 0)
        for (int board = 0; board < count; ++board)
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                float sum = 0.0f, mx = neg_inf();
                for (int hex = 0; hex < nhex; ++hex) {
                    float v = x[(board * nhex + hex) * TRUNK_LDS + j];
                    sum += v;
                    mx = fmaxf(mx, v);
                }
                out[(size_t)(first + board) * width + j] = sum / nhex;
                out[(size_t)(first + board) * width + c + j] = mx;
            }
    for (int board = 0; board < count; ++board)
        for (int k = lane + 32 * slot; k < loose; k += 32 * TRUNK_SPAN)
            out[(size_t)(first + board) * width + 2 * c + k] =
                xpub[(size_t)(first + board) * stride + off_loose + k];
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
// other either. The host lays that level out as a flat list of work items, one
// per (solve, node), and launches exactly as many blocks as it holds:
// `blockIdx.x` names the item and the item names the solve. A grid sized
// instead by the widest solve in the round paid for that width at every solve,
// and three quarters of its blocks returned at the first load.
//
// Everything a solve's iteration reads or writes is named by one descriptor, so
// a stage takes one array of them rather than thirty arrays of pointers. The
// field order below is `Card::describe` in `cuda/mod.rs`; every field is eight
// bytes wide so the layout is positional and nothing needs packing rules.

struct Tree {
    const unsigned int* kind;
    const unsigned int* player;
    // Whether the subtree under a node holds no expandable leaf. An expansion
    // trajectory that walked into one could only end on a leaf the host may
    // not grow, and the simulation would be spent for nothing.
    const unsigned int* exhausted;
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
    const unsigned int* cell_val;
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
    float* avg;
    const float* rootb;
    // The network state that outlives an iteration: board vectors, the join
    // cache, the config readout and pooling rows, and the belief index.
    const float* p;
    const float* jp;
    // Row -> the board it reads. Coin plays commute, so a tree spanning one
    // round holds the same public state at several places and the trunk runs
    // once for all of them.
    const unsigned int* board_of;
    const float* f;
    const float* g;
    // The policy readout's config row, which `k_prior` dots against an action.
    const float* fp;
    const unsigned int* cidx;
    const unsigned int* coff;
    // Batch row -> node, for the leaves the network answers for.
    const unsigned int* leaf_node;
    const unsigned int* term;
    unsigned long long* seed;
    unsigned long long nterm;
    /// One value arena per traverser, so both can be backpropagated at once.
    unsigned long long nvals;
    // A round holds solves at different points of their own sixty-four
    // iterations, and a CFR iterate is weighted by how many came before it. So
    // the decay factors are a function of *this* solve's step count and are
    // computed here rather than passed in: one number for the whole batch
    // would be some other solve's weights.
    unsigned long long step;
    /// Iterations this call asks of this solve, and distinct leaves to take
    /// after each of them. Both differ across a round once a tree is full.
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
#define NO_TRANS 0xffffffffu
#define KIND_LEAF 2u
#define KIND_CHANCE 1u

// A work item: the solve in the high bits, the node's place inside that solve's
// level in the low ones. `Card::lay` packs them, one per (solve, node),
// bucketed by level and in solve order -- so the first `k` solves own a prefix
// of a level's bucket and a round that fewer solves want is a shorter grid.
#define WORK_BITS 20
#define WORK_SLOT ((1u << WORK_BITS) - 1)

__device__ __forceinline__ unsigned int work_node(const Tree& t, int level,
                                                  unsigned int item) {
    return t.level_node[t.level_start[level] + (item & WORK_SLOT)];
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

// Reach probabilities for one level, a block to each (node, player). `avg`
// picks the reference strategy over the regret-matching iterate, which is what
// the value pass that produces a solve's targets propagates under.
__global__ void k_reach_sweep(const Tree* trees, const unsigned int* work, int at,
                              int level, int avg, int also_sum, int iter) {
    unsigned int item = work[at + blockIdx.x];
    const Tree& t = trees[item >> WORK_BITS];
    if ((unsigned long long)iter >= t.todo) return;
    const float* strat = avg ? t.avg : t.cur;
    unsigned int node = work_node(t, level, item);
    unsigned int par = t.parent[node];
    if (par == NO_ROW) return;
    unsigned int me = t.player[par];
    // A block to a player. The two write disjoint halves of the node's reach
    // region and both read a parent the level above already finished, so the
    // serial loop over them was parallelism left on the floor.
    unsigned int p = blockIdx.y;
    {
        unsigned int n = t.nc[2 * node + p];
        unsigned int dst = rbase(t, node, p), src = rbase(t, par, p);
        for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
            if (p != me) {
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
                    v += t.reach[src + t.rev_src[k]] * strat[t.rev_cell[k]];
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
    // The accumulation below reads the acting player's reach, which is the half
    // this block has just written when it is that player's block -- so it is
    // that block's alone, and still exactly one block a node.
    if (!also_sum || t.kind[node] != 0 || p != t.player[node]) return;
    // It reads that reach by *cell* where the sweep wrote it by *config*, so a
    // lane reads an address another warp of this block owns. Every early return
    // above is block-uniform, so every thread reaches this.
    __syncthreads();
    // The reach-weighted iterate, added to the running strategy sum. The reach
    // it needs is the one the loop above has just made current, and the thread
    // that owns a config there owns it here, so this costs a level of launches
    // less than a pass of its own would.
    unsigned int an = t.nc[2 * node + p], so = t.soff[node], lb = t.legal_base[node];
    unsigned int ra = rbase(t, node, p);
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
__global__ void k_backprop_sweep(const Tree* trees, const unsigned int* work, int at,
                                 int level, int avg, int iter,
                                 float alpha, float beta, float gamma, float predict) {
    const float EPS = 1e-6f;
    unsigned int item = work[at + blockIdx.x];
    const Tree& t = trees[item >> WORK_BITS];
    if ((unsigned long long)iter >= t.todo) return;
    // This solve's own iterate index, not the round's.
    float m = (float)(t.step + (unsigned long long)iter) + 1.0f;
    float da = cfr_factor(m, alpha), db = cfr_factor(m, beta);
    float dg = powf(m / (m + 1.0f), gamma);
    // The two traversers write disjoint cells and read disjoint value arenas,
    // so they are one launch rather than two.
    int traverser = blockIdx.y;
    float* vals = t.vals + traverser * t.nvals;
    unsigned int node = work_node(t, level, item);
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
    // thirty-two separate transactions per step over `cur`, `regret`, `qval`
    // and `sum` -- the largest arrays a solve holds. The reductions a config
    // needs are warp shuffles rather than serial sums.
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (avg) {
        for (unsigned int c = warp; c < n; c += warps) {
            unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
            float base = 0.0f;
            for (unsigned int cell = a + lane; cell < b; cell += 32) {
                unsigned int vc = t.cell_val[so + cell];
                if (vc != NO_ROW) base += vals[vc] * t.avg[so + cell];
            }
            base = warp_sum(base);
            if (lane == 0) vals[vi + c] = base;
        }
        return;
    }
    // The expansion phase reads `qval` as PUCT's Q. A sweep that computes the
    // action values and drops them leaves selection blind, and the tree it
    // grows is a different tree -- which is a wrong answer no shape check
    // would catch. Every cell is written here, illegal ones as zero, so the
    // pass that used to clear them first is gone.
    unsigned int ncells = t.legal_off[lb + n];
    for (unsigned int c = warp; c < n; c += warps) {
        unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
        float base = 0.0f;
        for (unsigned int cell = a + lane; cell < b; cell += 32) {
            unsigned int vc = t.cell_val[so + cell];
            float av = vc == NO_ROW ? 0.0f : vals[vc];
            t.qval[so + cell] = av;
            base += av * t.cur[so + cell];
        }
        base = warp_sum(base);
        if (lane == 0) vals[vi + c] = base;
        float total = 0.0f;
        for (unsigned int cell = a + lane; cell < b; cell += 32) {
            float delta = t.qval[so + cell] - base;
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
__global__ void k_avg_block(const Tree* trees, const unsigned int* work, int at,
                            int level, int iter) {
    unsigned int item = work[at + blockIdx.x];
    const Tree& t = trees[item >> WORK_BITS];
    if ((unsigned long long)iter >= t.todo) return;
    unsigned int node = work_node(t, level, item);
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

// ------------------------------------------------------------ the leaf network
//
// Belief pooling, `net::Net::join`, the head norm and the config readout in one
// launch. A block keeps all intermediate rows in registers or shared memory.
//
// A warp is the unit of a multiply, as in `k_trunk`. `mma.sync.m16n8k8` takes
// a 16x8 slab of the activations and an 8x8 slab of the weight; sixteen warps
// cover the 128-wide residual, two sixteen-row tiles cover the block's 32
// rows. `J_IN` is 136: JOIN_IN padded from 129 so the first `k` is whole
// fragments. Those seven extra columns are zero.
//
// The residual stream lives in the accumulators. Only the normalised
// activations go through shared memory, bank-padded, rounded to tf32 as they
// are stored. The weights arrive fragwise from the host, rounded once.
//
// Rows are interleaved by traverser so both pooled beliefs of a leaf stay in
// the same block. The shared buffer first holds the join operand and pooled
// beliefs, then becomes the 256-wide head. The reach masses follow it.
#define J_SPAN (J_W / 8)
#define J_MT (J_ROWS / 16)
#define J_LDS (J_IN + 4)
#define J_KS_IN (J_IN / 8)
#define J_KS (J_W / 8)
#define J_Q (J_W / 32)
#define J_OUT_TILES (J_D / 8)
#define J_PACK_B ((size_t)J_KS_IN * J_SPAN * 64)
#define J_PACK_W ((size_t)J_KS * J_SPAN * 64)

__device__ __forceinline__ void join_mma(float (&d)[J_MT][4], const float* act,
                                         const float* w, int ks, int ntiles,
                                         int nt0) {
    int lane = threadIdx.x, slot = threadIdx.y;
    for (int k = 0; k < ks; ++k) {
        unsigned b[2];
        frag_b(w, k, nt0 + slot, lane, ntiles, b);
#pragma unroll
        for (int m = 0; m < J_MT; ++m) {
            unsigned a[4];
            frag_a(act, m, 8 * k, lane, J_LDS, a);
            mma_tile(d[m], a, b);
        }
    }
}

// `Norm::apply` over the block's rows: the residual stream, plus the bias it is
// owed, normalised into `act` as the tf32 operand the next multiply reads.
// The stream itself stays in the accumulators.
__device__ __forceinline__ void join_norm(float (&z)[J_MT][4], float* act,
                                          const float* gamma, const float* beta,
                                          const float* add) {
    // The multiply that filled `z` is still reading `act` in other warps.
    __syncthreads();
    int lane = threadIdx.x, slot = threadIdx.y;
#pragma unroll
    for (int m = 0; m < J_MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = frag_row(m, i, lane), j = frag_col(i, slot, lane);
            act[r * J_LDS + j] = z[m][i] + add[j];
        }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < J_MT; ++t) {
        int r = slot + t * J_SPAN;
        float cur[J_Q];
#pragma unroll
        for (int q = 0; q < J_Q; ++q) cur[q] = act[r * J_LDS + lane + 32 * q];
        float s = 0.0f;
#pragma unroll
        for (int q = 0; q < J_Q; ++q) s += cur[q];
        s = warp_sum(s);
        float mean = s / (float)J_W, var = 0.0f;
#pragma unroll
        for (int q = 0; q < J_Q; ++q) {
            float d = cur[q] - mean;
            var += d * d;
        }
        var = warp_sum(var);
        float inv = rsqrtf(var / (float)J_W + 1e-5f);
#pragma unroll
        for (int q = 0; q < J_Q; ++q) {
            int j = lane + 32 * q;
            act[r * J_LDS + j] =
                tf32(gelu1((cur[q] - mean) * inv * gamma[j] + beta[j]));
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(32 * J_SPAN, 2)
void k_leaf(const Tree* trees, const int* part_of_row, const int* local_row,
            const int* base_of_part, const unsigned int* coff,
            const float* wj, const float* lnj, const float* owed,
            const float* cf_bias, const float* gamma, const float* beta,
            int rows, int q0) {
    __shared__ __align__(16) float shared[J_ROWS * (J_D + 1)];
    float* act = shared;
    float* pooled = shared + J_ROWS * J_LDS;
    float* mass = shared + J_ROWS * J_D;

    int lane = threadIdx.x, slot = threadIdx.y;
    int tid = lane + 32 * slot, nt = 32 * J_SPAN;
    int row0 = blockIdx.x * J_ROWS;
    float z[J_MT][4];

    // One warp pools one query at a time. Consecutive rows are the two
    // traversers of one leaf, so the join reuses both rows in this block.
    for (int qr = slot; qr < J_ROWS; qr += J_SPAN) {
        int query = row0 + qr;
        if (query >= rows) continue;
        int r = q0 + (query >> 1), p = query & 1;
        int part = part_of_row[r];
        const Tree& t = trees[part];
        unsigned int node = t.leaf_node[local_row[r]];
        unsigned int n = t.nc[2 * node + p], ra = rbase(t, node, p);
        unsigned int base = base_of_part[part];
        unsigned int lo = coff[2 * r + p], hi = coff[2 * r + p + 1];
        float total = 0.0f;
        for (unsigned int c = lane; c < n; c += 32) total += t.reach[ra + c];
        total = warp_sum(total);
        if (lane == 0) mass[qr] = total;
        float inv = total > 0.0f ? 1.0f / total : 1.0f / (float)max(n, 1u);
        for (int j = lane; j < J_POOL; j += 32) {
            float acc = 0.0f;
            for (unsigned int k = lo; k < hi; ++k) {
                float belief = total > 0.0f ? t.reach[ra + k - lo] * inv : inv;
                acc += belief * t.g[(size_t)t.cidx[k - base] * J_POOL + j];
            }
            pooled[(size_t)qr * J_POOL + j] = acc;
        }
    }
    __syncthreads();

    // jp into shared, then into the accumulators. The same buffer then takes
    // the belief input, so the seed is read once.
    for (int e = tid; e < J_ROWS * J_LDS; e += nt) {
        int i = e / J_LDS, c = e % J_LDS;
        int row = row0 + i;
        float v = 0.0f;
        if (row < rows && c < J_W) {
            int rr = q0 + (row >> 1);
            const Tree& t = trees[part_of_row[rr]];
            v = t.jp[(size_t)t.board_of[local_row[rr]] * J_W + c];
        }
        act[e] = v;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < J_MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            z[m][i] = act[frag_row(m, i, lane) * J_LDS + frag_col(i, slot, lane)];
    __syncthreads();

    for (int e = tid; e < J_ROWS * J_LDS; e += nt) {
        int i = e / J_LDS, c = e % J_LDS;
        int row = row0 + i;
        float v = 0.0f;
        if (row < rows && c < 2 * J_POOL + 1) {
            int qr = row - row0, p = row & 1;
            const float* mine = pooled + (size_t)qr * J_POOL;
            const float* theirs = pooled + (size_t)(qr ^ 1) * J_POOL;
            if (c < J_POOL) v = mine[c];
            else if (c < 2 * J_POOL) v = theirs[c - J_POOL];
            else v = p == 0 ? -1.0f : 1.0f;
        }
        act[e] = tf32(v);
    }
    __syncthreads();

    join_mma(z, act, wj, J_KS_IN, J_SPAN, 0);
    const float* w = wj + J_PACK_B;
    for (int blk = 0; blk < J_BLOCKS; ++blk) {
        join_norm(z, act, lnj + 2 * blk * J_W, lnj + (2 * blk + 1) * J_W,
                  owed + blk * J_W);
        join_mma(z, act, w, J_KS, J_SPAN, 0);
        w += J_PACK_W;
    }
    join_norm(z, act, lnj + 2 * J_BLOCKS * J_W, lnj + (2 * J_BLOCKS + 1) * J_W,
              owed + J_BLOCKS * J_W);

    // Keep the first half of join_out in registers while the second half still
    // reads `act`. Then the shared join operand can become the whole head.
    float head0[J_MT][4];
    for (int pass = 0; pass < J_D / J_W; ++pass) {
#pragma unroll
        for (int m = 0; m < J_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) z[m][i] = 0.0f;
        join_mma(z, act, w, J_KS, J_OUT_TILES, pass * J_SPAN);
        if (pass == 0) {
#pragma unroll
            for (int m = 0; m < J_MT; ++m)
#pragma unroll
                for (int i = 0; i < 4; ++i) head0[m][i] = z[m][i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < J_MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = frag_row(m, i, lane), j = frag_col(i, slot, lane);
            int row = row0 + r;
            if (row < rows) {
                int rr = q0 + (row >> 1);
                const Tree& t = trees[part_of_row[rr]];
                const float* seed = t.p + (size_t)t.board_of[local_row[rr]] * J_D;
                shared[(size_t)r * J_D + j] = head0[m][i] + seed[j] + owed[(J_BLOCKS + 1) * J_W + j];
                shared[(size_t)r * J_D + J_W + j] = z[m][i] + seed[J_W + j]
                    + owed[(J_BLOCKS + 1) * J_W + J_W + j];
            }
        }
    __syncthreads();

    // One warp normalises a head row and then reads out its configs.
    for (int lr = slot; lr < J_ROWS; lr += J_SPAN) {
        int row = row0 + lr;
        if (row >= rows) continue;
        float h[J_D / 32];
        float sum = 0.0f;
#pragma unroll
        for (int q = 0; q < J_D / 32; ++q) {
            h[q] = shared[(size_t)lr * J_D + lane + 32 * q];
            sum += h[q];
        }
        sum = warp_sum(sum);
        float mean = sum / (float)J_D, var = 0.0f;
#pragma unroll
        for (int q = 0; q < J_D / 32; ++q) {
            float d = h[q] - mean;
            var += d * d;
        }
        float inv = rsqrtf(warp_sum(var) / (float)J_D + 1e-5f);
#pragma unroll
        for (int q = 0; q < J_D / 32; ++q) {
            int j = lane + 32 * q;
            h[q] = (h[q] - mean) * inv * gamma[j] + beta[j];
        }

        int traverser = row & 1, r = q0 + (row >> 1);
        const Tree& t = trees[part_of_row[r]];
        unsigned int node = t.leaf_node[local_row[r]];
        unsigned int lo = coff[2 * r + traverser], hi = coff[2 * r + traverser + 1];
        unsigned int cs = t.coff[2 * local_row[r] + traverser];
        float scale = mass[lr ^ 1], bias = *cf_bias;
        float* vals = t.vals + traverser * t.nvals;
        unsigned int vo = t.voff[node];
        for (unsigned int k = lo; k < hi; ++k) {
            const float* fr = t.f + (size_t)t.cidx[cs + k - lo] * J_D;
            float acc = 0.0f;
#pragma unroll
            for (int q = 0; q < J_D / 32; ++q) acc += fr[lane + 32 * q] * h[q];
            for (int s = 16; s > 0; s >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, s);
            if (lane == 0) vals[vo + k - lo] = (acc + bias) * scale;
        }
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

// The weight a row draws against, summed the way a warp of thirty-two does:
// the lanes stride through the terms and a butterfly folds them. This is
// `search.rs::warp32_sum`, and it has to be -- f32 addition is not
// associative, so a total summed straight through would put a draw's cell
// boundaries a few ulps elsewhere and build a different tree.
__device__ __forceinline__ float pick_sum(const float* w, int n) {
    float total = 0.0f;
    for (int i = threadIdx.x; i < n; i += 32) total += fmaxf(w[i], 0.0f);
    return warp_sum(total);
}

// Draw an index from non-negative weights whose total is already in hand. A
// row whose weights have all underflowed is drawn uniformly rather than
// dropped.
//
// A warp does this together: the scan is the same in every lane, so it
// broadcasts one address at a time rather than gathering. Every lane draws
// from the same stream and takes the same turn, which is what keeps the walk
// below coherent. The needle is a double because `search.rs::pick` uses one,
// and the two must part company at no draw at all.
__device__ int pick_from(const float* w, int n, float total, unsigned long long* s) {
    if (!(total > 0.0f)) return n > 0 ? (int)(rng_next(s) % (unsigned long long)n) : 0;
    double needle = rng_unit(s) * (double)total;
    for (int i = 0; i < n; ++i) {
        needle -= (double)fmaxf(w[i], 0.0f);
        if (needle < 0.0) return i;
    }
    return n - 1;
}

// The two together, for a row whose total is wanted once.
__device__ __forceinline__ int pick(const float* w, int n, unsigned long long* s) {
    return pick_from(w, n, pick_sum(w, n), s);
}

// Whether the expansion phase may descend through one legal cell: the acting
// config has a successor there, and the subtree behind it still has somewhere
// to grow. Mirrors `Solver::live_cell`.
__device__ bool live_cell(const Tree& t, unsigned int so, unsigned int cell) {
    return t.legal_trans[so + cell] != NO_TRANS
        && t.exhausted[t.legal_child[so + cell]] == 0u;
}

// `pick` over the live cells of `[a, b)` alone, `NO_ROW` when there are none.
// `w` is the weight row, based at cell `a`.
__device__ unsigned int pick_live(const Tree& t, unsigned int so, unsigned int a,
                                  unsigned int b, const float* w,
                                  unsigned long long* s) {
    float total = 0.0f, n = 0.0f;
    for (unsigned int i = a + threadIdx.x; i < b; i += 32) {
        if (live_cell(t, so, i)) { total += fmaxf(w[i - a], 0.0f); n += 1.0f; }
    }
    total = warp_sum(total);
    int live = (int)warp_sum(n);
    if (live == 0) return NO_ROW;
    if (!(total > 0.0f)) {
        int k = (int)(rng_next(s) % (unsigned long long)live);
        for (unsigned int i = a; i < b; ++i) {
            if (!live_cell(t, so, i)) continue;
            if (k == 0) return i;
            --k;
        }
        return NO_ROW;
    }
    double needle = rng_unit(s) * (double)total;
    unsigned int last = NO_ROW;
    for (unsigned int i = a; i < b; ++i) {
        if (!live_cell(t, so, i)) continue;
        last = i;
        needle -= (double)fmaxf(w[i - a], 0.0f);
        if (needle < 0.0) return i;
    }
    return last;
}

// The cell PUCT would take from one config's legal row.
//
// `Q + c_puct * P * sqrt(sum N) / (1 + N)`, with `Q` divided by the opponent's
// reach mass at the node -- without it a node behind an unlikely opponent line
// looks worthless beside its siblings instead of being compared with them.
//
// Ties go to the lowest cell, which is what a serial scan keeping the first
// strictly greater score does.
//
// Every arithmetic operation here is written as an `_rn` intrinsic, so nvcc
// may not contract the multiply and the add into an FMA. `Solver::puct_choice`
// is plain Rust f32, which never fuses, and growth is discrete: one fused
// multiply-add would round differently, flip an argmax at a close call and
// build a different tree. The same reason the sums above are warp-shaped on
// both sides.
__device__ unsigned int puct_choice(const Tree& t, unsigned int node, unsigned int a,
                                    unsigned int b, int opp, float c_puct) {
    unsigned int so = t.soff[node], ra = rbase(t, node, opp);
    unsigned int nc = t.nc[2 * node + opp];
    float mass = 0.0f;
    for (unsigned int i = threadIdx.x; i < nc; i += 32) mass += t.reach[ra + i];
    mass = warp_sum(mass);
    float scale = mass > 1e-30f ? __fdiv_rn(1.0f, mass) : 0.0f;
    float total = 0.0f;
    for (unsigned int cell = a + threadIdx.x; cell < b; cell += 32)
        total += t.visits[so + cell];
    total = warp_sum(total);
    float explore = __fmul_rn(c_puct, sqrtf(fmaxf(total, 0.0f)));
    unsigned int best = NO_ROW;
    float score = neg_inf();
    for (unsigned int cell = a + threadIdx.x; cell < b; cell += 32) {
        if (!live_cell(t, so, cell)) continue;
        float u = __fdiv_rn(__fmul_rn(explore, t.prior[so + cell]),
                            __fadd_rn(1.0f, t.visits[so + cell]));
        float v = __fadd_rn(__fmul_rn(t.qval[so + cell], scale), u);
        if (v > score) { score = v; best = cell; }
    }
    for (int k = 16; k > 0; k >>= 1) {
        float os = __shfl_xor_sync(0xffffffff, score, k);
        unsigned int oc = __shfl_xor_sync(0xffffffff, best, k);
        if (os > score || (os == score && oc < best)) { score = os; best = oc; }
    }
    return best;
}

// ------------------------------------------------------------ the policy prior
//
// `Solver::refresh_priors` used to run on the host, and the round downloaded a
// board vector per fresh leaf and an `f_p` row per fresh config so that it
// could. Everything it reads is here; what it needs and the card does not hold
// is what an action *is*, which is five words an action in `desc`.

// The action encoder's one-hot input, expanded from `Net::action_feats`.
//
// `desc` is five words an action -- kind, coin slot, three hexes -- each
// already the column its block sets, so "spends nothing" and "names no hex"
// arrive as the column past the last rather than as a sentinel to fold here.
// The five blocks are `nkinds`, `nslot + 1` and three of `nhex + 1`.
__global__ void k_act_feats(const unsigned int* desc, float* feat, int n,
                            int nkinds, int nslot, int nhex, int afeat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * afeat) return;
    int col = i % afeat;
    int at = 0, k = 0, width = nkinds;
    while (col >= at + width) {
        at += width;
        ++k;
        width = k == 1 ? nslot + 1 : nhex + 1;
    }
    feat[i] = col == at + (int)desc[5 * (i / afeat) + k] ? 1.0f : 0.0f;
}

// The join input and its two cached public rows for every primed node. The
// belief is the node's normalised reach at the instant its prior is formed.
__global__ void k_prior_inputs(const Tree* trees, const unsigned int* part,
                               const unsigned int* node_of,
                               const unsigned int* row_of, float* input,
                               float* p_out, float* jp_out, int m, int pool,
                               int d, int jw) {
    int k = blockIdx.x, lane = threadIdx.x;
    if (k >= m) return;
    const Tree& t = trees[part[k]];
    unsigned int node = node_of[k], me = t.player[node];
    for (int role = 0; role < 2; ++role) {
        unsigned int player = role == 0 ? me : 1 - me;
        unsigned int n = t.nc[2 * node + player];
        unsigned int ra = rbase(t, node, player);
        float total = 0.0f;
        for (unsigned int c = lane; c < n; c += 32) total += t.reach[ra + c];
        total = warp_sum(total);
        float inv = total > 0.0f ? 1.0f / total : 1.0f / (float)max(n, 1u);
        unsigned int cs = t.coff[2 * row_of[k] + player];
        for (int j = lane; j < pool; j += 32) {
            float acc = 0.0f;
            for (unsigned int c = 0; c < n; ++c) {
                float belief = total > 0.0f ? t.reach[ra + c] * inv : inv;
                acc += belief * t.g[(size_t)t.cidx[cs + c] * pool + j];
            }
            input[(size_t)k * (2 * pool + 1) + role * pool + j] = acc;
        }
    }
    if (lane == 0) input[(size_t)k * (2 * pool + 1) + 2 * pool] = me == 0 ? -1.0f : 1.0f;
    unsigned int board = t.board_of[row_of[k]];
    for (int j = lane; j < d; j += 32) p_out[(size_t)k * d + j] = t.p[(size_t)board * d + j];
    for (int j = lane; j < jw; j += 32) jp_out[(size_t)k * jw + j] = t.jp[(size_t)board * jw + j];
}

// The board's projection, added to the action's. A batch spans nodes, so which
// board an action reads is an index rather than a property of the call.
__global__ void k_act_add(float* z, const float* proj, const unsigned int* of,
                          int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    z[i] += proj[(size_t)of[i / width] * width + i % width];
}

// `prior(c, .) = softmax(<f_p(c), e(a)> / temp)` over one config's legal row.
//
// A warp to a config. The dot is the warp's, so a row of `f_p` and a row of `e`
// are each read once and coalesced; the softmax that follows is over the cells,
// which are few, so it takes a lane apiece and `prior` itself as its scratch.
__global__ void k_prior(const Tree* trees, const unsigned int* part,
                        const unsigned int* node_of, const unsigned int* row_of,
                        const unsigned int* act_at, const unsigned int* cell_at,
                        const unsigned int* cells, const float* e,
                        const float* inv_t, int m, int d) {
    int k = blockIdx.y;
    if (k >= m) return;
    int c = blockIdx.x * blockDim.y + threadIdx.y;
    const Tree& t = trees[part[k]];
    unsigned int node = node_of[k];
    unsigned int me = t.player[node];
    if ((unsigned)c >= t.nc[2 * node + me]) return;
    unsigned int lb = t.legal_base[node], so = t.soff[node];
    unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
    if (a == b) return;
    int lane = threadIdx.x;
    unsigned int cs = t.coff[2 * row_of[k] + me];
    const float* fp = t.fp + (size_t)t.cidx[cs + t.cell_row[so + a]] * d;
    for (unsigned int cell = a; cell < b; ++cell) {
        const float* ea = e + (size_t)(act_at[k] + cells[cell_at[k] + cell]) * d;
        float acc = 0.0f;
        for (int j = lane; j < d; j += 32) acc += fp[j] * ea[j];
        acc = warp_sum(acc);
        if (lane == 0) t.prior[so + cell] = acc;
    }
    __syncwarp();
    float top = neg_inf();
    for (unsigned int cell = a + lane; cell < b; cell += 32)
        top = fmaxf(top, t.prior[so + cell]);
    for (int s = 16; s > 0; s >>= 1)
        top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, s));
    float mine = 0.0f;
    for (unsigned int cell = a + lane; cell < b; cell += 32) {
        float v = expf((t.prior[so + cell] - top) * inv_t[k]);
        t.prior[so + cell] = v;
        mine += v;
    }
    float total = warp_sum(mine);
    float scale = total > 0.0f ? 1.0f / total : 1.0f / (float)(b - a);
    for (unsigned int cell = a + lane; cell < b; cell += 32) t.prior[so + cell] *= scale;
}

// `out` is the whole round's buffer, `each` the stride between its phases, so
// the leaves every earlier phase of this round took are here to be read back --
// which is the state that makes a leaf "already taken". Thread zero is the only
// one that touches `out`, in this launch and in the ones before it, so there is
// nothing to synchronise.
__global__ void k_expand(const Tree* trees, unsigned int* out, int parts,
                         int sims, float c_puct, int iter, int each, int tries) {
    int part = blockIdx.x;
    if (part >= parts) return;
    const Tree& t = trees[part];
    unsigned int* taken = out + (size_t)iter * each + (size_t)part * sims;
    // The row is `sims` wide however many leaves this phase takes, so the host
    // can slice it: a solve that has had its share of the round's iterations, a
    // solve whose tree has spent its budget and a phase that gave up short all
    // read as nothing.
    if (threadIdx.x == 0)
        for (int sim = 0; sim < sims; ++sim) taken[sim] = NO_ROW;
    __syncwarp();
    if ((unsigned long long)iter >= t.todo || t.nexpand == 0) return;
    unsigned long long s = *t.seed;
    unsigned int n0 = t.nc[0], n1 = t.nc[1];
    // The root's belief does not move through a phase, so the weight its two
    // draws work against is summed once rather than at every draw.
    float b0 = pick_sum(t.rootb, (int)n0), b1 = pick_sum(t.rootb + n0, (int)n1);
    int want = (int)t.nexpand, got = 0;
    for (int draw = 0; draw < want * tries && got < want; ++draw) {
        int c[2];
        c[0] = pick_from(t.rootb, (int)n0, b0, &s);
        c[1] = pick_from(t.rootb + n0, (int)n1, b1, &s);
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
            unsigned int cell;
            // Student of Games selects by half PUCT and half the search's own
            // average: `pi_select = 1/2 pi_PUCT + 1/2 pi_CFR`. PUCT is a
            // maximisation, so its half is a point mass on the argmax, and
            // sampling the mixture is a coin flip between the two.
            //
            // Both halves are restricted to the cells this world can still
            // grow through. A config whose every legal action is a dead end
            // ends the trajectory here.
            if (rng_unit(&s) < 0.5) {
                cell = puct_choice(t, node, a, b, 1 - me, c_puct);
            } else {
                bool mine = false;
                for (unsigned int q = a + threadIdx.x; q < b; q += 32)
                    mine |= t.sum[so + q] > 0.0f;
                bool any = __any_sync(0xffffffff, mine);
                const float* row = any ? t.sum + so + a : t.cur + so + a;
                cell = pick_live(t, so, a, b, row, &s);
            }
            if (cell == NO_ROW) break;
            // Counted as the trajectory passes, which is also the virtual loss
            // Student of Games adds across the simulations of one iteration:
            // a later simulation of the same phase sees this one's visit.
            if (threadIdx.x == 0) t.visits[so + cell] += 1.0f;
            __syncwarp();
            c[me] = (int)t.legal_trans[so + cell];
            node = t.legal_child[so + cell];
        }
        if (found == NO_ROW) continue;
        // The tree is frozen until the round ends, so a leaf a trajectory of
        // this round already took would grow nothing twice and the phase draws
        // again. `NO_ROW` never matches, so a padded slot scans as empty.
        // The whole warp scans the phases before this one: it is an equality
        // search, so nothing about it depends on the order the rows are read.
        bool dup = false;
        for (int k = threadIdx.x; k < (iter + 1) * sims; k += 32) {
            const unsigned int* r = out + (size_t)(k / sims) * each + (size_t)part * sims;
            dup |= r[k % sims] == found;
        }
        if (!__any_sync(0xffffffff, dup)) {
            if (threadIdx.x == 0) taken[got] = found;
            // The row just written is one the next draw scans.
            __syncwarp();
            ++got;
        }
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
__global__ void k_finish(const Tree* trees, const unsigned int* work, int at,
                         int level, const int* touched) {
    unsigned int item = work[at + blockIdx.x];
    unsigned int part = item >> WORK_BITS;
    int mask = touched[part];
    if (mask < 0) return;
    const Tree& t = trees[part];
    unsigned int node = work_node(t, level, item);
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
