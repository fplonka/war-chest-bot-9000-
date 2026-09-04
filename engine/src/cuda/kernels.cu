
extern "C" {

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
    return v;
}

__device__ __forceinline__ float gelu1(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

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
            if (g == 0) v = row[ROW_WP];
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

__global__ void k_norm_ip(float* x, const float* gamma, const float* beta,
                          int rows, int width, int act) {
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    float* row = x + (size_t)r * width;
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

__global__ void k_bias(float* out, const float* b, int rows, int width) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float* row = out + (size_t)r * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) row[j] += b[j];
}


__global__ void k_window(const float* src, float* out, int rows, int stride,
                         int off, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    out[i] = src[(size_t)(i / width) * stride + off + (i % width)];
}

__global__ void k_tokens(const float* cards, const int* card_of_row,
                         const float* seat, float* out, int rows, int ntype,
                         int type, int nslot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ntype * type) return;
    int r = i / (ntype * type), t = (i / type) % ntype, j = i % type;
    out[i] += cards[((size_t)card_of_row[r] * ntype + t) * type + j]
            + seat[(size_t)(t / nslot) * type + j];
}

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


#define TRUNK_OFF 12
#define TRUNK_MT (TRUNK_ROWS / 16)
#define TRUNK_KS (TRUNK_C / 8)
#define TRUNK_SPAN (TRUNK_C / 8)
#define TRUNK_MAXH (TRUNK_ROWS / TRUNK_SPAN)
#define TRUNK_Q (TRUNK_C / 32)
#define TRUNK_LDS (TRUNK_C + 4)

__device__ __forceinline__ float tf32(float v) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return __uint_as_float(r);
}

__device__ __forceinline__ void mma_tile(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void frag_a(const float* a, int m, int k, int lane,
                                       int lds, unsigned (&f)[4]) {
    const float* p = a + (size_t)(16 * m + (lane >> 2)) * lds + k + (lane & 3);
    f[0] = __float_as_uint(p[0]);
    f[1] = __float_as_uint(p[8 * lds]);
    f[2] = __float_as_uint(p[4]);
    f[3] = __float_as_uint(p[8 * lds + 4]);
}

__device__ __forceinline__ void frag_b(const float* w, int k, int slot, int lane,
                                       int ntiles, unsigned (&f)[2]) {
    float2 v = *(const float2*)(w + (((size_t)k * ntiles + slot) * 32 + lane) * 2);
    f[0] = __float_as_uint(v.x);
    f[1] = __float_as_uint(v.y);
}

__device__ __forceinline__ int frag_row(int m, int i, int lane) {
    return 16 * m + (lane >> 2) + ((i & 2) ? 8 : 0);
}
__device__ __forceinline__ int frag_col(int i, int slot, int lane) {
    return 8 * slot + 2 * (lane & 3) + (i & 1);
}

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

__device__ __forceinline__ void pool_rows(const float* v, int nhex, int c, float* out) {
    const int lane = threadIdx.x, slot = threadIdx.y;
    for (int i = 0; i < 8; ++i) {
        int j = 8 * slot + i;
        float lo = lane < nhex ? v[lane * TRUNK_LDS + j] : 0.0f;
        float hi = lane + 32 < nhex ? v[(lane + 32) * TRUNK_LDS + j] : 0.0f;
        float sum = warp_sum(lo + hi);
        float mx = fmaxf(lane < nhex ? lo : neg_inf(), lane + 32 < nhex ? hi : neg_inf());
        for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, s));
        if (lane == 0) {
            out[j] = sum / nhex;
            out[c + j] = mx;
        }
    }
}

__global__ __launch_bounds__(32 * TRUNK_SPAN, TRUNK_MIN_BLOCKS)
void k_trunk(float* x0, const int* nb, const float* __restrict__ w,
             const float* __restrict__ wt, const float* __restrict__ bias,
             const float* __restrict__ ln,
             const int* __restrict__ off, const float* xpub, float* out,
             int rows, int nhex, int c, int blocks, int stride, int off_loose,
             int loose) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    float* x = sm;
    float* a = x + nhex * TRUNK_LDS;
    float* u = a + TRUNK_ROWS * TRUNK_LDS;
    float* pooled = u + nhex * TRUNK_LDS;
    float* gb = pooled + 2 * c;

    const int lane = threadIdx.x, slot = threadIdx.y;
    int hex[TRUNK_MAXH];
    bool live[TRUNK_MAXH];
#pragma unroll
    for (int t = 0; t < TRUNK_MAXH; ++t) {
        hex[t] = slot + t * TRUNK_SPAN;
        live[t] = hex[t] < nhex;
    }
    float cur[TRUNK_Q];

#pragma unroll
    for (int t = 0; t < TRUNK_MAXH; ++t)
        if (live[t])
            for (int q = 0; q < TRUNK_Q; ++q)
                x[hex[t] * TRUNK_LDS + lane + 32 * q] =
                    x0[((size_t)row * nhex + hex[t]) * TRUNK_C + lane + 32 * q];
    __syncthreads();

    for (int blk = 0; blk < blocks; ++blk) {
        const int* o = off + blk * TRUNK_OFF;
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t) {
            for (int q = 0; q < TRUNK_Q; ++q)
                cur[q] = live[t] ? x[hex[t] * TRUNK_LDS + lane + 32 * q] : 0.0f;
            float mean, inv;
            row_stats(cur, c, &mean, &inv);
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                a[hex[t] * TRUNK_LDS + j] =
                    live[t] ? tf32(gelu1((cur[q] - mean) * inv * ln[o[6] + j] + ln[o[7] + j]))
                            : 0.0f;
            }
        }
        __syncthreads();
        pool_rows(a, nhex, c, pooled);
        __syncthreads();
        for (int q = 0; q < TRUNK_Q; ++q) {
            int j = lane + 32 * q;
            float sv = 0.0f;
            for (int k = 16 * slot; k < 16 * slot + 16; ++k) sv += pooled[k] * w[o[2] + (size_t)k * c + j];
            u[slot * c + j] = sv;
        }
        __syncthreads();
        for (int j = lane + 32 * slot; j < c; j += 32 * TRUNK_SPAN) {
            float sv = bias[o[3] + j];
            for (int s = 0; s < TRUNK_SPAN; ++s) sv += u[s * c + j];
            gb[j] = sv;
        }
        float an[TRUNK_MT][4], as[TRUNK_MT][4];
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) an[m][i] = as[m][i] = 0.0f;
        for (int k = 0; k < TRUNK_KS; ++k) {
            unsigned bs[2], bn[2];
            frag_b(wt + o[10], k, slot, lane, TRUNK_SPAN, bs);
            frag_b(wt + o[10] + (size_t)TRUNK_C * TRUNK_C, k, slot, lane, TRUNK_SPAN, bn);
#pragma unroll
            for (int m = 0; m < TRUNK_MT; ++m) {
                unsigned af[4];
                frag_a(a, m, 8 * k, lane, TRUNK_LDS, af);
                mma_tile(as[m], af, bs);
                mma_tile(an[m], af, bn);
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = frag_row(m, i, lane), j = frag_col(i, slot, lane);
                if (r < nhex) u[r * TRUNK_LDS + j] = an[m][i];
                a[r * TRUNK_LDS + j] = as[m][i] + bias[o[1] + j] + gb[j];
            }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < TRUNK_MAXH; ++t) {
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                float v = a[hex[t] * TRUNK_LDS + j];
                if (live[t])
                    for (int k = 0; k < 6; ++k) {
                        int n = nb[hex[t] * 6 + k];
                        if (n >= 0) v += u[n * TRUNK_LDS + j];
                    }
                cur[q] = v;
            }
            float mean, inv;
            row_stats(cur, c, &mean, &inv);
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                a[hex[t] * TRUNK_LDS + j] =
                    live[t] ? tf32(gelu1((cur[q] - mean) * inv * ln[o[8] + j] + ln[o[9] + j]))
                            : 0.0f;
            }
        }
        __syncthreads();
        float ao[TRUNK_MT][4];
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) ao[m][i] = 0.0f;
        for (int k = 0; k < TRUNK_KS; ++k) {
            unsigned bo[2];
            frag_b(wt + o[11], k, slot, lane, TRUNK_SPAN, bo);
#pragma unroll
            for (int m = 0; m < TRUNK_MT; ++m) {
                unsigned af[4];
                frag_a(a, m, 8 * k, lane, TRUNK_LDS, af);
                mma_tile(ao[m], af, bo);
            }
        }
#pragma unroll
        for (int m = 0; m < TRUNK_MT; ++m)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = frag_row(m, i, lane), j = frag_col(i, slot, lane);
                if (r < nhex) x[r * TRUNK_LDS + j] += ao[m][i] + bias[o[5] + j];
            }
        __syncthreads();
    }

    const int* tn = off + blocks * TRUNK_OFF;
#pragma unroll
    for (int t = 0; t < TRUNK_MAXH; ++t) {
        for (int q = 0; q < TRUNK_Q; ++q)
            cur[q] = live[t] ? x[hex[t] * TRUNK_LDS + lane + 32 * q] : 0.0f;
        float mean, inv;
        row_stats(cur, c, &mean, &inv);
        if (live[t])
            for (int q = 0; q < TRUNK_Q; ++q) {
                int j = lane + 32 * q;
                float v = gelu1((cur[q] - mean) * inv * ln[tn[0] + j] + ln[tn[1] + j]);
                x[hex[t] * TRUNK_LDS + j] = v;
                x0[((size_t)row * nhex + hex[t]) * c + j] = v;
            }
    }
    __syncthreads();
    int width = 2 * c + loose;
    pool_rows(x, nhex, c, out + (size_t)row * width);
    for (int k = lane + 32 * slot; k < loose; k += 32 * TRUNK_SPAN)
        out[(size_t)row * width + 2 * c + k] = xpub[(size_t)row * stride + off_loose + k];
}

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
    else slots[i] = cards[((size_t)owner[cfg] * nslot + k) * type + j - 3];
}

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

__global__ void k_bag(const float* bag, const float* phi,
                      const unsigned int* owner, float* g, int n, int nslot,
                      int ntype, int cfeat, int pool) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * pool) return;
    int cfg = i / pool, j = i % pool;
    const float* v = bag + (size_t)owner[cfg] * nslot * 3 * pool;
    float acc = 0.0f;
    for (int k = 0; k < nslot; ++k)
        for (int zone = 0; zone < 3; ++zone) {
            float count = phi[(size_t)cfg * cfeat + zone * nslot + k];
            if (count != 0.0f) acc += count * v[((size_t)k * 3 + zone) * pool + j];
        }
    g[i] += acc;
}


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


__device__ __forceinline__ float cfr_factor(float t, float p) {
    if (isinf(p)) return p > 0.0f ? 1.0f : 0.0f;
    float x = powf(t, p);
    return x / (x + 1.0f);
}

#define SMOOTH 1e-30f
#define NO_ROW 0xffffffffu
#define NO_TRANS 0xffffffffu
#define KIND_LEAF 2u
#define KIND_CHANCE 1u

#define WORK_BITS 20
#define WORK_SLOT ((1u << WORK_BITS) - 1)

__device__ __forceinline__ unsigned int work_node(const Tree& t, int level,
                                                  unsigned int item) {
    return t.level_node[t.level_start[level] + (item & WORK_SLOT)];
}

__device__ __forceinline__ unsigned int rbase(const Tree& t, unsigned int i, int p) {
    return t.roff[i] + (p == 1 ? t.nc[2 * i] : 0);
}

__global__ void k_reach_sweep(const Tree* trees, const unsigned int* work, int at,
                              int level, int avg, int iter) {
    unsigned int item = work[at + blockIdx.x];
    const Tree& t = trees[item >> WORK_BITS];
    if ((unsigned long long)iter >= t.todo) return;
    const float* strat = avg ? t.avg : t.cur;
    unsigned int node = work_node(t, level, item);
    unsigned int par = t.parent[node];
    if (par == NO_ROW) return;
    unsigned int me = t.player[par];
    unsigned int p = blockIdx.y;
    {
        unsigned int n = t.nc[2 * node + p];
        unsigned int dst = rbase(t, node, p), src = rbase(t, par, p);
        for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
            if (p != me) {
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
}

__global__ void k_backprop_sweep(const Tree* trees, const unsigned int* work, int at,
                                 int level, int avg, int iter,
                                 float alpha, float beta, float gamma, float predict) {
    unsigned int item = work[at + blockIdx.x];
    const Tree& t = trees[item >> WORK_BITS];
    if ((unsigned long long)iter >= t.todo) return;
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
        unsigned int a = t.child_at[node], k = t.child_n[node];
        for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
            float v = 0.0f;
            for (unsigned int j = a; j < a + k; ++j) v += vals[t.voff[t.child[j]] + c];
            vals[vi + c] = v;
        }
        return;
    }

    unsigned int so = t.soff[node], lb = t.legal_base[node];
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
    float m = (float)(t.step + (unsigned long long)iter) + 1.0f;
    float da = cfr_factor(m, alpha), db = cfr_factor(m, beta);
    float dg = powf((m - 1.0f) / m, gamma);
    unsigned int ra = rbase(t, node, traverser);
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
            t.sum[so + cell] =
                t.sum[so + cell] * dg + t.reach[ra + c] * t.cur[so + cell];
            float v = fmaxf(r + predict * delta, 0.0f);
            t.cur[so + cell] = v;
            total += v;
        }
        total = warp_sum(total);
        if (total > SMOOTH) {
            float inv = 1.0f / total;
            for (unsigned int cell = a + lane; cell < b; cell += 32) t.cur[so + cell] *= inv;
        } else {
            float v = 1.0f / (float)(b - a);
            for (unsigned int cell = a + lane; cell < b; cell += 32) t.cur[so + cell] = v;
        }
    }
}

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

__device__ __forceinline__ void join_norm(float (&z)[J_MT][4], float* act,
                                          const float* gamma, const float* beta,
                                          const float* add) {
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
    __shared__ __align__(16) float shared[J_ROWS * J_D];
    float* act = shared;
    float* pooled = shared + J_ROWS * J_LDS;

    int lane = threadIdx.x, slot = threadIdx.y;
    int tid = lane + 32 * slot, nt = 32 * J_SPAN;
    int row0 = blockIdx.x * J_ROWS;
    float z[J_MT][4];

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
        float inv = total > SMOOTH ? 1.0f / total : 1.0f / (float)max(n, 1u);
        for (int j = lane; j < J_POOL; j += 32) {
            float acc = 0.0f;
            for (unsigned int k = lo; k < hi; ++k) {
                float belief = total > SMOOTH ? t.reach[ra + k - lo] * inv : inv;
                acc += belief * t.g[(size_t)t.cidx[k - base] * J_POOL + j];
            }
            pooled[(size_t)qr * J_POOL + j] = acc;
        }
    }
    __syncthreads();

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
        float bias = *cf_bias;
        float* opinion = t.opinion + traverser * t.nvals;
        unsigned int vo = t.voff[node];
        for (unsigned int k = lo; k < hi; ++k) {
            const float* fr = t.f + (size_t)t.cidx[cs + k - lo] * J_D;
            float acc = 0.0f;
#pragma unroll
            for (int q = 0; q < J_D / 32; ++q) acc += fr[lane + 32 * q] * h[q];
            for (int s = 16; s > 0; s >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, s);
            if (lane == 0) opinion[vo + k - lo] = acc + bias;
        }
    }
}

__global__ void k_leaf_scale(const Tree* trees, const int* part_of_row,
                             const int* local_row, int rows) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int traverser = row & 1, r = row >> 1, opp = 1 - traverser;
    const Tree& t = trees[part_of_row[r]];
    unsigned int node = t.leaf_node[local_row[r]];
    unsigned int n = t.nc[2 * node + opp], ra = rbase(t, node, opp);
    float mass = 0.0f;
    for (unsigned int c = threadIdx.x; c < n; c += 32) mass += t.reach[ra + c];
    mass = warp_sum(mass);
    unsigned int m = t.nc[2 * node + traverser], vo = t.voff[node];
    const float* opinion = t.opinion + traverser * t.nvals;
    float* vals = t.vals + traverser * t.nvals;
    for (unsigned int c = threadIdx.x; c < m; c += 32) vals[vo + c] = opinion[vo + c] * mass;
}


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

__device__ __forceinline__ float pick_sum(const float* w, int n) {
    float total = 0.0f;
    for (int i = threadIdx.x; i < n; i += 32) total += fmaxf(w[i], 0.0f);
    return warp_sum(total);
}

__device__ int pick_from(const float* w, int n, float total, unsigned long long* s) {
    if (!(total > 0.0f)) return n > 0 ? (int)(rng_next(s) % (unsigned long long)n) : 0;
    double needle = rng_unit(s) * (double)total;
    for (int i = 0; i < n; ++i) {
        needle -= (double)fmaxf(w[i], 0.0f);
        if (needle < 0.0) return i;
    }
    return n - 1;
}

__device__ __forceinline__ int pick(const float* w, int n, unsigned long long* s) {
    return pick_from(w, n, pick_sum(w, n), s);
}

__device__ bool live_cell(const Tree& t, unsigned int so, unsigned int cell) {
    return t.legal_trans[so + cell] != NO_TRANS
        && t.exhausted[t.legal_child[so + cell]] == 0u;
}

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

__device__ unsigned int puct_choice(const Tree& t, unsigned int node, unsigned int a,
                                    unsigned int b, int opp, float c_puct) {
    unsigned int so = t.soff[node], ra = rbase(t, node, opp);
    unsigned int nc = t.nc[2 * node + opp];
    float mass = 0.0f;
    for (unsigned int i = threadIdx.x; i < nc; i += 32) mass += t.reach[ra + i];
    mass = warp_sum(mass);
    float scale = mass > SMOOTH ? __fdiv_rn(1.0f, mass) : 0.0f;
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


__global__ void k_act_feats(const Tree* trees, const unsigned int* part,
                            const unsigned int* row_of, const unsigned int* desc,
                            const unsigned int* node_of, const float* kind,
                            const float* role, float* out, int n, int ntype,
                            int nhex, int chan) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * chan) return;
    int action = i / chan, j = i % chan;
    const unsigned int* d = desc + 6 * action;
    int node = node_of[action];
    const Tree& t = trees[part[node]];
    unsigned int board = t.board_of[row_of[node]];
    float v = kind[(size_t)d[0] * chan + j];
    for (int r = 0; r < 5; ++r) {
        unsigned int entity = d[r + 1];
        const float* src = nullptr;
        if (r < 2 && entity < (unsigned int)ntype)
            src = t.tokens + ((size_t)board * ntype + entity) * chan;
        if (r >= 2 && entity < (unsigned int)nhex)
            src = t.spatial + ((size_t)board * nhex + entity) * chan;
        if (src) v += role[r * chan + j] * src[j];
    }
    out[i] = v;
}

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
        float inv = total > SMOOTH ? 1.0f / total : 1.0f / (float)max(n, 1u);
        unsigned int cs = t.coff[2 * row_of[k] + player];
        for (int j = lane; j < pool; j += 32) {
            float acc = 0.0f;
            for (unsigned int c = 0; c < n; ++c) {
                float belief = total > SMOOTH ? t.reach[ra + c] * inv : inv;
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

__global__ void k_act_add(float* z, const float* proj, const unsigned int* of,
                          int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    z[i] += proj[(size_t)of[i / width] * width + i % width];
}

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
    float scale = total > SMOOTH ? 1.0f / total : 1.0f / (float)(b - a);
    for (unsigned int cell = a + lane; cell < b; cell += 32) t.prior[so + cell] *= scale;
}

__device__ __forceinline__ unsigned long long rng_seed(unsigned long long seed,
                                                       unsigned long long iter,
                                                       unsigned long long draw) {
    unsigned long long z = seed + iter * 0x9E3779B97F4A7C15ULL + (draw + 1) * 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z ? z : 1;
}

__global__ void k_expand(const Tree* trees, unsigned int* out, int parts, int sims,
                         float c_puct, int iter, int each, int tries, int depth) {
    extern __shared__ unsigned int path[];
    int part = blockIdx.x;
    if (part >= parts) return;
    const Tree& t = trees[part];
    const int lane = threadIdx.x, warp = threadIdx.y, warps = blockDim.y;
    const int draws = sims * tries;
    unsigned int* cand = path + (size_t)draws * depth;
    unsigned int* len = cand + draws;
    unsigned int* taken = out + (size_t)iter * each + (size_t)part * sims;
    for (int i = lane + 32 * warp; i < sims; i += 32 * warps) taken[i] = NO_ROW;
    if ((unsigned long long)iter >= t.todo) return;
    int want = (int)t.nexpand;
    unsigned int n0 = t.nc[0], n1 = t.nc[1];
    const float* root = t.reach + t.roff[0];
    for (int draw = warp; draw < draws; draw += warps) {
        unsigned int* mine = path + (size_t)draw * depth;
        unsigned int found = NO_ROW;
        int steps = 0;
        if (draw < want * tries) {
            unsigned long long s = rng_seed(*t.seed, t.step + iter, draw);
            int c[2];
            c[0] = pick(root, (int)n0, &s);
            c[1] = pick(root + n0, (int)n1, &s);
            unsigned int node = 0;
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
                if (rng_unit(&s) < 0.5) {
                    cell = puct_choice(t, node, a, b, 1 - me, c_puct);
                } else {
                    bool any_sum = false;
                    for (unsigned int q = a + lane; q < b; q += 32) any_sum |= t.sum[so + q] > 0.0f;
                    any_sum = __any_sync(0xffffffff, any_sum);
                    const float* row = any_sum ? t.sum + so + a : t.cur + so + a;
                    cell = pick_live(t, so, a, b, row, &s);
                }
                if (cell == NO_ROW) break;
                if (lane == 0) mine[steps] = so + cell;
                ++steps;
                c[me] = (int)t.legal_trans[so + cell];
                node = t.legal_child[so + cell];
            }
        }
        if (lane == 0) {
            cand[draw] = found;
            len[draw] = steps;
        }
    }
    __syncthreads();
    for (int draw = warp; draw < draws; draw += warps)
        for (int d = lane; d < (int)len[draw]; d += 32)
            atomicAdd(&t.visits[path[(size_t)draw * depth + d]], 1.0f);
    if (warp != 0) return;
    int got = 0;
    for (int draw = 0; draw < draws && got < want; ++draw) {
        unsigned int found = cand[draw];
        if (found == NO_ROW) continue;
        bool dup = false;
        for (int k = lane; k < (iter + 1) * sims; k += 32) {
            const unsigned int* r = out + (size_t)(k / sims) * each + (size_t)part * sims;
            dup |= r[k % sims] == found;
        }
        if (__any_sync(0xffffffff, dup)) continue;
        if (lane == 0) taken[got] = found;
        __syncwarp();
        ++got;
    }
}

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
    float u = t.player[node] == (unsigned)traverser ? t.util[node] : -t.util[node];
    unsigned int m = t.nc[2 * node + traverser], vo = t.voff[node];
    float* vals = t.vals + traverser * t.nvals;
    for (unsigned int c = threadIdx.x; c < m; c += 32) vals[vo + c] = u * acc;
}

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
            t.avg[so + cell] = sum > SMOOTH ? t.sum[so + cell] / sum : 1.0f / k;
    }
}

}
