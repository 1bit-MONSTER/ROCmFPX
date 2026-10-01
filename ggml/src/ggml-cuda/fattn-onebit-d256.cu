// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Prompt-processing FlashAttention for 256-wide heads on gfx1151.
//
// Work split: one workgroup = 8 waves = 64 "packed" query rows of one KV head.
// Packed row r maps to query position r / gqa and query head
// kv_head * gqa + r % gqa, so every staged K/V tile is used by all query heads
// that share it. Waves come in pairs; a pair owns 16 packed rows and each wave
// of the pair owns one half (128 channels) of the head dimension, for both the
// Q*K contraction and the P*V output. That halves the per-wave accumulator and
// lets each wave keep its Q slice in registers.
//
// Per 32-token KV tile:
//   1. the workgroup stores the prefetched K (row-major) and V (transposed)
//      tile to LDS and starts the global loads of the next tile;
//   2. each wave computes partial 16x32 scores over its 128 channels;
//   3. the pair sums its partial scores through LDS (reusing the K buffer),
//      both waves run the same online softmax in the base-2 domain;
//   4. the F16 probabilities go through LDS to become the A operand of the
//      P*V product over the wave's 128 output channels.
//
// gfx11 wave32 WMMA operand layouts used here:
//   A (16x16 f16): lane l%16 holds row l%16, its 16 K-dimension values;
//   B (16x16 f16): lane l%16 holds column l%16, its 16 K-dimension values;
//   C (16x16 f32): lane l holds column l%16 of rows 2*i + l/16, i = 0..7.
// Lanes 16..31 repeat the A/B operands of lanes 0..15.
//
// A pre-pass finds, for every query position, the half-open range of KV
// positions whose mask is not -inf and whether the mask inside it is all zero.
// A workgroup visits only tiles inside the union of its positions' ranges and
// reads the mask only for tiles not fully inside the dense intersection.
//
// Sparse prefill (GGML_ONEBIT_FLASH_PREFILL=<alpha>), after FlashPrefill V2
// (Fan et al., arXiv:2608.19758), implemented here from the paper's equations:
//   * fa256_block_means averages K and V over 128-token blocks;
//   * fa256_select scores, for each workgroup's 64 query rows, every block that
//     lies fully inside the rows' dense visible range against the block-mean
//     keys (same WMMA path as Q*K), accumulating per-block energies
//     sum_rows 2^(s - M) with a running maximum M, and keeps a block when its
//     energy is at least alpha times the largest. The first kSink tokens, the
//     kWindow tokens before the diagonal and every partially visible block are
//     always kept;
//   * fa256_prefill skips the pruned blocks' tiles and instead runs correction
//     tiles whose rows are the pruned blocks' mean K/V with the logit raised by
//     log2(kBlock): each pruned block contributes kBlock * 2^(s_mean) to the
//     softmax denominator and the matching multiple of its mean V to the output.

#include "fattn-onebit-d256.cuh"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#if defined(GGML_USE_HIP)

namespace {

constexpr int kHead      = 256;            // head size (K and V)
constexpr int kHalf      = kHead / 2;      // channels per wave of a pair
constexpr int kWaves     = 8;
constexpr int kRows      = 16 * (kWaves / 2);  // packed query rows per workgroup
constexpr int kTile      = 32;             // KV tokens per tile
constexpr int kKStride   = kHead + 8;      // LDS row stride (halves) for K
constexpr int kVtStride  = kTile + 8;      // LDS row stride (halves) for transposed V and P (even)
constexpr int kThreads   = 32 * kWaves;
constexpr int kLoads     = kTile * kHead / 8 / kThreads;  // 16-byte K (and V) loads per thread per tile
constexpr float kLog2e   = 1.4426950408889634f;
constexpr int kBlock     = 128;            // sparse selection block (tokens)
constexpr int kSink      = 256;            // always-kept leading tokens
constexpr int kWindow    = 512;            // always-kept tokens before the diagonal
constexpr float kLogBlock = 7.0f;          // log2(kBlock): a mean row stands for kBlock tokens
constexpr int kMaxBlocks = 2048;           // selection state in LDS (n_kv <= 256K)
static_assert(kBlock % kTile == 0, "a block is a whole number of tiles");

typedef _Float16 half16_t  __attribute__((ext_vector_type(16)));
typedef float    float8_t  __attribute__((ext_vector_type(8)));

struct fa256_params {
    const char * q;
    const char * k;
    const char * v;
    const char * mask;
    char       * dst;
    int64_t q_nb1, q_nb2;
    int64_t k_nb1, k_nb2;
    int64_t v_nb1, v_nb2;
    int     mask_row_halves;
    int64_t dst_nb1, dst_nb2;
    int     n_q;
    int     n_kv;
    int     gqa;
    float   scale_log2;
    const int4 * range;
    // Sparse prefill; keep == nullptr means dense.
    const uint32_t * keep;   // per workgroup: keep_words words, bit b = block b is computed exactly
    int              keep_words;
    const half     * kbar;   // [kv head][block][kHead] block-mean keys
    const half     * vbar;   // [kv head][block][kHead] block-mean values
    int              nblk;
};

// Visited KV range of a workgroup's rows (union of the positions' ranges) and,
// when every row is dense, the intersection inside which no mask is needed.
// Written to s_range by thread 0: {lo, hi, full_lo, full_hi}; an empty
// intersection is {INT_MAX, 0}.
__device__ __forceinline__ void fa256_group_range(const fa256_params & p, int row0, int rows, int * s_range) {
    const int last    = min(row0 + kRows, rows) - 1;
    int       lo      = INT_MAX;
    int       hi      = 0;
    int       full_lo = 0;
    int       full_hi = INT_MAX;
    bool      dense   = true;
    for (int pos = row0 / p.gqa; pos <= last / p.gqa; ++pos) {
        const int4 r = p.range[pos];
        if (r.y > r.x) {
            lo = min(lo, r.x);
            hi = max(hi, r.y);
        }
        full_lo = max(full_lo, r.x);
        full_hi = min(full_hi, r.y);
        dense   = dense && r.z != 0;
    }
    s_range[0] = lo;
    s_range[1] = hi;
    s_range[2] = dense ? full_lo : INT_MAX;
    s_range[3] = dense ? full_hi : 0;
}

// A wave's Q slice as WMMA A operands: packed row row0 + pair*16 + col,
// channels part*kHalf + 16*ks.
__device__ __forceinline__ void fa256_load_q(const fa256_params & p, int row0, int rows, int pair, int part, int col,
                                             int kvh, half16_t (&qa)[kHalf / 16]) {
    const int gr = row0 + pair * 16 + col;
    const float * qrow = nullptr;
    if (gr < rows) {
        const int pos  = gr / p.gqa;
        const int head = kvh * p.gqa + gr % p.gqa;
        qrow = (const float *) (p.q + pos * p.q_nb1 + head * p.q_nb2) + part * kHalf;
    }
#pragma unroll
    for (int ks = 0; ks < kHalf / 16; ++ks) {
#pragma unroll
        for (int e = 0; e < 16; e += 4) {
            float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (qrow != nullptr) {
                x = *(const float4 *) (qrow + ks * 16 + e);
            }
            qa[ks][e + 0] = (_Float16) x.x;
            qa[ks][e + 1] = (_Float16) x.y;
            qa[ks][e + 2] = (_Float16) x.z;
            qa[ks][e + 3] = (_Float16) x.w;
        }
    }
}

// Q*K for one 32-row K tile in LDS: each wave of a pair contracts its 128
// channels, then the pair sums partial scores through xch (which aliases the K
// tile, so the caller's barriers order the reuse). On return s0/s1 hold the full
// 16x16 scores of rows 2*i + hrow against tile rows col and 16 + col.
__device__ __forceinline__ void fa256_pair_scores(const half16_t (&qa)[kHalf / 16], const _Float16 * k_part,
                                                  float * xch, int wave, int lane, int col,
                                                  float8_t & s0, float8_t & s1) {
    s0 = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    s1 = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int ks = 0; ks < kHalf / 16; ++ks) {
        const half16_t b0 = *(const half16_t *) (k_part + col * kKStride + ks * 16);
        const half16_t b1 = *(const half16_t *) (k_part + (16 + col) * kKStride + ks * 16);
        s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b0, s0);
        s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b1, s1);
        __builtin_amdgcn_sched_barrier(0);  // bound operand hoisting (register pressure)
    }
    __syncthreads();  // K reads done; the K buffer now carries partial scores

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        xch[((wave * 2 + 0) * 8 + i) * 32 + lane] = s0[i];
        xch[((wave * 2 + 1) * 8 + i) * 32 + lane] = s1[i];
    }
    __syncthreads();
    const int other = wave ^ 1;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s0[i] += xch[((other * 2 + 0) * 8 + i) * 32 + lane];
        s1[i] += xch[((other * 2 + 1) * 8 + i) * 32 + lane];
    }
}

// Block means of K and V: one workgroup per (block, kv head). Thread t loads
// 16-byte chunks (8 channels) for channel group t % 32 and every 8th token from
// t / 32; the 8 partial sums per channel group are combined through LDS.
__global__ void __launch_bounds__(256) fa256_block_means(const fa256_params p, half * kbar, half * vbar) {
    __shared__ float s_k[8][kHead];
    __shared__ float s_v[8][kHead];
    const int blk = blockIdx.x;
    const int kvh = blockIdx.y;
    const int cg  = threadIdx.x % 32;  // channel group: channels 8*cg .. 8*cg+7
    const int ts  = threadIdx.x / 32;  // token stride slot
    const int kv0 = blk * kBlock;
    const int kv1 = min(kv0 + kBlock, p.n_kv);
    float sk[8] = {};
    float sv[8] = {};
    for (int kv = kv0 + ts; kv < kv1; kv += 8) {
        const uint4 xk = *(const uint4 *) (p.k + kv * p.k_nb1 + kvh * p.k_nb2 + cg * 8 * sizeof(half));
        const uint4 xv = *(const uint4 *) (p.v + kv * p.v_nb1 + kvh * p.v_nb2 + cg * 8 * sizeof(half));
        const half * hk = (const half *) &xk;
        const half * hv = (const half *) &xv;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sk[j] += __half2float(hk[j]);
            sv[j] += __half2float(hv[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s_k[ts][cg * 8 + j] = sk[j];
        s_v[ts][cg * 8 + j] = sv[j];
    }
    __syncthreads();
    const int   c   = threadIdx.x;
    const float inv = kv1 > kv0 ? 1.0f / (float) (kv1 - kv0) : 0.0f;
    float tk = 0.0f;
    float tv = 0.0f;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        tk += s_k[r][c];
        tv += s_v[r][c];
    }
    const size_t o = ((size_t) kvh * p.nblk + blk) * kHead + c;
    kbar[o] = __float2half(tk * inv);
    vbar[o] = __float2half(tv * inv);
}

// range[pos] = {first, last + 1, dense, 0}: the KV positions with a finite mask
// value, and whether every mask value in [first, last + 1) is exactly zero.
__global__ void fa256_kv_range(const char * mask, int64_t mask_nb1, int n_kv, int n_q, int4 * range) {
    const int pos = blockIdx.x;
    if (pos >= n_q) {
        return;
    }
    const half * row = (const half *) (mask + pos * mask_nb1);

    int lo      = INT_MAX;
    int hi      = -1;
    int finite  = 0;
    int nonzero = 0;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        const float m = __half2float(row[i]);
        if (m > -INFINITY) {
            lo = min(lo, i);
            hi = max(hi, i);
            finite++;
            nonzero += m != 0.0f;
        }
    }

    __shared__ int s_lo, s_hi, s_finite, s_nonzero;
    if (threadIdx.x == 0) {
        s_lo      = INT_MAX;
        s_hi      = -1;
        s_finite  = 0;
        s_nonzero = 0;
    }
    __syncthreads();
    atomicMin(&s_lo, lo);
    atomicMax(&s_hi, hi);
    atomicAdd(&s_finite, finite);
    atomicAdd(&s_nonzero, nonzero);
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_hi < 0) {
            range[pos] = make_int4(0, 0, 1, 0);
        } else {
            const int dense = s_nonzero == 0 && s_finite == s_hi + 1 - s_lo;
            range[pos] = make_int4(s_lo, s_hi + 1, dense, 0);
        }
    }
}

// kCorr = false: cache tiles (all, or the kept blocks' tiles when sparse); when
// sparse it also saves each row's softmax state (max, sum) to ml.
// kCorr = true: resumes from dst + ml and adds the correction tiles of the
// pruned blocks. Splitting the passes keeps each instantiation's registers lean.
template <bool kCorr>
__global__ void __launch_bounds__(kThreads) fa256_prefill(const fa256_params p, float2 * ml) {
#if defined(__GFX11__)
    __shared__ __attribute__((aligned(16))) _Float16 k_lds[kTile * kKStride];
    __shared__ __attribute__((aligned(16))) _Float16 vt_lds[kHead * kVtStride];
    __shared__ __attribute__((aligned(16))) _Float16 p_lds[kRows * kVtStride];
    __shared__ int s_range[4];
    __shared__ int s_cblk[kTile];   // correction tile rows: pruned block ids, -1 = empty
    __shared__ int s_ccount;        // valid rows in s_cblk
    __shared__ int s_kv0_next;      // where the next correction scan starts
    __shared__ uint32_t s_keep[kMaxBlocks / 32];  // this workgroup's keep bits (sparse only)
    static_assert(kWaves * 2 * 8 * 32 * sizeof(float) <= sizeof(k_lds), "score exchange must fit in the K buffer");

    const int tid  = threadIdx.x;
    const int wave = tid / 32;
    const int pair = wave / 2;
    const int part = wave % 2;      // which half of the head dimension
    const int lane = tid % 32;
    const int col  = lane % 16;
    const int hrow = lane / 16;
    const int kvh  = blockIdx.y;
    const int row0 = blockIdx.x * kRows;
    const int rows = p.n_q * p.gqa;

    if (tid == 0) {
        fa256_group_range(p, row0, rows, s_range);
    }
    // This wave's Q slice as A operands: row = pair*16 + col, channels part*128 + 16*ks.
    half16_t qa[kHalf / 16];
    {
        const int gr = row0 + pair * 16 + col;
        const float * qrow = nullptr;
        if (gr < rows) {
            const int pos  = gr / p.gqa;
            const int head = kvh * p.gqa + gr % p.gqa;
            qrow = (const float *) (p.q + pos * p.q_nb1 + head * p.q_nb2) + part * kHalf;
        }
#pragma unroll
        for (int ks = 0; ks < kHalf / 16; ++ks) {
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (qrow != nullptr) {
                    x = *(const float4 *) (qrow + ks * 16 + e);
                }
                qa[ks][e + 0] = (_Float16) x.x;
                qa[ks][e + 1] = (_Float16) x.y;
                qa[ks][e + 2] = (_Float16) x.z;
                qa[ks][e + 3] = (_Float16) x.w;
            }
        }
    }
    __syncthreads();

    const int tile_lo = s_range[0] < s_range[1] ? s_range[0] / kTile : 0;
    const int tile_hi = s_range[0] < s_range[1] ? (s_range[1] + kTile - 1) / kTile : 0;
    const int full_lo = s_range[2];
    const int full_hi = s_range[3];

    // Rows this lane sees in the score/output layout: 2*i + hrow of the pair's 16.
    // Mask row offsets in halves; -1 marks a row past the end.
    const half * mask = (const half *) p.mask;
    int mask_off[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gr = row0 + pair * 16 + 2 * i + hrow;
        mask_off[i]  = gr < rows ? (gr / p.gqa) * p.mask_row_halves : -1;
    }

    float8_t acc[kHalf / 16];
#pragma unroll
    for (int j = 0; j < kHalf / 16; ++j) {
        acc[j] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
    float row_max[8];
    float row_sum[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }
    if constexpr (kCorr) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int gr = row0 + pair * 16 + 2 * i + hrow;
            if (gr >= rows) {
                continue;
            }
            const float2 st = ml[(size_t) kvh * rows + gr];
            row_max[i] = st.x;
            row_sum[i] = st.y;
            const int pos  = gr / p.gqa;
            const int head = kvh * p.gqa + gr % p.gqa;
            const float * out = (const float *) (p.dst + pos * p.dst_nb2 + head * p.dst_nb1) + part * kHalf;
#pragma unroll
            for (int j = 0; j < kHalf / 16; ++j) {
                acc[j][i] = out[j * 16 + col] * st.y;
            }
        }
    }

    // One tile's K and V as kLoads 16-byte chunks per thread. K chunks run along
    // the channels (coalesced, stored row-major); V chunks put one token per lane
    // so the transposed store below writes consecutive tokens of a channel row.
    uint4 kx[kLoads];
    uint4 vx[kLoads];
    const int v_tok = lane;                    // token within the tile
    auto load_tile = [&](int kind, int kv0) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int e  = tid + u * kThreads;
            const int r  = e / (kHead / 8);
            const int c  = (e % (kHead / 8)) * 8;
            const int cv = (wave + kWaves * u) * 8;
            kx[u] = make_uint4(0, 0, 0, 0);
            vx[u] = make_uint4(0, 0, 0, 0);
            if (kind == 1) {
                const int kv = kv0 + r;
                if (kv < p.n_kv) {
                    kx[u] = *(const uint4 *) (p.k + kv * p.k_nb1 + kvh * p.k_nb2 + c * sizeof(half));
                }
                const int kvv = kv0 + v_tok;
                if (kvv < p.n_kv) {
                    vx[u] = *(const uint4 *) (p.v + kvv * p.v_nb1 + kvh * p.v_nb2 + cv * sizeof(half));
                }
            } else {
                const int bk = s_cblk[r];
                if (bk >= 0) {
                    kx[u] = *(const uint4 *) (p.kbar + ((size_t) kvh * p.nblk + bk) * kHead + c);
                }
                const int bv = s_cblk[v_tok];
                if (bv >= 0) {
                    vx[u] = *(const uint4 *) (p.vbar + ((size_t) kvh * p.nblk + bv) * kHead + cv);
                }
            }
        }
    };

    float          * xch    = (float *) k_lds;  // partial-score exchange, after Q*K is done
    _Float16       * p_pair = p_lds + pair * 16 * kVtStride;
    const _Float16 * k_part = k_lds + part * kHalf;

    // Work items: kind 1 = cache tile at kv0 (main pass, kept blocks only),
    // kind 2 = correction tile with rows from s_cblk (correction pass). The tile
    // body is written once; kind is a constant in each instantiation. (A lambda
    // around the body made the compiler spill the accumulators.)
    const bool sparse = p.keep != nullptr;
    if (sparse) {
        const uint32_t * keep_g = p.keep + (size_t) (blockIdx.y * gridDim.x + blockIdx.x) * p.keep_words;
        for (int w = tid; w < p.keep_words; w += kThreads) {
            s_keep[w] = keep_g[w];
        }
        __syncthreads();
    }
    int next_tile  = tile_lo;
    int next_block = 0;
    for (;;) {
        int kind;
        int kv0;
        if constexpr (!kCorr) {
            while (next_tile < tile_hi && sparse) {
                const int      b = next_tile * kTile / kBlock;
                const uint32_t w = __builtin_amdgcn_readfirstlane(s_keep[b >> 5]);
                if ((w >> (b & 31)) & 1u) {
                    break;
                }
                ++next_tile;
            }
            if (next_tile >= tile_hi) {
                break;
            }
            kind = 1;
            kv0  = next_tile * kTile;
            ++next_tile;
        } else {
            if (tid == 0) {
                // Next pruned blocks (zero bits) from next_block on, via count-trailing-zeros.
                int n = 0;
                int b = next_block;
                while (b < p.nblk && n < kTile) {
                    uint32_t z = ~s_keep[b >> 5] & (~0u << (b & 31));
                    if (z == 0) {
                        b = (b | 31) + 1;
                        continue;
                    }
                    const int bb = (b & ~31) + __builtin_ctz(z);
                    if (bb >= p.nblk) {
                        b = p.nblk;
                        break;
                    }
                    s_cblk[n++] = bb;
                    b = bb + 1;
                }
                for (int k = n; k < kTile; ++k) {
                    s_cblk[k] = -1;
                }
                s_ccount   = n;
                s_kv0_next = b;
            }
            __syncthreads();
            if (__builtin_amdgcn_readfirstlane(s_ccount) == 0) {
                break;
            }
            next_block = __builtin_amdgcn_readfirstlane(s_kv0_next);
            kind = 2;
            kv0  = 0;
        }

        load_tile(kind, kv0);
        // V transpose: lanes 2m and 2m+1 hold tokens 2m and 2m+1 of the same 8
        // channels. The even lane writes channels 0..3 and the odd lane channels
        // 4..7, each as dwords packing (token 2m, token 2m+1).
        const bool even = (lane & 1) == 0;
        const int  tok0 = v_tok & ~1;
        const int  ch0  = even ? 0 : 4;
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int e = tid + u * kThreads;
            const int r = e / (kHead / 8);
            const int c = (e % (kHead / 8)) * 8;
            *(uint4 *) (k_lds + r * kKStride + c) = kx[u];

            const uint4    x  = vx[u];
            const uint32_t r0 = (uint32_t) __shfl_xor((int) (even ? x.z : x.x), 1);
            const uint32_t r1 = (uint32_t) __shfl_xor((int) (even ? x.w : x.y), 1);
            const uint32_t lo0 = even ? x.x : r0;   // token 2m:   channels ch0, ch0+1
            const uint32_t hi0 = even ? r0 : x.z;   // token 2m+1: channels ch0, ch0+1
            const uint32_t lo1 = even ? x.y : r1;   // token 2m:   channels ch0+2, ch0+3
            const uint32_t hi1 = even ? r1 : x.w;   // token 2m+1: channels ch0+2, ch0+3
            const int      cv  = (wave + kWaves * u) * 8 + ch0;
            uint32_t * vt32 = (uint32_t *) (vt_lds + cv * kVtStride + tok0);
            constexpr int row32 = kVtStride / 2;
            vt32[0 * row32] = (lo0 & 0xFFFFu) | (hi0 << 16);
            vt32[1 * row32] = (lo0 >> 16) | (hi0 & 0xFFFF0000u);
            vt32[2 * row32] = (lo1 & 0xFFFFu) | (hi1 << 16);
            vt32[3 * row32] = (lo1 >> 16) | (hi1 & 0xFFFF0000u);
        }
        __syncthreads();

        // Partial scores over this wave's 128 channels, two 16-token halves of the tile.
        float8_t s0 = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
        float8_t s1 = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < kHalf / 16; ++ks) {
            const half16_t b0 = *(const half16_t *) (k_part + col * kKStride + ks * 16);
            const half16_t b1 = *(const half16_t *) (k_part + (16 + col) * kKStride + ks * 16);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b0, s0);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b1, s1);
            __builtin_amdgcn_sched_barrier(0);  // bound operand hoisting (register pressure)
        }
        __syncthreads();  // K reads done; the K buffer now carries partial scores

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            xch[((wave * 2 + 0) * 8 + i) * 32 + lane] = s0[i];
            xch[((wave * 2 + 1) * 8 + i) * 32 + lane] = s1[i];
        }
        __syncthreads();
        const int other = wave ^ 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s0[i] += xch[((other * 2 + 0) * 8 + i) * 32 + lane];
            s1[i] += xch[((other * 2 + 1) * 8 + i) * 32 + lane];
        }

        // Cache tile: positions kva/kvb, masked near the diagonal. Correction
        // tile: block means, valid where the row holds a block, logit + log2(kBlock).
        const bool corr      = kind == 2;
        const int  kva       = corr ? (s_cblk[col] >= 0 ? 0 : p.n_kv) : kv0 + col;
        const int  kvb       = corr ? (s_cblk[16 + col] >= 0 ? 0 : p.n_kv) : kv0 + 16 + col;
        const bool need_mask = !corr && (kv0 < full_lo || kv0 + kTile > full_hi);  // workgroup-uniform
        const float bias     = corr ? kLogBlock : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float xa = -INFINITY;
            float xb = -INFINITY;
            if (mask_off[i] >= 0) {
                if (kva < p.n_kv) {
                    xa = s0[i] * p.scale_log2 + bias;
                    if (need_mask) {
                        xa += __half2float(mask[mask_off[i] + kva]) * kLog2e;
                    }
                }
                if (kvb < p.n_kv) {
                    xb = s1[i] * p.scale_log2 + bias;
                    if (need_mask) {
                        xb += __half2float(mask[mask_off[i] + kvb]) * kLog2e;
                    }
                }
            }
            float tile_max = fmaxf(xa, xb);
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 16));
            }
            const float m_new = fmaxf(row_max[i], tile_max);
            const float a     = m_new == -INFINITY ? 1.0f : exp2f(row_max[i] - m_new);
            const float pa    = m_new == -INFINITY ? 0.0f : exp2f(xa - m_new);
            const float pb    = m_new == -INFINITY ? 0.0f : exp2f(xb - m_new);
            float tile_sum = pa + pb;
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                tile_sum += __shfl_xor(tile_sum, off, 16);
            }
            row_sum[i] = row_sum[i] * a + tile_sum;
            row_max[i] = m_new;
            if (part == 0) {
                p_pair[(2 * i + hrow) * kVtStride + col]      = (_Float16) pa;
                p_pair[(2 * i + hrow) * kVtStride + 16 + col] = (_Float16) pb;
            }
#pragma unroll
            for (int j = 0; j < kHalf / 16; ++j) {
                acc[j][i] *= a;
            }
        }
        __syncthreads();

        const half16_t p0 = *(const half16_t *) (p_pair + col * kVtStride);
        const half16_t p1 = *(const half16_t *) (p_pair + col * kVtStride + 16);
#pragma unroll
        for (int j = 0; j < kHalf / 16; ++j) {
            const _Float16 * vrow = vt_lds + (part * kHalf + j * 16 + col) * kVtStride;
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p0, *(const half16_t *) (vrow), acc[j]);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p1, *(const half16_t *) (vrow + 16), acc[j]);
            __builtin_amdgcn_sched_barrier(0);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gr = row0 + pair * 16 + 2 * i + hrow;
        if (gr >= rows) {
            continue;
        }
        const int   pos  = gr / p.gqa;
        const int   head = kvh * p.gqa + gr % p.gqa;
        const float inv  = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        float * out = (float *) (p.dst + pos * p.dst_nb2 + head * p.dst_nb1) + part * kHalf;
        if constexpr (!kCorr) {
            if (sparse && part == 0 && col == 0) {
                ml[(size_t) kvh * rows + gr] = make_float2(row_max[i], row_sum[i]);
            }
        }
#pragma unroll
        for (int j = 0; j < kHalf / 16; ++j) {
            out[j * 16 + col] = acc[j][i] * inv;
        }
    }
#else
    GGML_UNUSED(p);
#endif // defined(__GFX11__)
}

// Block selection for sparse prefill: one workgroup per main-kernel workgroup
// (same 64 packed rows), writing its keep bitmask.
__global__ void __launch_bounds__(kThreads) fa256_select(const fa256_params p, uint32_t * keep_out, float alpha) {
#if defined(__GFX11__)
    __shared__ __attribute__((aligned(16))) _Float16 k_lds[kTile * kKStride];
    __shared__ float s_e[kMaxBlocks];   // per-block energy, taken against s_m
    __shared__ float s_m[kMaxBlocks];   // running maximum when the energy was taken
    __shared__ float s_wred[kWaves];
    __shared__ int   s_range[4];

    const int tid  = threadIdx.x;
    const int wave = tid / 32;
    const int pair = wave / 2;
    const int part = wave % 2;
    const int lane = tid % 32;
    const int col  = lane % 16;
    const int hrow = lane / 16;
    const int kvh  = blockIdx.y;
    const int row0 = blockIdx.x * kRows;
    const int rows = p.n_q * p.gqa;

    if (tid == 0) {
        fa256_group_range(p, row0, rows, s_range);
    }
    half16_t qa[kHalf / 16];
    fa256_load_q(p, row0, rows, pair, part, col, kvh, qa);
    __syncthreads();

    const int full_lo = s_range[2];
    const int full_hi = s_range[3];
    uint32_t * out = keep_out + (size_t) (blockIdx.y * gridDim.x + blockIdx.x) * p.keep_words;

    // Candidates: blocks fully inside the dense intersection of the rows' ranges.
    const bool any   = full_lo < full_hi;
    const int  cb_lo = any ? (full_lo + kBlock - 1) / kBlock : 0;
    const int  cb_hi = any ? full_hi / kBlock : 0;
    if (cb_lo >= cb_hi) {
        for (int w = tid; w < p.keep_words; w += kThreads) {
            out[w] = ~0u;
        }
        return;
    }

    bool row_ok[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        row_ok[i] = row0 + pair * 16 + 2 * i + hrow < rows;
    }

    float m_run = -INFINITY;
    for (int b0 = cb_lo; b0 < cb_hi; b0 += kTile) {
        for (int e = tid; e < kTile * kHead / 8; e += kThreads) {
            const int r = e / (kHead / 8);
            const int c = (e % (kHead / 8)) * 8;
            const int b = b0 + r;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (b < cb_hi) {
                x = *(const uint4 *) (p.kbar + ((size_t) kvh * p.nblk + b) * kHead + c);
            }
            *(uint4 *) (k_lds + r * kKStride + c) = x;
        }
        if (tid < kTile && b0 + tid < cb_hi) {
            s_e[b0 + tid] = 0.0f;
        }
        __syncthreads();

        float8_t s0;
        float8_t s1;
        fa256_pair_scores(qa, k_lds + part * kHalf, (float *) k_lds, wave, lane, col, s0, s1);

        const bool va = b0 + col < cb_hi;
        const bool vb = b0 + 16 + col < cb_hi;
        float lmax = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s0[i] = row_ok[i] && va ? s0[i] * p.scale_log2 : -INFINITY;
            s1[i] = row_ok[i] && vb ? s1[i] * p.scale_log2 : -INFINITY;
            lmax  = fmaxf(lmax, fmaxf(s0[i], s1[i]));
        }
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            lmax = fmaxf(lmax, __shfl_xor(lmax, off, 32));
        }
        if (lane == 0) {
            s_wred[wave] = lmax;
        }
        __syncthreads();
        float cmax = s_wred[0];
#pragma unroll
        for (int w = 1; w < kWaves; ++w) {
            cmax = fmaxf(cmax, s_wred[w]);
        }
        const float m_new = fmaxf(m_run, cmax);

        float ea = 0.0f;
        float eb = 0.0f;
        if (m_new > -INFINITY) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                ea += exp2f(s0[i] - m_new);
                eb += exp2f(s1[i] - m_new);
            }
        }
        ea += __shfl_xor(ea, 16, 32);
        eb += __shfl_xor(eb, 16, 32);
        if (part == 0 && hrow == 0) {
            if (va) {
                atomicAdd(&s_e[b0 + col], ea);
            }
            if (vb) {
                atomicAdd(&s_e[b0 + 16 + col], eb);
            }
        }
        if (tid < kTile && b0 + tid < cb_hi) {
            s_m[b0 + tid] = m_new;
        }
        m_run = m_new;
        __syncthreads();  // before the next chunk reuses k_lds and s_wred
    }

    // Rescale every energy to the final maximum and threshold against the largest.
    float lmax = 0.0f;
    for (int b = cb_lo + tid; b < cb_hi; b += kThreads) {
        s_e[b] *= exp2f(s_m[b] - m_run);
        lmax = fmaxf(lmax, s_e[b]);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        lmax = fmaxf(lmax, __shfl_xor(lmax, off, 32));
    }
    if (lane == 0) {
        s_wred[wave] = lmax;
    }
    __syncthreads();
    float emax = s_wred[0];
#pragma unroll
    for (int w = 1; w < kWaves; ++w) {
        emax = fmaxf(emax, s_wred[w]);
    }
    const float thresh = alpha * emax;

    for (int w = tid; w < p.keep_words; w += kThreads) {
        uint32_t bits = 0;
        for (int k = 0; k < 32; ++k) {
            const int b = w * 32 + k;
            bool kept = true;
            if (b >= cb_lo && b < cb_hi && b * kBlock >= kSink && (b + 1) * kBlock <= full_hi - kWindow) {
                kept = s_e[b] >= thresh;
            }
            bits |= (uint32_t) kept << k;
        }
        out[w] = bits;
    }
#else
    GGML_UNUSED(p);
    GGML_UNUSED(keep_out);
    GGML_UNUSED(alpha);
#endif // defined(__GFX11__)
}

bool aligned16(const void * ptr) {
    return ((uintptr_t) ptr) % 16 == 0;
}

}  // namespace

bool ggml_cuda_fattn_onebit_d256_eligible(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_ONEBIT_FA256");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V == nullptr || V->type != GGML_TYPE_F16) {
        return false;
    }
    if (Q->ne[0] != kHead || K->ne[0] != kHead || V->ne[0] != kHead) {
        return false;
    }
    if (Q->ne[1] < 16 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) {
        return false;
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->nb[1] % sizeof(half) ||
        (int64_t) mask->nb[1] * Q->ne[1] >= INT_MAX) {
        return false;
    }
    if (dst->src[4] != nullptr) {
        return false;  // attention sinks
    }
    float max_bias = 0.0f;
    float softcap  = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half)) {
        return false;
    }
    if (!aligned16(Q->data) || !aligned16(K->data) || !aligned16(V->data) || Q->nb[1] % 16 || Q->nb[2] % 16 ||
        K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != kHead || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1]) {
        return false;
    }
    return true;
}

void ggml_cuda_fattn_onebit_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_q  = (int) Q->ne[1];
    const int n_kv = (int) K->ne[1];
    const int gqa  = (int) (Q->ne[2] / K->ne[2]);

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int4> range(ctx.pool(), n_q);
    fa256_kv_range<<<n_q, 256, 0, stream>>>((const char *) mask->data, mask->nb[1], n_kv, n_q, range.get());

    fa256_params p;
    p.q          = (const char *) Q->data;
    p.k          = (const char *) K->data;
    p.v          = (const char *) V->data;
    p.mask       = (const char *) mask->data;
    p.dst        = (char *) dst->data;
    p.q_nb1      = Q->nb[1];
    p.q_nb2      = Q->nb[2];
    p.k_nb1      = K->nb[1];
    p.k_nb2      = K->nb[2];
    p.v_nb1      = V->nb[1];
    p.v_nb2      = V->nb[2];
    p.mask_row_halves = (int) (mask->nb[1] / sizeof(half));
    p.dst_nb1    = dst->nb[1];
    p.dst_nb2    = dst->nb[2];
    p.n_q        = n_q;
    p.n_kv       = n_kv;
    p.gqa        = gqa;
    p.scale_log2 = scale * kLog2e;
    p.range      = range.get();

    const dim3 grid((n_q * gqa + kRows - 1) / kRows, (unsigned) K->ne[2], 1);

    // Sparse prefill: GGML_ONEBIT_FLASH_PREFILL=<alpha> (0 or unset = dense),
    // for KV lengths of at least GGML_ONEBIT_FLASH_PREFILL_MIN (default 8192).
    static const float sparse_alpha = [] {
        const char * env = getenv("GGML_ONEBIT_FLASH_PREFILL");
        return env ? (float) atof(env) : 0.0f;
    }();
    static const int sparse_min_kv = [] {
        const char * env = getenv("GGML_ONEBIT_FLASH_PREFILL_MIN");
        return env ? atoi(env) : 8192;
    }();
    const int  nblk   = (n_kv + kBlock - 1) / kBlock;
    const bool sparse = sparse_alpha > 0.0f && n_kv >= sparse_min_kv && nblk <= kMaxBlocks;

    ggml_cuda_pool_alloc<half>     kbar(ctx.pool());
    ggml_cuda_pool_alloc<half>     vbar(ctx.pool());
    ggml_cuda_pool_alloc<uint32_t> keep(ctx.pool());
    p.keep       = nullptr;
    p.keep_words = 0;
    p.kbar       = nullptr;
    p.vbar       = nullptr;
    p.nblk       = nblk;
    if (sparse) {
        const size_t means = (size_t) K->ne[2] * nblk * kHead;
        p.keep_words = (nblk + 31) / 32;
        p.kbar       = kbar.alloc(means);
        p.vbar       = vbar.alloc(means);
        uint32_t * keep_ptr = keep.alloc((size_t) grid.x * grid.y * p.keep_words);
        fa256_block_means<<<dim3(nblk, (unsigned) K->ne[2], 1), 256, 0, stream>>>(p, kbar.get(), vbar.get());
        fa256_select<<<grid, kThreads, 0, stream>>>(p, keep_ptr, sparse_alpha);
        p.keep = keep_ptr;

        // GGML_ONEBIT_FLASH_PREFILL_STATS=1: print pruned blocks per call (synchronizes; diagnostics).
        static const bool stats = [] {
            const char * env = getenv("GGML_ONEBIT_FLASH_PREFILL_STATS");
            return env != nullptr && atoi(env) != 0;
        }();
        if (stats) {
            const size_t n = (size_t) grid.x * grid.y * p.keep_words;
            std::vector<uint32_t> h(n);
            CUDA_CHECK(cudaMemcpyAsync(h.data(), keep_ptr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            size_t pruned = 0;
            for (size_t w = 0; w < n; ++w) {
                const int valid = (int) std::min<size_t>(32, nblk - (w % p.keep_words) * 32);
                const uint32_t m = valid >= 32 ? ~0u : ((1u << valid) - 1u);
                pruned += __builtin_popcount(~h[w] & m);
            }
            // visible (workgroup, block) pairs, causal approximation: row tile x sees blocks up to its last position
            size_t visible = 0;
            for (unsigned x = 0; x < grid.x; ++x) {
                const int last_pos = std::min(n_q, (int) ((x + 1) * kRows + gqa - 1) / gqa) - 1;
                visible += (size_t) grid.y * std::min(nblk, (n_kv - n_q + last_pos) / kBlock + 1);
            }
            fprintf(stderr, "fa256 sparse: n_kv %d n_q %d pruned %zu of ~%zu visible blocks (%.1f%% kept)\n", n_kv, n_q,
                    pruned, visible, 100.0 * (1.0 - (double) pruned / (double) (visible ? visible : 1)));
        }
    }

    ggml_cuda_pool_alloc<float2> ml(ctx.pool());
    if (sparse) {
        ml.alloc((size_t) K->ne[2] * n_q * gqa);
    }
    fa256_prefill<false><<<grid, kThreads, 0, stream>>>(p, ml.ptr);
    // GGML_ONEBIT_FLASH_PREFILL_NOCORR=1 drops the mean correction (diagnostics).
    static const bool no_corr = [] {
        const char * env = getenv("GGML_ONEBIT_FLASH_PREFILL_NOCORR");
        return env != nullptr && atoi(env) != 0;
    }();
    if (sparse && !no_corr) {
        fa256_prefill<true><<<grid, kThreads, 0, stream>>>(p, ml.ptr);
    }
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_fattn_onebit_d256_eligible(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_fattn_onebit_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn-onebit-d256 is HIP only");
}

#endif // defined(GGML_USE_HIP)
