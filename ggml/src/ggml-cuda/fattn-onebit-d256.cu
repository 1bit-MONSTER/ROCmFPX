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

#include "fattn-onebit-d256.cuh"

#include <climits>
#include <cmath>
#include <cstdlib>

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
};

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

__global__ void __launch_bounds__(kThreads) fa256_prefill(const fa256_params p) {
#if defined(__GFX11__)
    __shared__ __attribute__((aligned(16))) _Float16 k_lds[kTile * kKStride];
    __shared__ __attribute__((aligned(16))) _Float16 vt_lds[kHead * kVtStride];
    __shared__ __attribute__((aligned(16))) _Float16 p_lds[kRows * kVtStride];
    __shared__ int s_lo;
    __shared__ int s_hi;
    __shared__ int s_full_lo;
    __shared__ int s_full_hi;
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

    // Visited KV range = union of the positions' ranges. Inside the intersection,
    // when every row is dense (all-zero mask), tiles skip the mask entirely.
    if (tid == 0) {
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
        s_lo      = lo;
        s_hi      = hi;
        s_full_lo = dense ? full_lo : INT_MAX;
        s_full_hi = dense ? full_hi : 0;
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

    const int tile_lo = s_lo < s_hi ? s_lo / kTile : 0;
    const int tile_hi = s_lo < s_hi ? (s_hi + kTile - 1) / kTile : 0;
    const int full_lo = s_full_lo;
    const int full_hi = s_full_hi;

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

    // One tile's K and V as kLoads 16-byte chunks per thread. K chunks run along
    // the channels (coalesced, stored row-major); V chunks put one token per lane
    // so the transposed store below writes consecutive tokens of a channel row.
    uint4 kx[kLoads];
    uint4 vx[kLoads];
    const int v_tok = lane;                    // token within the tile
    auto load_tile = [&](int kv0) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int e  = tid + u * kThreads;
            const int r  = e / (kHead / 8);
            const int c  = (e % (kHead / 8)) * 8;
            const int kv = kv0 + r;
            kx[u] = make_uint4(0, 0, 0, 0);
            if (kv < p.n_kv) {
                kx[u] = *(const uint4 *) (p.k + kv * p.k_nb1 + kvh * p.k_nb2 + c * sizeof(half));
            }
            const int cv  = (wave + kWaves * u) * 8;
            const int kvv = kv0 + v_tok;
            vx[u] = make_uint4(0, 0, 0, 0);
            if (kvv < p.n_kv) {
                vx[u] = *(const uint4 *) (p.v + kvv * p.v_nb1 + kvh * p.v_nb2 + cv * sizeof(half));
            }
        }
    };

    float          * xch    = (float *) k_lds;  // partial-score exchange, after Q*K is done
    _Float16       * p_pair = p_lds + pair * 16 * kVtStride;
    const _Float16 * k_part = k_lds + part * kHalf;

    for (int t = tile_lo; t < tile_hi; ++t) {
        const int kv0 = t * kTile;

        load_tile(kv0);
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

        const int  kva       = kv0 + col;
        const int  kvb       = kv0 + 16 + col;
        const bool need_mask = kv0 < full_lo || kv0 + kTile > full_hi;  // workgroup-uniform
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float xa = -INFINITY;
            float xb = -INFINITY;
            if (mask_off[i] >= 0) {
                if (kva < p.n_kv) {
                    xa = s0[i] * p.scale_log2;
                    if (need_mask) {
                        xa += __half2float(mask[mask_off[i] + kva]) * kLog2e;
                    }
                }
                if (kvb < p.n_kv) {
                    xb = s1[i] * p.scale_log2;
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
#pragma unroll
        for (int j = 0; j < kHalf / 16; ++j) {
            out[j * 16 + col] = acc[j][i] * inv;
        }
    }
#else
    GGML_UNUSED(p);
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
    fa256_prefill<<<grid, kThreads, 0, stream>>>(p);
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
