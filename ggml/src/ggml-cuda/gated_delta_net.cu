#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int64_t state_size_per_token = S_v * S_v * H * n_seqs; // per-slot stride in output
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = (dst + attn_score_elems) + target_slot * state_size_per_token + state_out_offset;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Reduce-and-broadcast over the GDN_PF_SPLIT adjacent lanes of one state column. On RDNA the
// butterfly is DPP (quad_perm XOR 1 / XOR 2, row_half_mirror for 8 lanes), a VALU modifier; a
// shuffle lowers to ds_bpermute on the LDS pipe, where it contends with the k/q reads and its
// wait retires them onto the token recurrence.
template <int WIDTH>
static __device__ __forceinline__ float gdn_pf_group_sum(float x) {
#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
    static_assert(WIDTH == 4 || WIDTH == 8, "DPP reduction covers 4 and 8 lanes");
    if constexpr (WIDTH == 8) {
        x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0x141, 0xF, 0xF, true));
    }
    x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0xB1, 0xF, 0xF, true));
    x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0x4E, 0xF, 0xF, true));
    return x;
#else
    return warp_reduce_sum<WIDTH>(x);
#endif
}

// Prefill variant (scalar gate). GDN_PF_SPLIT lanes share each state column's rows; each lane
// serves GDN_PF_CPL adjacent columns, so one read of a token's k/q row from shared memory feeds
// GDN_PF_CPL columns (the per column-token shared-memory traffic is what bounds this kernel). A
// lane's rows are interleaved in groups of 4 (GDN_PF_ROW) so the lanes of a column read different
// banks. The per-token k/q/v/g/beta are staged in shared memory GDN_PF_CHUNK tokens at a time with
// 16-byte loads (the launcher checks alignment). With keep_rs_t it also writes the state after
// each of the last K tokens (slot 0 = the final state, slot s = s tokens back), the slots
// speculative rollback reads.
//
// Found by a KernelForge campaign on gfx1151 (tools/kernelforge in the engine): 7.4x on a
// Qwen3.8-27B layer's 512-token micro-batch (3.02 -> 0.41 ms), 5.0x with 8 snapshots.
#define GDN_PF_ROW(r) ((((r) >> 2) * (GDN_PF_SPLIT * 4)) + part * 4 + ((r) & 3))

template <int S_v, int GDN_PF_SPLIT, int GDN_PF_CPL, int GDN_PF_COLS, int GDN_PF_CHUNK, bool keep_rs_t>
__global__ void __launch_bounds__((GDN_PF_COLS / GDN_PF_CPL) * GDN_PF_SPLIT)
gated_delta_net_prefill_cuda(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       n_seqs,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale,
                             int           K) {
    constexpr int ROWS     = S_v / GDN_PF_SPLIT;
    constexpr int NTHREADS = (GDN_PF_COLS / GDN_PF_CPL) * GDN_PF_SPLIT;
    static_assert(ROWS % 4 == 0, "the row interleave needs ROWS a multiple of 4");
    static_assert(GDN_PF_CPL == 2, "the output store pairs two columns");
    static_assert(NTHREADS >= GDN_PF_CHUNK, "g/beta staging uses one thread per chunk token");

    __shared__ float  k_s[GDN_PF_CHUNK][S_v];
    __shared__ float  q_s[GDN_PF_CHUNK][S_v];
    __shared__ float  v_s[GDN_PF_CHUNK][GDN_PF_COLS];
    __shared__ float2 gb_s[GDN_PF_CHUNK];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid      = threadIdx.x;
    const int      part     = tid % GDN_PF_SPLIT;
    const int      cloc     = (tid / GDN_PF_SPLIT) * GDN_PF_CPL;
    const int      col0     = blockIdx.z * GDN_PF_COLS;
    const int      col      = col0 + cloc;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst + (sequence * n_tokens * H + h_idx) * S_v;
    float *       state            = dst + attn_score_elems + (sequence * H + h_idx) * S_v * S_v;
    curr_state += sequence * H * S_v * S_v + h_idx * S_v * S_v;

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1 + col0;
    const float * g_base = g    + sequence * sb3 + h_idx * sb1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;

    ggml_cuda_pdl_sync();

    float s[GDN_PF_CPL][ROWS];
#pragma unroll
    for (int cc = 0; cc < GDN_PF_CPL; cc++) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            s[cc][r] = curr_state[(col + cc) * S_v + GDN_PF_ROW(r)];
        }
    }

    for (int64_t t0 = 0; t0 < n_tokens; t0 += GDN_PF_CHUNK) {
        const int nt = (int) min((int64_t) GDN_PF_CHUNK, n_tokens - t0);

        __syncthreads();
        for (int idx = tid * 4; idx < nt * S_v; idx += NTHREADS * 4) {
            const int tt = idx / S_v;
            const int i  = idx % S_v;
            *reinterpret_cast<float4 *>(&k_s[tt][i]) = *reinterpret_cast<const float4 *>(k_base + (t0 + tt) * sq2 + i);
            *reinterpret_cast<float4 *>(&q_s[tt][i]) = *reinterpret_cast<const float4 *>(q_base + (t0 + tt) * sq2 + i);
        }
        for (int idx = tid * 4; idx < nt * GDN_PF_COLS; idx += NTHREADS * 4) {
            const int tt = idx / GDN_PF_COLS;
            const int c  = idx % GDN_PF_COLS;
            *reinterpret_cast<float4 *>(&v_s[tt][c]) = *reinterpret_cast<const float4 *>(v_base + (t0 + tt) * sv2 + c);
        }
        if (tid < nt) {
            gb_s[tid] = make_float2(expf(g_base[(t0 + tid) * sb2]), b_base[(t0 + tid) * sb2]);
        }
        __syncthreads();

        for (int tt = 0; tt < nt; tt++) {
            const float2 gb = gb_s[tt];

            float kr[ROWS];
            float kv[GDN_PF_CPL][4] = {};
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                kr[r] = k_s[tt][GDN_PF_ROW(r)];
#pragma unroll
                for (int cc = 0; cc < GDN_PF_CPL; cc++) {
                    kv[cc][r % 4] += s[cc][r] * kr[r];
                }
            }
            float delta[GDN_PF_CPL];
#pragma unroll
            for (int cc = 0; cc < GDN_PF_CPL; cc++) {
                const float kv_col = gdn_pf_group_sum<GDN_PF_SPLIT>((kv[cc][0] + kv[cc][1]) + (kv[cc][2] + kv[cc][3]));
                delta[cc] = (v_s[tt][cloc + cc] - gb.x * kv_col) * gb.y;
            }

            float at[GDN_PF_CPL][4] = {};
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const float qr = q_s[tt][GDN_PF_ROW(r)];
#pragma unroll
                for (int cc = 0; cc < GDN_PF_CPL; cc++) {
                    s[cc][r]       = gb.x * s[cc][r] + kr[r] * delta[cc];
                    at[cc][r % 4] += s[cc][r] * qr;
                }
            }
            float2 o;
            o.x = gdn_pf_group_sum<GDN_PF_SPLIT>((at[0][0] + at[0][1]) + (at[0][2] + at[0][3])) * scale;
            o.y = gdn_pf_group_sum<GDN_PF_SPLIT>((at[1][0] + at[1][1]) + (at[1][2] + at[1][3])) * scale;
            if (part == 0) {
                *reinterpret_cast<float2 *>(&attn_data[(t0 + tt) * S_v * H + col]) = o;
            }
            if constexpr (keep_rs_t) {
                const int64_t slot = n_tokens - 1 - (t0 + tt);
                if (slot < K) {
                    float * snap = state + slot * (S_v * S_v * H * n_seqs);
#pragma unroll
                    for (int cc = 0; cc < GDN_PF_CPL; cc++) {
#pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            snap[(col + cc) * S_v + GDN_PF_ROW(r)] = s[cc][r];
                        }
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int cc = 0; cc < GDN_PF_CPL; cc++) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                state[(col + cc) * S_v + GDN_PF_ROW(r)] = s[cc][r];
            }
        }
    }
}
#undef GDN_PF_ROW

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    if constexpr (!KDA) {
        // the prefill kernel stages k/q/v with 16-byte loads and writes pairs of outputs
        const bool aligned16 = ((uintptr_t) q_d | (uintptr_t) k_d | (uintptr_t) v_d | (uintptr_t) dst_d) % 16 == 0 &&
                               sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0;
        if (S_v == 128 && n_tokens >= 16 && aligned16) {
            // 8 lanes per column group, 2 columns per lane, 64 columns per block, 16-token chunks:
            // the KernelForge campaign's layout (tools/kernelforge in the engine)
            constexpr int SPLIT = 8, CPL = 2, COLS = 64, CHUNK = 16;
            const uint3 neqk1_magic = init_fastdiv_values(neqk1);
            const uint3 rq3_magic   = init_fastdiv_values(rq3);
            dim3 grid_dims(H, n_seqs, 128 / COLS);
            dim3 block_dims((COLS / CPL) * SPLIT, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_prefill_cuda<128, SPLIT, CPL, COLS, CHUNK, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            return;
        }
    }
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        }
    }
}
