// 03-cuda-llm-kernels/paged_attention/paged_split_kv.cu
#include "paged_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

template <int HEAD_DIM>
__global__ void paged_attention_splitkv_stage1_kernel(
    float* __restrict__ tmp_out,
    float* __restrict__ tmp_metadata,
    const half* __restrict__ q,
    const half* __restrict__ k_pool,
    const half* __restrict__ v_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    int max_blocks_per_seq,
    int num_heads,
    int num_splits,
    float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;

    const size_t split_offset =
        ((static_cast<size_t>(seq_idx) * num_heads + head_idx) * num_splits + split_idx) * HEAD_DIM;
    const size_t meta_offset =
        ((static_cast<size_t>(seq_idx) * num_heads + head_idx) * num_splits + split_idx) * 2;

    float* my_tmp_out = tmp_out + split_offset;
    float* my_tmp_meta = tmp_metadata + meta_offset;

    const int context_len = context_lens[seq_idx];
    const int start_token = (context_len > 0) ? ((split_idx * context_len) / num_splits) : 0;
    const int end_token = (context_len > 0) ? (((split_idx + 1) * context_len) / num_splits) : 0;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Fast-path guard: sequence has no tokens or split interval is empty
    if (context_len <= 0 || start_token >= end_token) {
        if (tid * 4 < HEAD_DIM) {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(my_tmp_out + tid * 4) = zero;
        }
        if (tid == 0) {
            my_tmp_meta[0] = -INFINITY;
            my_tmp_meta[1] = 0.0f;
        }
        return;
    }

    constexpr int NUM_WARPS = 4;
    constexpr int TOKENS_PER_WARP_REDUCTION = BLOCK_SIZE / NUM_WARPS;

    // Shared memory scratchpads
    alignas(16) __shared__ half s_q[HEAD_DIM];
    __shared__ float s_logits[BLOCK_SIZE];
    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_block_max;
    __shared__ float s_block_sum;

    // 1. Cooperative load of Query vector using 128-bit float4 (8 halves per thread)
    const half* q_ptr = q + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
    if (tid * 8 < HEAD_DIM) {
        *reinterpret_cast<float4*>(&s_q[tid * 8]) =
            *reinterpret_cast<const float4*>(q_ptr + tid * 8);
    }
    __syncthreads();

    // Running online softmax statistics for this split
    float m_running = -INFINITY;
    float l_running = 0.0f;
    float o_running[8] = {0.0f};

    // Logical block range belonging to interval [start_token, end_token)
    const int start_block = start_token / BLOCK_SIZE;
    const int end_block = (end_token + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Work partitioning: 4 warps cooperatively process tokens in each logical block
    constexpr int THREADS_PER_TOKEN = HEAD_DIM / 8;
    constexpr int TOKENS_PER_WARP = WARP_SIZE / THREADS_PER_TOKEN;
    constexpr int TOKENS_PER_STEP = NUM_WARPS * TOKENS_PER_WARP;
    constexpr int NUM_STEPS = (BLOCK_SIZE + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP;

    const int group_id = lane_id / THREADS_PER_TOKEN;
    const int group_lane = lane_id % THREADS_PER_TOKEN;

    // Loop over assigned logical blocks
    for (int b = start_block; b < end_block; ++b) {
        const int p_block = block_tables[seq_idx * max_blocks_per_seq + b];
        if (p_block < 0) {
            break;
        }

        const half* k_block =
            k_pool + (static_cast<size_t>(p_block) * num_heads + head_idx) * (BLOCK_SIZE * HEAD_DIM);
        const half* v_block =
            v_pool + (static_cast<size_t>(p_block) * num_heads + head_idx) * (BLOCK_SIZE * HEAD_DIM);

        // Phase 1: Compute Q * K_t * scale for each token in block
        #pragma unroll
        for (int step = 0; step < NUM_STEPS; ++step) {
            int token_in_block = (warp_id * TOKENS_PER_WARP + group_id) + step * TOKENS_PER_STEP;
            int token_idx = b * BLOCK_SIZE + token_in_block;
            bool valid = (token_in_block < BLOCK_SIZE && token_idx >= start_token && token_idx < end_token);

            float dot = 0.0f;
            if (valid) {
                // Vectorized 128-bit load of K chunk (8 halves per thread)
                const float4* k_ptr_f4 = reinterpret_cast<const float4*>(
                    k_block + token_in_block * HEAD_DIM + group_lane * 8
                );
                float4 k_val = *k_ptr_f4;
                float4 q_val = *reinterpret_cast<const float4*>(&s_q[group_lane * 8]);

                const half* k_h = reinterpret_cast<const half*>(&k_val);
                const half* q_h = reinterpret_cast<const half*>(&q_val);

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dot += __half2float(q_h[j]) * __half2float(k_h[j]);
                }
            }

            // Intra-token reduction using warp shuffle without divergence
            #pragma unroll
            for (int offset = THREADS_PER_TOKEN / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }

            if (group_lane == 0 && token_in_block < BLOCK_SIZE) {
                s_logits[token_in_block] = valid ? (dot * scale) : -INFINITY;
            }
        }
        __syncthreads();

        // Phase 2: Online Softmax Normalization
        // Step 2a: Inter-warp max reduction
        if (lane_id == 0) {
            float w_max = -INFINITY;
            #pragma unroll
            for (int k = 0; k < TOKENS_PER_WARP_REDUCTION; ++k) {
                w_max = fmaxf(w_max, s_logits[warp_id * TOKENS_PER_WARP_REDUCTION + k]);
            }
            s_warp_max[warp_id] = w_max;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp_max[lane_id] : -INFINITY;
            val = warp_reduce_max(val);
            if (lane_id == 0) {
                s_block_max = val;
            }
        }
        __syncthreads();

        const float block_max = s_block_max;
        const float new_max = fmaxf(m_running, block_max);
        const float alpha = (m_running == -INFINITY) ? 0.0f : expf(m_running - new_max);
        m_running = new_max;

        // Step 2b: Exponentiate logits into unnormalized probabilities
        if (tid < BLOCK_SIZE) {
            float logit = s_logits[tid];
            s_logits[tid] = (logit == -INFINITY) ? 0.0f : expf(logit - new_max);
        }
        __syncthreads();

        // Step 2c: Inter-warp sum reduction for normalizer
        if (lane_id == 0) {
            float w_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < TOKENS_PER_WARP_REDUCTION; ++k) {
                w_sum += s_logits[warp_id * TOKENS_PER_WARP_REDUCTION + k];
            }
            s_warp_sum[warp_id] = w_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp_sum[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane_id == 0) {
                s_block_sum = val;
            }
        }
        __syncthreads();

        const float block_sum = s_block_sum;
        l_running = alpha * l_running + block_sum;

        // Step 2d: Accumulate Value vectors into output registers
        if (tid < HEAD_DIM / 8) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                o_running[j] *= alpha;
            }

            // Vectorized 128-bit load of V (8 halves per thread)
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE; ++t) {
                float p = s_logits[t];
                if (p > 0.0f) {
                    const float4* v_ptr_f4 = reinterpret_cast<const float4*>(
                        v_block + t * HEAD_DIM + tid * 8
                    );
                    float4 v_val = *v_ptr_f4;
                    const half* v_h = reinterpret_cast<const half*>(&v_val);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        o_running[j] += p * __half2float(v_h[j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write Stage 1 unnormalized accumulator o_running to tmp_out using 128-bit float4
    if (tid < HEAD_DIM / 8) {
        float4 v0 = make_float4(o_running[0], o_running[1], o_running[2], o_running[3]);
        float4 v1 = make_float4(o_running[4], o_running[5], o_running[6], o_running[7]);
        float* out_f = my_tmp_out + tid * 8;
        *reinterpret_cast<float4*>(out_f) = v0;
        *reinterpret_cast<float4*>(out_f + 4) = v1;
    }

    // Write local maximum m_k and local normalizer l_k to tmp_metadata
    if (tid == 0) {
        my_tmp_meta[0] = m_running;
        my_tmp_meta[1] = l_running;
    }
}

template <int HEAD_DIM>
__global__ void paged_attention_splitkv_stage2_kernel(
    half* __restrict__ out,
    const float* __restrict__ tmp_out,
    const float* __restrict__ tmp_metadata,
    const int32_t* __restrict__ context_lens,
    int num_heads,
    int num_splits
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int MAX_SPLITS = 128;
    __shared__ float s_m[MAX_SPLITS];
    __shared__ float s_l[MAX_SPLITS];
    __shared__ float s_beta[MAX_SPLITS];
    __shared__ float s_m_global;
    __shared__ float s_l_global;

    // Fast-path guard: sequence with no tokens
    if (context_lens != nullptr && context_lens[seq_idx] <= 0) {
        half* out_ptr = out + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
        if (tid * 8 < HEAD_DIM) {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(out_ptr + tid * 8) = zero;
        }
        return;
    }

    // Step 1: Cooperatively load split metadata into shared memory
    for (int k = tid; k < num_splits; k += blockDim.x) {
        if (k < MAX_SPLITS) {
            size_t meta_offset =
                ((static_cast<size_t>(seq_idx) * num_heads + head_idx) * num_splits + k) * 2;
            s_m[k] = tmp_metadata[meta_offset + 0];
            s_l[k] = tmp_metadata[meta_offset + 1];
        }
    }
    __syncthreads();

    // Step 2: Global maximum reduction m_global = max_k m_k across warps
    float local_max = -INFINITY;
    for (int k = tid; k < num_splits; k += blockDim.x) {
        if (k < MAX_SPLITS) {
            local_max = fmaxf(local_max, s_m[k]);
        }
    }
    local_max = warp_reduce_max(local_max);
    if (tid == 0) {
        s_m_global = local_max;
    }
    __syncthreads();

    const float m_global = s_m_global;
    if (m_global == -INFINITY) {
        half* out_ptr = out + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
        if (tid * 8 < HEAD_DIM) {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(out_ptr + tid * 8) = zero;
        }
        return;
    }

    // Step 3: Compute rescale factors beta_k = exp(m_k - m_global) and sum l_global = sum_k beta_k * l_k
    float thread_l_sum = 0.0f;
    for (int k = tid; k < num_splits; k += blockDim.x) {
        if (k < MAX_SPLITS) {
            float m_k = s_m[k];
            float beta_k = (m_k == -INFINITY) ? 0.0f : expf(m_k - m_global);
            s_beta[k] = beta_k;
            thread_l_sum += beta_k * s_l[k];
        }
    }
    thread_l_sum = warp_reduce_sum(thread_l_sum);
    if (tid == 0) {
        s_l_global = thread_l_sum;
    }
    __syncthreads();

    // Step 4: Reduce partial vectors into final output vector: O_final = (1 / l_global) * sum_k beta_k * o_k
    if (tid < HEAD_DIM / 8) {
        const float inv_l_global = (s_l_global > 0.0f) ? (1.0f / s_l_global) : 0.0f;
        float o_accum[8] = {0.0f};

        for (int k = 0; k < num_splits; ++k) {
            if (k >= MAX_SPLITS) {
                break;
            }
            float beta_k = s_beta[k];
            if (beta_k > 0.0f) {
                size_t split_offset =
                    ((static_cast<size_t>(seq_idx) * num_heads + head_idx) * num_splits + k) * HEAD_DIM;
                const float* split_out_ptr = tmp_out + split_offset + tid * 8;

                // Vectorized 128-bit float4 loads of partial accumulator
                float4 v0 = *reinterpret_cast<const float4*>(split_out_ptr);
                float4 v1 = *reinterpret_cast<const float4*>(split_out_ptr + 4);

                o_accum[0] += beta_k * v0.x;
                o_accum[1] += beta_k * v0.y;
                o_accum[2] += beta_k * v0.z;
                o_accum[3] += beta_k * v0.w;

                o_accum[4] += beta_k * v1.x;
                o_accum[5] += beta_k * v1.y;
                o_accum[6] += beta_k * v1.z;
                o_accum[7] += beta_k * v1.w;
            }
        }

        // Final normalization and 128-bit float4 store to global memory
        alignas(16) half out_h[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_h[j] = __float2half(o_accum[j] * inv_l_global);
        }
        half* out_ptr = out + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
        *reinterpret_cast<float4*>(out_ptr + tid * 8) = *reinterpret_cast<const float4*>(out_h);
    }
}

extern "C" void paged_attention_splitkv_launcher(
    half* out,
    float* tmp_out,
    float* tmp_metadata,
    const half* q,
    const half* k_pool,
    const half* v_pool,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_splits,
    float scale,
    cudaStream_t stream
) {
    if (batch_size <= 0 || num_heads <= 0 || num_splits <= 0) {
        return;
    }

    bool alloc_out = (tmp_out == nullptr);
    bool alloc_meta = (tmp_metadata == nullptr);
    float* d_tmp_out = tmp_out;
    float* d_tmp_meta = tmp_metadata;

    if (alloc_out) {
        size_t tmp_out_bytes =
            static_cast<size_t>(batch_size) * num_heads * num_splits * head_dim * sizeof(float);
        cudaMalloc(reinterpret_cast<void**>(&d_tmp_out), tmp_out_bytes);
    }
    if (alloc_meta) {
        size_t tmp_meta_bytes =
            static_cast<size_t>(batch_size) * num_heads * num_splits * 2 * sizeof(float);
        cudaMalloc(reinterpret_cast<void**>(&d_tmp_meta), tmp_meta_bytes);
    }

    dim3 grid1(batch_size, num_heads, num_splits);
    dim3 block1(128);

    if (head_dim == 64) {
        paged_attention_splitkv_stage1_kernel<64><<<grid1, block1, 0, stream>>>(
            d_tmp_out,
            d_tmp_meta,
            q,
            k_pool,
            v_pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_heads,
            num_splits,
            scale
        );
    } else if (head_dim == 128) {
        paged_attention_splitkv_stage1_kernel<128><<<grid1, block1, 0, stream>>>(
            d_tmp_out,
            d_tmp_meta,
            q,
            k_pool,
            v_pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_heads,
            num_splits,
            scale
        );
    } else if (head_dim == 256) {
        paged_attention_splitkv_stage1_kernel<256><<<grid1, block1, 0, stream>>>(
            d_tmp_out,
            d_tmp_meta,
            q,
            k_pool,
            v_pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_heads,
            num_splits,
            scale
        );
    }

    dim3 grid2(batch_size, num_heads);
    dim3 block2(32);

    if (head_dim == 64) {
        paged_attention_splitkv_stage2_kernel<64><<<grid2, block2, 0, stream>>>(
            out,
            d_tmp_out,
            d_tmp_meta,
            context_lens,
            num_heads,
            num_splits
        );
    } else if (head_dim == 128) {
        paged_attention_splitkv_stage2_kernel<128><<<grid2, block2, 0, stream>>>(
            out,
            d_tmp_out,
            d_tmp_meta,
            context_lens,
            num_heads,
            num_splits
        );
    } else if (head_dim == 256) {
        paged_attention_splitkv_stage2_kernel<256><<<grid2, block2, 0, stream>>>(
            out,
            d_tmp_out,
            d_tmp_meta,
            context_lens,
            num_heads,
            num_splits
        );
    }

    if (alloc_out || alloc_meta) {
        cudaStreamSynchronize(stream);
    }
    if (alloc_out) {
        cudaFree(d_tmp_out);
    }
    if (alloc_meta) {
        cudaFree(d_tmp_meta);
    }
}
