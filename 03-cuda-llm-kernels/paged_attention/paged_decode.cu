// 03-cuda-llm-kernels/paged_attention/paged_decode.cu
#include "paged_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

template <int HEAD_DIM>
__global__ void paged_attention_v1_kernel(
    half* __restrict__ out,
    const half* __restrict__ q,
    const half* __restrict__ k_pool,
    const half* __restrict__ v_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    int max_blocks_per_seq,
    int num_heads,
    float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) {
        const int tid = threadIdx.x;
        half* out_ptr = out + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
        if (tid * 8 < HEAD_DIM) {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(out_ptr + tid * 8) = zero;
        }
        return;
    }

    constexpr int NUM_WARPS = 4;
    constexpr int TOKENS_PER_WARP_REDUCTION = BLOCK_SIZE / NUM_WARPS;

    // Shared memory allocations
    alignas(16) __shared__ half s_q[HEAD_DIM];
    __shared__ float s_logits[BLOCK_SIZE];
    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_block_max;
    __shared__ float s_block_sum;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // 1. Cooperative load of Q vector into shared memory using 128-bit float4
    const half* q_ptr = q + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
    if (tid * 8 < HEAD_DIM) {
        *reinterpret_cast<float4*>(&s_q[tid * 8]) =
            *reinterpret_cast<const float4*>(q_ptr + tid * 8);
    }
    __syncthreads();

    // Running online softmax state
    float m_running = -INFINITY;
    float l_running = 0.0f;
    float o_running[8] = {0.0f};

    // Number of logical blocks for this sequence
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Work partitioning: 4 warps process blocks of tokens cooperatively
    constexpr int THREADS_PER_TOKEN = HEAD_DIM / 8;
    constexpr int TOKENS_PER_WARP = WARP_SIZE / THREADS_PER_TOKEN;
    constexpr int TOKENS_PER_STEP = NUM_WARPS * TOKENS_PER_WARP;
    constexpr int NUM_STEPS = (BLOCK_SIZE + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP;

    const int group_id = lane_id / THREADS_PER_TOKEN;
    const int group_lane = lane_id % THREADS_PER_TOKEN;

    // Loop over logical blocks
    for (int b = 0; b < num_blocks; ++b) {
        const int p_block = block_tables[seq_idx * max_blocks_per_seq + b];
        if (p_block < 0) {
            break;
        }

        const half* k_block = k_pool + (static_cast<size_t>(p_block) * num_heads + head_idx) * (BLOCK_SIZE * HEAD_DIM);
        const half* v_block = v_pool + (static_cast<size_t>(p_block) * num_heads + head_idx) * (BLOCK_SIZE * HEAD_DIM);

        // Phase 1: Compute Q * K_t * scale for each token in the block
        #pragma unroll
        for (int step = 0; step < NUM_STEPS; ++step) {
            int token_in_block = (warp_id * TOKENS_PER_WARP + group_id) + step * TOKENS_PER_STEP;
            int token_idx = b * BLOCK_SIZE + token_in_block;

            float dot = 0.0f;
            if (token_in_block < BLOCK_SIZE && token_idx < context_len) {
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

            // Intra-group reduction using warp shuffles without thread divergence
            #pragma unroll
            for (int offset = THREADS_PER_TOKEN / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }

            if (group_lane == 0 && token_in_block < BLOCK_SIZE) {
                s_logits[token_in_block] = (token_idx < context_len) ? (dot * scale) : -INFINITY;
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

        // Step 2b: Exponentiate logits into probabilities
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

    // Final normalization: O = o_running / l_running, written via float4
    if (tid < HEAD_DIM / 8) {
        float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
        alignas(16) half out_h[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_h[j] = __float2half(o_running[j] * inv_l);
        }
        half* out_ptr = out + (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
        *reinterpret_cast<float4*>(out_ptr + tid * 8) = *reinterpret_cast<const float4*>(out_h);
    }
}

extern "C" void paged_attention_v1_launcher(
    half* out,
    const half* q,
    const half* k_pool,
    const half* v_pool,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    if (batch_size <= 0 || num_heads <= 0) {
        return;
    }

    dim3 grid(batch_size, num_heads);
    dim3 block(128);

    if (head_dim == 64) {
        paged_attention_v1_kernel<64><<<grid, block, 0, stream>>>(
            out,
            q,
            k_pool,
            v_pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_heads,
            scale
        );
    } else if (head_dim == 128) {
        paged_attention_v1_kernel<128><<<grid, block, 0, stream>>>(
            out,
            q,
            k_pool,
            v_pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_heads,
            scale
        );
    } else if (head_dim == 256) {
        paged_attention_v1_kernel<256><<<grid, block, 0, stream>>>(
            out,
            q,
            k_pool,
            v_pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_heads,
            scale
        );
    }
}
