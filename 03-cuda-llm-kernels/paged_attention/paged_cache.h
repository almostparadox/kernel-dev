// 03-cuda-llm-kernels/paged_attention/paged_cache.h
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

#ifdef __cplusplus
extern "C" {
#endif

void paged_attention_v1_launcher(
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
);

#ifdef __cplusplus
}
#endif
