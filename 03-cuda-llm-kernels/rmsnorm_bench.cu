#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// DeepSeek-R1-Distill-Qwen-14B exact architecture parameters
constexpr int HIDDEN_DIM = 5120;
constexpr float EPS = 1e-5f;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                  \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// Level 0: CPU reference for correctness verification
// -----------------------------------------------------------------------------
void rmsnorm_cpu(const float *x, const float *weight, float *out, int rows,
                 int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float *rx = x + r * cols;
    float *rout = out + r * cols;
    float sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum_sq += rx[c] * rx[c];
    }
    float rsqrt_val = 1.0f / std::sqrt((sum_sq / cols) + eps);
    for (int c = 0; c < cols; ++c) {
      rout[c] = rx[c] * rsqrt_val * weight[c];
    }
  }
}

// -----------------------------------------------------------------------------
// Level 1: Naive CUDA Kernel
// One block per row. Shared memory reduction with strided loop and syncthreads.
// -----------------------------------------------------------------------------
__global__ void rmsnorm_kernel_naive(const float *__restrict__ x,
                                     const float *__restrict__ weight,
                                     float *__restrict__ out, int cols,
                                     float eps) {
  int row = blockIdx.x;
  const float *rx = x + row * cols;
  float *rout = out + row * cols;

  __shared__ float s_sum[512];
  float thread_sum = 0.0f;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float val = rx[i];
    thread_sum += val * val;
  }
  s_sum[threadIdx.x] = thread_sum;
  __syncthreads();

  // Simple power-of-2 shared memory reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float rrms = rsqrtf((s_sum[0] / (float)cols) + eps);

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    rout[i] = rx[i] * rrms * weight[i];
  }
}

// -----------------------------------------------------------------------------
// Level 2: Optimized Kernel with Warp-Level Shuffle Reduction
// Uses __shfl_down_sync to avoid shared memory bank conflicts & syncthreads.
// -----------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void rmsnorm_kernel_warp_shuffle(const float *__restrict__ x,
                                            const float *__restrict__ weight,
                                            float *__restrict__ out, int cols,
                                            float eps) {
  int row = blockIdx.x;
  const float *rx = x + row * cols;
  float *rout = out + row * cols;

  __shared__ float s_warp_sums[32]; // Max 32 warps per block (1024 threads)
  __shared__ float s_rrms;

  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = rx[i];
    thread_sum += v * v;
  }

  // 1. Reduce within warp
  thread_sum = warp_reduce_sum(thread_sum);

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  // Warp leaders write to shared memory
  if (lane_id == 0) {
    s_warp_sums[warp_id] = thread_sum;
  }
  __syncthreads();

  // First warp reduces the warp sums
  if (warp_id == 0) {
    int num_warps = blockDim.x / 32;
    float w_val = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
    w_val = warp_reduce_sum(w_val);
    if (lane_id == 0) {
      s_rrms = rsqrtf((w_val / (float)cols) + eps);
    }
  }
  __syncthreads();

  float rrms = s_rrms;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    rout[i] = rx[i] * rrms * weight[i];
  }
}

// -----------------------------------------------------------------------------
// Level 3: Vectorized Fused Kernel (float4) + Warp Shuffle
// Loads 128 bits (4 floats) per instruction. Maximizes memory bus bandwidth.
// -----------------------------------------------------------------------------
__global__ void rmsnorm_kernel_vectorized(const float *__restrict__ x,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out, int cols,
                                         float eps) {
  // cols must be divisible by 4 (5120 is divisible by 4: 1280 float4)
  int row = blockIdx.x;
  const float4 *rx4 = reinterpret_cast<const float4 *>(x + row * cols);
  const float4 *rw4 = reinterpret_cast<const float4 *>(weight);
  float4 *rout4 = reinterpret_cast<float4 *>(out + row * cols);
  int cols4 = cols / 4;

  __shared__ float s_warp_sums[32];
  __shared__ float s_rrms;

  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    float4 v = rx4[i];
    thread_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  thread_sum = warp_reduce_sum(thread_sum);

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  if (lane_id == 0) {
    s_warp_sums[warp_id] = thread_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    int num_warps = blockDim.x / 32;
    float w_val = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
    w_val = warp_reduce_sum(w_val);
    if (lane_id == 0) {
      s_rrms = rsqrtf((w_val / (float)cols) + eps);
    }
  }
  __syncthreads();

  float rrms = s_rrms;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    float4 vx = rx4[i];
    float4 vw = rw4[i];
    float4 res;
    res.x = vx.x * rrms * vw.x;
    res.y = vx.y * rrms * vw.y;
    res.z = vx.z * rrms * vw.z;
    res.w = vx.w * rrms * vw.w;
    rout4[i] = res;
  }
}

// -----------------------------------------------------------------------------
// Benchmark Runner
// -----------------------------------------------------------------------------
int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "======================================================\n";
  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "Target Model: DeepSeek-R1-Distill-Qwen-14B\n";
  std::cout << "Hidden Dim: " << HIDDEN_DIM << "\n";
  std::cout << "======================================================\n\n";

  // Test across two realistic LLM workloads:
  // 1. Decode step: batch=1, seq=1 (Latency critical: single token generation)
  // 2. Prefill step: batch=1, seq=2048 (Throughput critical: prompt processing)
  std::vector<int> token_counts = {1, 16, 2048};

  for (int num_tokens : token_counts) {
    size_t x_bytes = num_tokens * HIDDEN_DIM * sizeof(float);
    size_t w_bytes = HIDDEN_DIM * sizeof(float);

    std::vector<float> h_x(num_tokens * HIDDEN_DIM);
    std::vector<float> h_weight(HIDDEN_DIM);
    std::vector<float> h_out_ref(num_tokens * HIDDEN_DIM);
    std::vector<float> h_out_gpu(num_tokens * HIDDEN_DIM);

    // Initialize with synthetic data
    for (size_t i = 0; i < h_x.size(); ++i) {
      h_x[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < HIDDEN_DIM; ++i) {
      h_weight[i] = 1.0f + 0.1f * (static_cast<float>(rand()) / RAND_MAX);
    }

    // Reference CPU run
    rmsnorm_cpu(h_x.data(), h_weight.data(), h_out_ref.data(), num_tokens,
                HIDDEN_DIM, EPS);

    float *d_x, *d_weight, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, x_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), w_bytes,
                          cudaMemcpyHostToDevice));

    // Warmup & timing setup
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    const int ITERS = (num_tokens == 1) ? 2000 : 500;

    std::cout << "--- Workload: " << num_tokens << " Token(s) (Tensor: ["
              << num_tokens << ", " << HIDDEN_DIM << "]) ---\n";

    auto benchmark_kernel = [&](const char *name, auto kernel, dim3 grid,
                                dim3 block) {
      // Warmup & verify correctness
      kernel<<<grid, block>>>(d_x, d_weight, d_out, HIDDEN_DIM, EPS);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, x_bytes,
                            cudaMemcpyDeviceToHost));

      float max_err = 0.0f;
      for (size_t i = 0; i < h_out_gpu.size(); ++i) {
        max_err = std::max(max_err, std::abs(h_out_gpu[i] - h_out_ref[i]));
      }

      // Timing
      CHECK_CUDA(cudaEventRecord(start));
      for (int it = 0; it < ITERS; ++it) {
        kernel<<<grid, block>>>(d_x, d_weight, d_out, HIDDEN_DIM, EPS);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      float avg_us = (total_ms / ITERS) * 1000.0f;

      // Memory traffic: read X (bytes), read weight (bytes), write Out (bytes)
      // For RMSNorm, theoretical bytes transferred per call:
      double data_bytes = (2.0 * x_bytes) + w_bytes;
      double gb_per_sec = (data_bytes / (avg_us * 1e-6)) / 1e9;

      std::cout << "  " << name << ":\n"
                << "    Avg Latency: " << avg_us << " us\n"
                << "    Bandwidth:   " << gb_per_sec << " GB/s\n"
                << "    Max Error:   " << max_err << (max_err < 1e-4 ? " (PASS)" : " (FAIL)") << "\n";
    };

    dim3 grid(num_tokens);
    benchmark_kernel("Level 1: Naive (Shared Mem)", rmsnorm_kernel_naive, grid,
                     dim3(256));
    benchmark_kernel("Level 2: Warp Shuffle", rmsnorm_kernel_warp_shuffle,
                     grid, dim3(256));
    benchmark_kernel("Level 3: Vectorized float4", rmsnorm_kernel_vectorized,
                     grid, dim3(256));
    std::cout << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
  }

  return 0;
}
