#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// DeepSeek-R1-Distill-Qwen-14B FFN intermediate dimension
constexpr int INTERMEDIATE_DIM = 13824;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                  \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__device__ __inline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------
// Approach 1: Un-fused multi-pass (Separate kernels, high memory traffic)
// -----------------------------------------------------------------------------
__global__ void silu_kernel(const float *__restrict__ gate,
                            float *__restrict__ gate_act, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    gate_act[idx] = silu(gate[idx]);
  }
}

__global__ void mul_kernel(const float *__restrict__ a,
                           const float *__restrict__ b,
                           float *__restrict__ out, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    out[idx] = a[idx] * b[idx];
  }
}

// -----------------------------------------------------------------------------
// Approach 2: Fused CUDA Kernel (Single pass, keeps activation in registers)
// -----------------------------------------------------------------------------
__global__ void swiglu_fused_kernel(const float *__restrict__ gate,
                                    const float *__restrict__ up,
                                    float *__restrict__ out,
                                    int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    float g = gate[idx];
    float u = up[idx];
    out[idx] = silu(g) * u;
  }
}

// -----------------------------------------------------------------------------
// Approach 3: Fused Vectorized Kernel (float4, 128-bit memory transactions)
// -----------------------------------------------------------------------------
__global__ void swiglu_fused_vectorized(const float *__restrict__ gate,
                                        const float *__restrict__ up,
                                        float *__restrict__ out,
                                        int total_elements4) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements4) {
    const float4 *gate4 = reinterpret_cast<const float4 *>(gate);
    const float4 *up4 = reinterpret_cast<const float4 *>(up);
    float4 *out4 = reinterpret_cast<float4 *>(out);

    float4 g = gate4[idx];
    float4 u = up4[idx];
    float4 r;
    r.x = silu(g.x) * u.x;
    r.y = silu(g.y) * u.y;
    r.z = silu(g.z) * u.z;
    r.w = silu(g.w) * u.w;
    out4[idx] = r;
  }
}

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "======================================================\n";
  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "Target: DeepSeek-14B SwiGLU FFN Layer\n";
  std::cout << "Intermediate Dim: " << INTERMEDIATE_DIM << "\n";
  std::cout << "======================================================\n\n";

  std::vector<int> token_counts = {1, 16, 2048};

  for (int num_tokens : token_counts) {
    int total_elements = num_tokens * INTERMEDIATE_DIM;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_gate(total_elements);
    std::vector<float> h_up(total_elements);
    std::vector<float> h_out_ref(total_elements);
    std::vector<float> h_out_gpu(total_elements);

    for (int i = 0; i < total_elements; ++i) {
      h_gate[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
      h_up[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
      float g = h_gate[i];
      h_out_ref[i] = (g / (1.0f + std::exp(-g))) * h_up[i];
    }

    float *d_gate, *d_up, *d_out, *d_tmp;
    CHECK_CUDA(cudaMalloc(&d_gate, bytes));
    CHECK_CUDA(cudaMalloc(&d_up, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_tmp, bytes)); // Intermediate buffer for un-fused

    CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_up, h_up.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    const int ITERS = (num_tokens == 1) ? 2000 : 500;

    std::cout << "--- Workload: " << num_tokens << " Token(s) (Elements: "
              << total_elements << ") ---\n";

    // 1. Benchmark Un-fused
    {
      int block = 256;
      int grid = (total_elements + block - 1) / block;

      CHECK_CUDA(cudaEventRecord(start));
      for (int it = 0; it < ITERS; ++it) {
        silu_kernel<<<grid, block>>>(d_gate, d_tmp, total_elements);
        mul_kernel<<<grid, block>>>(d_tmp, d_up, d_out, total_elements);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      float avg_us = (total_ms / ITERS) * 1000.0f;

      // Traffic: silu reads gate, writes tmp. mul reads tmp, reads up, writes out.
      // Total: 5 * bytes
      double un_fused_gb = (5.0 * bytes) / (avg_us * 1e-6) / 1e9;
      std::cout << "  Approach 1: Un-fused (2 Kernels + VRAM spill):\n"
                << "    Avg Latency: " << avg_us << " us\n"
                << "    Bandwidth:   " << un_fused_gb << " GB/s\n";
    }

    // 2. Benchmark Fused Naive
    {
      int block = 256;
      int grid = (total_elements + block - 1) / block;

      CHECK_CUDA(cudaEventRecord(start));
      for (int it = 0; it < ITERS; ++it) {
        swiglu_fused_kernel<<<grid, block>>>(d_gate, d_up, d_out, total_elements);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      float avg_us = (total_ms / ITERS) * 1000.0f;

      // Fused traffic: read gate, read up, write out = 3 * bytes
      double fused_gb = (3.0 * bytes) / (avg_us * 1e-6) / 1e9;
      std::cout << "  Approach 2: Fused Single Kernel:\n"
                << "    Avg Latency: " << avg_us << " us\n"
                << "    Bandwidth:   " << fused_gb << " GB/s\n";
    }

    // 3. Benchmark Fused Vectorized (float4)
    {
      int total_elements4 = total_elements / 4;
      int block = 256;
      int grid = (total_elements4 + block - 1) / block;

      CHECK_CUDA(cudaEventRecord(start));
      for (int it = 0; it < ITERS; ++it) {
        swiglu_fused_vectorized<<<grid, block>>>(d_gate, d_up, d_out,
                                                total_elements4);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      float avg_us = (total_ms / ITERS) * 1000.0f;

      double fused_gb = (3.0 * bytes) / (avg_us * 1e-6) / 1e9;

      // Check correctness
      CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, bytes,
                            cudaMemcpyDeviceToHost));
      float max_err = 0.0f;
      for (int i = 0; i < total_elements; ++i) {
        max_err = std::max(max_err, std::abs(h_out_gpu[i] - h_out_ref[i]));
      }

      std::cout << "  Approach 3: Fused + Vectorized float4 (128-bit):\n"
                << "    Avg Latency: " << avg_us << " us\n"
                << "    Bandwidth:   " << fused_gb << " GB/s\n"
                << "    Max Error:   " << max_err << (max_err < 1e-5 ? " (PASS)" : " (FAIL)") << "\n\n";
    }

    CHECK_CUDA(cudaFree(d_gate));
    CHECK_CUDA(cudaFree(d_up));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
  }

  return 0;
}
