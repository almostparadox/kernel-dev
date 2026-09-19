#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__inline__ __device__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __inline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------
// RMSNorm FP16: Single-pass, Warp Shuffle reduction, zero intermediate DRAM
// -----------------------------------------------------------------------------
__global__ void rmsnorm_kernel_fp16(const half *__restrict__ x,
                                    const half *__restrict__ weight,
                                    half *__restrict__ out,
                                    int cols, float eps) {
  int row = blockIdx.x;
  const half *rx = x + row * cols;
  half *rout = out + row * cols;

  __shared__ float s_warp_sums[32];
  __shared__ float s_rrms;

  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = __half2float(rx[i]);
    thread_sum += v * v;
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
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float vx = __half2float(rx[i]);
    float vw = __half2float(weight[i]);
    rout[i] = __float2half(vx * rrms * vw);
  }
}

// -----------------------------------------------------------------------------
// SwiGLU FP16: Single-pass Fused silu(gate) * up
// -----------------------------------------------------------------------------
__global__ void swiglu_kernel_fp16(const half *__restrict__ gate,
                                   const half *__restrict__ up,
                                   half *__restrict__ out,
                                   int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    out[idx] = __float2half(silu(g) * u);
  }
}

// -----------------------------------------------------------------------------
// C Export Interface for Python ctypes
// -----------------------------------------------------------------------------
extern "C" {
  __declspec(dllexport) void launch_rmsnorm_fp16(uintptr_t x, uintptr_t weight,
                                                 uintptr_t out, int rows,
                                                 int cols, float eps,
                                                 uintptr_t stream) {
    dim3 grid(rows);
    dim3 block(256);
    rmsnorm_kernel_fp16<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const half *>(x),
        reinterpret_cast<const half *>(weight),
        reinterpret_cast<half *>(out),
        cols, eps);
  }

  __declspec(dllexport) void launch_swiglu_fp16(uintptr_t gate, uintptr_t up,
                                                uintptr_t out,
                                                int total_elements,
                                                uintptr_t stream) {
    int block = 256;
    int grid = (total_elements + block - 1) / block;
    swiglu_kernel_fp16<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const half *>(gate),
        reinterpret_cast<const half *>(up),
        reinterpret_cast<half *>(out),
        total_elements);
  }
}
