# DeepSeek-R1 14B GPU Kernel Benchmarks

Benchmarks for GPU compute kernels matching exact architecture of `deepseek-r1:14b`.

## Model Dimensions (from local GGUF metadata)
- Architecture: `qwen2` (DeepSeek-R1-Distill-Qwen-14B)
- Hidden Dimension ($D$): `5120`
- FFN Intermediate Dimension ($D_{ffn}$): `13824`
- Transformer Layers: `48`
- Attention Heads: `40` (Q), `8` (KV - GQA)
- RMSNorm Epsilon: `1e-5`

## Kernels Implemented

### 1. `rmsnorm_bench.cu`
Formula: $y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot w$

- **Level 1 (Naive)**: Row-level block, shared memory tree reduction with `__syncthreads()`.
- **Level 2 (Warp Shuffle)**: Register-level reduction using `__shfl_down_sync()`. Zero bank conflicts.
- **Level 3 (Vectorized float4)**: 128-bit memory transactions (4 floats/thread load). Full memory bus utilization.

### 2. `swiglu_bench.cu`
Formula: $\text{SwiGLU}(x, W_{gate}, W_{up}) = \text{SiLU}(x W_{gate}) \odot (x W_{up})$

- **Approach 1 (Un-fused)**: 2 separate kernel launches (`silu` + `mul`) with intermediate tensor written to VRAM.
- **Approach 2 (Fused)**: Single kernel launch. Keeps activation in registers.
- **Approach 3 (Fused + Vectorized float4)**: Single kernel with 128-bit coalesced memory loads.

## Benchmark Results

Detailed profiling and model-level impact: see [BENCHMARK_STATS.md](BENCHMARK_STATS.md).

## Build and Run

From repository root:
```cmd
:: 1. Compile and run microbenchmarks
scripts\nvcc_run.bat -arch=native 03-cuda-llm-kernels\rmsnorm_bench.cu -o 03-cuda-llm-kernels\rmsnorm_bench.exe
03-cuda-llm-kernels\rmsnorm_bench.exe

scripts\nvcc_run.bat -arch=native 03-cuda-llm-kernels\swiglu_bench.cu -o 03-cuda-llm-kernels\swiglu_bench.exe
03-cuda-llm-kernels\swiglu_bench.exe

:: 2. Compile custom ops shared library for PyTorch integration
scripts\nvcc_run.bat --shared 03-cuda-llm-kernels\custom_ops.cu -o 03-cuda-llm-kernels\custom_ops.dll

:: 3. Run side-by-side LLM inference benchmarks
scripts\run_side_by_side_7b.bat "Explain quantum computing in one short sentence."
scripts\run_side_by_side_3b.bat "Explain quantum computing in one short sentence."
```
