# Kernel Development Lab

Systems programming and hardware-level performance engineering workspace:
1. **GPU Compute Kernels (CUDA)**: Memory-bound LLM operator optimization (RMSNorm, SwiGLU) for DeepSeek-R1/Qwen architectures with live side-by-side PyTorch patching.
2. **OS Kernel Modules (Linux LKM)**: Loadable kernel modules, character device drivers, and user-space I/O.

---

## Repository Layout

```text
kernel-dev/
├── 01-hello-lkm/                # Minimal Linux Loadable Kernel Module
│   ├── Makefile
│   └── hello.c
├── 02-char-device/              # Linux miscdevice Driver & User-space Test
│   ├── Makefile
│   ├── chardev.c
│   └── test.c
├── 03-cuda-llm-kernels/         # CUDA LLM Optimization Kernels & Benchmarks
│   ├── BENCHMARK_STATS.md       # Detailed profiling tables & DRAM bandwidth analysis
│   ├── custom_ops.cu            # FP16 single-pass RMSNorm & fused SwiGLU kernels
│   ├── rmsnorm_bench.cu         # DeepSeek-14B RMSNorm benchmark (Naive vs Warp Shuffle vs Vectorized)
│   ├── swiglu_bench.cu          # DeepSeek-14B SwiGLU benchmark (Un-fused vs Fused)
│   ├── side_by_side_gen_7b.py   # Live side-by-side inference on 4-bit DeepSeek-R1 7B
│   ├── side_by_side_gen.py      # Live side-by-side inference on FP16 Qwen2.5 3B
│   ├── generate_bench.py        # Ollama telemetry mapper (48 layers)
│   └── test_custom_ops.py       # PyTorch vs CUDA unit correctness check
└── scripts/                     # Helper wrappers for build and execution
    ├── nvcc_run.bat             # MSVC + NVCC 13.0 compilation wrapper
    ├── run_side_by_side_7b.sh   # Run 7B 4-bit side-by-side benchmark (Linux/WSL)
    ├── run_side_by_side_7b.bat  # Run 7B 4-bit side-by-side benchmark (Windows)
    ├── run_side_by_side_3b.sh   # Run 3B FP16 side-by-side benchmark (Linux/WSL)
    ├── run_side_by_side_3b.bat  # Run 3B FP16 side-by-side benchmark (Windows)
    ├── run_generation.sh        # Run Ollama generation telemetry (Linux/WSL)
    └── run_generation.bat       # Run Ollama generation telemetry (Windows)
```

---

## Part 1: CUDA LLM Kernel Optimization

Benchmarked on **NVIDIA GeForce RTX 5070 Laptop GPU** (36 SMs, 8GB GDDR7, CC 12.0).

### Key Architectural Optimizations
- **Vectorized Memory (`float4`)**: Loads 128-bit memory transactions per instruction to saturate bus bandwidth.
- **Warp-Level Shuffle (`__shfl_down_sync`)**: Eliminates shared memory bank conflicts and `__syncthreads()` stalls.
- **Kernel Fusion**: Combines SiLU activation and elementwise multiplication into a single memory pass, eliminating DRAM round-trips.

### Benchmark Highlights

#### 1. Live LLM Generation: Native PyTorch vs Custom CUDA
Tested on identical prompts with model layer hot-patching:

| Model | Format | PyTorch Throughput | Custom CUDA Throughput | Speedup | Token Match |
| --- | --- | --- | --- | --- | --- |
| **DeepSeek-R1-Distill-Qwen-7B** | 4-bit NF4 | 19.91 tok/s | **22.80 tok/s** | **+14.6%** | **100% Identical** |
| **Qwen2.5-3B-Instruct** | FP16 | 17.97 tok/s | **26.34 tok/s** | **+46.6%** | **100% Identical** |

#### 2. DeepSeek-R1 14B Operator Microbenchmarks ($D=5120, D_{ffn}=13824$)
- **RMSNorm (Decode, 1 Token)**: `14.54 µs` $\to$ `9.56 µs` (**1.52x faster**).
- **SwiGLU (Decode, 1 Token)**: `17.35 µs` $\to$ `8.74 µs` (**1.99x faster**).
- **Cumulative impact across 48 layers**: Saves **~0.87 ms per token** generated and **~37.1 ms per prefill**.

For detailed bandwidth measurements, see [03-cuda-llm-kernels/BENCHMARK_STATS.md](03-cuda-llm-kernels/BENCHMARK_STATS.md).

### Quickstart: Running CUDA Benchmarks

#### Compile Custom Ops DLL
```cmd
scripts\nvcc_run.bat --shared 03-cuda-llm-kernels\custom_ops.cu -o 03-cuda-llm-kernels\custom_ops.dll
```

#### Run Side-by-Side 7B Inference (DeepSeek-R1 7B NF4)
```bash
# WSL / Linux:
./scripts/run_side_by_side_7b.sh "Explain quantum computing in one short sentence."

# Windows CMD / PowerShell:
.\scripts\run_side_by_side_7b.bat "Explain quantum computing in one short sentence."
```

#### Run DeepSeek-14B Operator Benchmarks
```cmd
scripts\nvcc_run.bat -arch=native 03-cuda-llm-kernels\rmsnorm_bench.cu -o 03-cuda-llm-kernels\rmsnorm_bench.exe
03-cuda-llm-kernels\rmsnorm_bench.exe

scripts\nvcc_run.bat -arch=native 03-cuda-llm-kernels\swiglu_bench.cu -o 03-cuda-llm-kernels\swiglu_bench.exe
03-cuda-llm-kernels\swiglu_bench.exe
```

---

## Part 2: Linux OS Kernel Modules

Windows cannot compile Linux LKMs directly. Use WSL2 Ubuntu with generic kernel headers.

### 1. Install Dependencies
```bash
sudo apt update
sudo apt install -y build-essential linux-headers-generic kmod
```

### 2. Build & Test
- `01-hello-lkm`: Minimal Loadable Kernel Module (`init`, `exit`, `pr_info`).
  ```bash
  cd 01-hello-lkm && make
  ```
- `02-char-device`: `miscdevice` driver with user-space read/write and assert verification.
  ```bash
  cd 02-char-device && make && ./test
  ```

---

## Part 3: PagedAttention & FlashDecoding (Production Inference Engine)

Hardware-level implementation of virtual memory paging for LLM Key-Value caches and multi-split parallel decode attention [1], [4].

### Key Architectural Optimizations
- **Virtual Page Allocation ($B=16$)**: Emulates OS page tables with fixed-size physical blocks, eliminating 60–80% HBM fragmentation during autoregressive decoding [1].
- **Online Softmax Accumulation**: Single-pass fused dot-product attention maintaining running numerical max $m$ and sum-of-exponentials $l$ in SRAM [2], [5].
- **FlashDecoding Split-KV**: Sequence-axis partitioning across independent CUDA thread blocks for long contexts ($L \ge 2048$), followed by a log-sum-exp reduction kernel [4].
- **Mathematical Specification**: Formal LaTeX derivations and algorithmic proofs available in [`docs/math/paged_attention_math.tex`](docs/math/paged_attention_math.tex) (compiled for TeXShop with IEEE bibliography).

---

## Authoritative Literature & References (IEEE Style)

- **[1]** W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in *Proc. 29th ACM Symp. Operating Syst. Princ. (SOSP '23)*, 2023, pp. 611–626. doi: [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165).
- **[2]** T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," in *Adv. Neural Inf. Process. Syst. (NeurIPS 2022)*, vol. 35, 2022, pp. 16344–16359.
- **[3]** T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," in *Int. Conf. Learn. Represent. (ICLR 2024)*, 2024. arXiv: [2307.08691](https://arxiv.org/abs/2307.08691).
- **[4]** T. Dao, D. Haziza, F. Massa, and G. Sizov, "Flash-Decoding for long-context inference," *Stanford CRFM*, 2023. [Online]. Available: https://crfm.stanford.edu/2023/10/12/flashdecoding.html.
- **[5]** M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax," *arXiv preprint arXiv:1805.02867*, 2018. doi: [10.48550/arXiv.1805.02867](https://doi.org/10.48550/arXiv.1805.02867).
- **[6]** NVIDIA Corporation, *CUDA C++ Programming Guide (Release 12.x)*, Santa Clara, CA, USA, 2024. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
- **[7]** NVIDIA Corporation, *NVIDIA A100 Tensor Core GPU Architecture*, Whitepaper WP-10019-001_v01, Santa Clara, CA, USA, 2020. [Online]. Available: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf.
