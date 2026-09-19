# Kernel Benchmark Stats: DeepSeek-R1 14B on RTX 5070

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (36 SMs, 8GB GDDR7, CC 12.0)  
Architecture: `qwen2` / DeepSeek-R1-Distill-Qwen-14B ($D=5120$, $D_{ffn}=13824$, 48 Layers)

---

## 1. RMSNorm Benchmark ($D=5120$)

Target: 2 RMSNorm calls per transformer layer = **96 calls per model forward pass**.

| Workload | Metric | Level 1 (Naive SMEM) | Level 2 (Warp Shuffle) | Level 3 (Vectorized `float4`) | Speedup (L3 vs L1) |
|---|---|---|---|---|---|
| 1 Token (Decode) | Avg Latency | 14.75 µs | 12.91 µs | **10.00 µs** | **1.47x** (-32.2%) |
| 1 Token (Decode) | Bandwidth | 4.16 GB/s | 4.76 GB/s | **6.14 GB/s** | +47.6% |
| 16 Tokens (Batch) | Avg Latency | 14.72 µs | 13.73 µs | **9.33 µs** | **1.58x** (-36.6%) |
| 16 Tokens (Batch) | Bandwidth | 45.91 GB/s | 49.24 GB/s | **72.40 GB/s** | +57.7% |
| 2048 Tokens (Prefill) | Avg Latency | 291.23 µs | 277.97 µs | **272.97 µs** | **1.07x** (-6.3%) |
| 2048 Tokens (Prefill) | Bandwidth | 288.11 GB/s | 301.86 GB/s | **307.38 GB/s** | ~80% bus saturation |

---

## 2. SwiGLU Activation Benchmark ($D_{ffn}=13824$)

Target: 1 SwiGLU call per transformer layer = **48 calls per model forward pass**.  
Formula: $\text{SwiGLU}(x) = \text{SiLU}(x W_{gate}) \odot (x W_{up})$

| Workload | Metric | Approach 1 (Un-fused) | Approach 2 (Fused) | Approach 3 (Fused + `float4`) | Speedup (Fused vs Un-fused) |
|---|---|---|---|---|---|
| 1 Token (Decode) | Avg Latency | 17.35 µs | **8.74 µs** | 9.10 µs | **1.99x** (-49.6%) |
| 1 Token (Decode) | VRAM Transfers | 5 passes | 3 passes | 3 passes | -40% DRAM traffic |
| 16 Tokens (Batch) | Avg Latency | 16.44 µs | **8.46 µs** | 9.32 µs | **1.94x** (-48.5%) |
| 16 Tokens (Batch) | Bandwidth | 269.16 GB/s | 313.67 GB/s | 284.91 GB/s | +16.5% |
| 2048 Tokens (Prefill) | Avg Latency | 1806.14 µs (1.81 ms) | 1074.38 µs (1.07 ms) | **1069.85 µs (1.07 ms)** | **1.69x** (-40.8%) |
| 2048 Tokens (Prefill) | Per-call Saving | Baseline | 0.732 ms / call | **0.736 ms / call** | 40.8% cut |

---

## 3. Cumulative Model-Level Impact (DeepSeek-14B, 48 Layers)

Calculated across full forward pass ($96 \times \text{RMSNorm} + 48 \times \text{SwiGLU}$):

### A. Decode Generation Phase (Single Token, Latency-Critical)
- **RMSNorm total (96x)**: 1.416 ms $\to$ 0.960 ms (saves **0.456 ms**)
- **SwiGLU total (48x)**: 0.833 ms $\to$ 0.419 ms (saves **0.414 ms**)
- **Net saving per token**: **0.870 ms / token**
- **1000-token generation**: Saves **0.87 seconds** of pure DRAM stall time.

### B. Prompt Prefill Phase (2048 Tokens, Throughput-Critical)
- **RMSNorm total (96x)**: 27.96 ms $\to$ 26.21 ms (saves **1.75 ms**)
- **SwiGLU total (48x)**: 86.70 ms $\to$ 51.35 ms (saves **35.35 ms**)
- **Net saving per prefill**: **37.10 ms / forward pass**
