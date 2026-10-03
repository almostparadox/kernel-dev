# PagedAttention + FlashDecoding Engine Design Specification

- **Date:** 2026-10-03
- **Author:** System & Kernel Engineer
- **Status:** Approved for Implementation Planning
- **Target Hardware:** NVIDIA Ampere (A100 sm_80), Ada Lovelace (RTX 4090 sm_89), Blackwell / Laptop (RTX 5070 sm_120)

---

## 1. Executive Summary & Problem Formulation

Standard LLM autoregressive inference suffers from severe memory fragmentation and memory-bandwidth bottlenecks during the decode phase ($M=1$ token generation) [1].

### 1.1 Memory Fragmentation in Contiguous Allocations
Vanilla PyTorch allocates contiguous tensors for Key-Value (KV) caches:
$$\text{Shape} = [\text{Batch}, \text{NumHeads}, \text{MaxSeqLen}, \text{HeadDim}]$$

Because sequence lengths vary per request, static pre-allocation wastes 60% to 80% of GPU HBM via [1]:
1. **Internal Fragmentation:** Reserved space for ungenerated tokens up to `MaxSeqLen`.
2. **External Fragmentation:** Dynamic reallocations create discontinuous memory holes across requests.
3. **Sharing Barriers:** Inability to share prompt KV caches across parallel sampling beams or system prompts.

### 1.2 Underutilization in Long-Context Decode
During single-token decode ($M=1$), computing attention against long contexts ($L \ge 2048$) underutilizes modern GPU compute capability [4]. A single thread-block per attention head starves the Streaming Multiprocessors (SMs), turning the operation into a pure DRAM-latency bottleneck.

### 1.3 Proposed System Solution
This system implements:
1. **PagedAttention CUDA Kernels:** Virtual-memory mapped KV cache with fixed-size physical pages ($B=16$ tokens) [1].
2. **FlashDecoding (Split-KV):** Context partitioning along the sequence dimension ($L$) into $K$ parallel splits across independent SMs, followed by an online log-sum-exp reduction kernel [4].
3. **Virtual Memory Host Allocator:** A zero-fragmentation page table manager in Python mirroring OS page tables [1].

---

## 2. Mathematical Model & Derivations

### 2.1 Paged Addressing Function
Let context length of sequence $s$ be $L_s$. The sequence is partitioned into logical blocks of size $B = 16$.
The logical block index of token $t \in [0, L_s - 1]$ is:
$$b_{\text{logical}}(t) = \left\lfloor \frac{t}{B} \right\rfloor$$
The intra-block token offset is:
$$o(t) = t \pmod B$$

Given block table $\mathcal{T} \in \mathbb{Z}^{S \times M_{\text{blocks}}}$, the physical block index is:
$$p = \mathcal{T}[s, b_{\text{logical}}(t)]$$

For physical cache tensors $\mathbf{K}_{\text{pool}}, \mathbf{V}_{\text{pool}} \in \mathbb{R}^{N_{\text{blocks}} \times H \times B \times d}$, physical byte offset is:
$$\text{Offset}(p, h, o) = p \cdot (H \cdot B \cdot d) + h \cdot (B \cdot d) + o \cdot d$$

### 2.2 Online Softmax with Running Max and Normalizer
For query vector $\mathbf{q} \in \mathbb{R}^d$ and scaling factor $\sigma = \frac{1}{\sqrt{d}}$:
At any step over tokens $t \in [0, L_s - 1]$:
$$S_t = \sigma (\mathbf{q} \cdot \mathbf{k}_t)$$

Online numerical stabilization maintains running maximum $m^{(t)}$ and running denominator $l^{(t)}$ [2], [5]:
$$m^{(t)} = \max\left(m^{(t-1)}, S_t\right)$$
$$\alpha = \exp\left(m^{(t-1)} - m^{(t)}\right)$$
$$l^{(t)} = \alpha \cdot l^{(t-1)} + \exp\left(S_t - m^{(t)}\right)$$

Output vector $\mathbf{o}^{(t)}$ is updated without materializing intermediate scores in HBM:
$$\mathbf{o}^{(t)} = \alpha \cdot \mathbf{o}^{(t-1)} + \exp\left(S_t - m^{(t)}\right) \mathbf{v}_t$$

Final normalized output:
$$\mathbf{O} = \frac{\mathbf{o}^{(L_s - 1)}}{l^{(L_s - 1)}}$$

### 2.3 Split-KV (FlashDecoding) Two-Stage Reduction
For context split into $K$ disjoint chunks along the sequence axis, each split $k \in [0, K-1]$ computes local partial results [4]:
$$\mathbf{o}_k \in \mathbb{R}^d, \quad m_k \in \mathbb{R}, \quad l_k \in \mathbb{R}$$

Stage 2 reduction merges the $K$ splits:
$$m_{\text{global}} = \max_{k \in [0, K-1]} m_k$$
$$\beta_k = \exp(m_k - m_{\text{global}})$$
$$l_{\text{global}} = \sum_{k=0}^{K-1} \beta_k \cdot l_k$$
$$\mathbf{O}_{\text{final}} = \frac{1}{l_{\text{global}}} \sum_{k=0}^{K-1} \beta_k \cdot \mathbf{o}_k$$

---

## 3. Data Structures & Memory Layout

### 3.1 Physical KV Memory Pool
- **Tensor:** `key_cache`, `value_cache`
- **Data type:** `half` (FP16, 2 bytes) or `nv_bfloat16`
- **Shape:** `[total_num_blocks, num_heads, block_size, head_dim]`
- **Block size ($B$):** 16 tokens
- **Head dimensions ($d$):** 64, 128 (configurable at compile time)
- **Alignment:** 16-byte aligned to allow `float4` (8x FP16) vectorized memory instructions (`LDG.E.128`).

### 3.2 Block Table
- **Tensor:** `block_tables`
- **Data type:** `int32_t`
- **Shape:** `[batch_size, max_num_blocks_per_seq]`
- **Semantics:** Maps logical block index to physical slot index in `key_cache`/`value_cache`.

### 3.3 Context Lengths
- **Tensor:** `context_lens`
- **Data type:** `int32_t`
- **Shape:** `[batch_size]`
- **Semantics:** Actual number of valid cached tokens for each sequence.

---

## 4. CUDA Kernel Architecture

### 4.1 Kernel 1: `paged_attention_v1_kernel` (Single-pass Decode)
Designed for small to moderate context lengths ($L \le 2048$).
- **Grid:** `(batch_size, num_heads)`
- **Block:** `(128, 1, 1)` (4 warps per thread block)
- **Shared Memory:**
  - `s_q[head_dim]`: Query vector loaded once via `float4` and held in SRAM.
  - `s_logits[block_size]`: Temporary warp reductions.
  - `s_warp_max[4]`, `s_warp_sum[4]`: Inter-warp reduction scratchpad.
- **Register Footprint Target:** $\le 64$ registers/thread to achieve $\ge 50\%$ theoretical occupancy.

### 4.2 Kernel 2: `paged_attention_splitkv_stage1_kernel`
Designed for long contexts ($L > 2048$).
- **Grid:** `(batch_size, num_heads, num_splits)`
- **Block:** `(128, 1, 1)`
- **Output:** 
  - `tmp_out[batch_size, num_heads, num_splits, head_dim]`
  - `tmp_metadata[batch_size, num_heads, num_splits, 2]` (stores $m_k, l_k$)

### 4.3 Kernel 3: `paged_attention_splitkv_stage2_kernel` (Cross-Split Reduction)
- **Grid:** `(batch_size, num_heads)`
- **Block:** `(32, 1, 1)` or `(64, 1, 1)`
- Reads $K$ outputs per `(batch, head)`, executes online softmax merge via warp shuffle (`__shfl_down_sync`), and writes final `output[batch_size, num_heads, head_dim]`.

---

## 5. Host Block Allocator (Python)

Implemented in `03-cuda-llm-kernels/paged_attention/paged_allocator.py`:
- `PhysicalBlockPool`: Allocates GPU memory slab at engine initialization.
- `BlockAllocator`:
  - `free_blocks`: Stack / bitset of available block IDs.
  - `allocate(num_blocks)`: O(1) allocation.
  - `free(block_ids)`: O(1) release back to pool.
- `SequenceGroup`: Tracks logical-to-physical block table mappings dynamically as tokens arrive.

---

## 6. Integration & Verification Plan

### 6.1 PyTorch Extension Interface
Build shared library `paged_attention.so` via `torch.utils.cpp_extension` or direct `nvcc` compilation with C-linkage entry points:
```c
extern "C" void launch_paged_attention_v1(
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
```

### 6.2 Correctness Checks (`test_paged_attention.py`)
- Compare output against `torch.nn.functional.scaled_dot_product_attention` on unpaged ground-truth tensors.
- Tolerances: Max absolute error $< 1e-3$, mean squared error $< 1e-5$ for FP16.

### 6.3 Benchmarks (`paged_bench.py` & `paged_bench.cu`)
1. **Microbenchmark:** Contiguous decode attention vs PagedAttention vs FlashDecoding across context lengths ($L \in \{256, 1024, 4096, 16384\}$).
2. **Serving Throughput Benchmark:** Measure token generation throughput and VRAM footprint under high batch concurrency (Batch $= 1, 4, 16, 32$).

---

## 7. Vast.ai Execution & Compilation Specification

- **Target Architecture Flag:** `-gencode arch=compute_80,code=sm_80` (A100) or `-gencode arch=compute_89,code=sm_89` (RTX 4090)
- **Compiler Flags:**
  ```bash
  nvcc -O3 -std=c++17 --use_fast_math -Xcompiler -fPIC \
       -gencode arch=compute_80,code=sm_80 \
       -gencode arch=compute_89,code=sm_89 \
       -gencode arch=compute_120,code=sm_120 \
       -I${CUDA_HOME}/include --shared paged_attention.cu -o paged_attention.so
  ```
- **NCU Profiling Command:**
  ```bash
  ncu --set full --target-processes all python paged_bench.py
  ```

---

## 8. References (IEEE Style)

[1] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23)*, Koblenz, Germany, 2023, pp. 611–626. doi: [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165).

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," in *Advances in Neural Information Processing Systems (NeurIPS 2022)*, vol. 35, 2022, pp. 16344–16359.

[3] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," in *International Conference on Learning Representations (ICLR 2024)*, Vienna, Austria, 2024. arXiv: [2307.08691](https://arxiv.org/abs/2307.08691).

[4] T. Dao, D. Haziza, F. Massa, and G. Sizov, "Flash-Decoding for long-context inference," *Stanford Center for Research on Foundation Models (CRFM)*, Oct. 2023. [Online]. Available: https://crfm.stanford.edu/2023/10/12/flashdecoding.html

[5] M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax," *arXiv preprint arXiv:1805.02867*, 2018. doi: [10.48550/arXiv.1805.02867](https://doi.org/10.48550/arXiv.1805.02867).

[6] NVIDIA Corporation, *CUDA C++ Programming Guide (Release 12.x)*, Santa Clara, CA, USA, 2024. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[7] NVIDIA Corporation, *NVIDIA A100 Tensor Core GPU Architecture: Unprecedented Acceleration at Every Scale*, Whitepaper WP-10019-001_v01, Santa Clara, CA, USA, 2020. [Online]. Available: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
