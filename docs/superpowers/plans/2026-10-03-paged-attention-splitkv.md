# PagedAttention + FlashDecoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a production-grade PagedAttention V1 and FlashDecoding (Split-KV) CUDA inference kernel with virtual page block allocator and benchmark suite for modern NVIDIA GPUs (Ampere sm_80 / Ada sm_89 / Blackwell sm_120).

**Architecture:** A host-side virtual memory block manager allocates fixed 16-token physical KV cache pages, feeding block tables into custom CUDA decode kernels. Short contexts use single-pass online softmax; long contexts partition the sequence into parallel splits with a second-stage log-sum-exp reduction kernel.

**Tech Stack:** C++17, CUDA C (NVCC 12.x/13.x), PyTorch C++ Extension API, NumPy, Python 3.10+.

**Spec:** `docs/superpowers/specs/2026-10-03-paged-attention-splitkv-design.md`

## Authoritative Literature & Standards
- **PagedAttention:** Woosuk Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- **FlashAttention-2:** Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024 [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
- **Flash-Decoding:** Tri Dao et al., *Flash-Decoding for Long-Context Inference*, Stanford CRFM 2023 [https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).
- **Online Softmax:** Maxim Milakov & Natalia Gimelshein, *Online normalizer calculation for softmax*, 2018 [arXiv:1805.02867](https://arxiv.org/abs/1805.02867).

## Global Constraints
- Token block size $B = 16$ tokens per physical page (Kwon et al. 2023 Section 3.2).
- Supported head dimensions: $d \in \{64, 128\}$.
- Data type: IEEE 754 Half-Precision (`half` / `torch.float16`).
- Vectorized memory transfers: All memory accesses to KV blocks must use 128-bit `float4` transactions (8 `half` values per instruction).
- Numerically stable online softmax using running max $m$ and sum-of-exponentials $l$ (Milakov & Gimelshein 2018).

---

## File Structure & Responsibilities

```text
03-cuda-llm-kernels/
├── paged_attention/
│   ├── __init__.py               # Python package export
│   ├── paged_allocator.py        # Host virtual memory block allocator & table builder
│   ├── paged_cache.h             # Common constants, vector types, and kernel signatures
│   ├── paged_decode.cu           # PagedAttention V1 single-pass decode kernel
│   ├── paged_split_kv.cu         # FlashDecoding Split-KV stage 1 + stage 2 reduction
│   ├── paged_ops.cpp             # PyTorch C++ bindings / ctypes wrapper
│   └── test_paged_attention.py   # Unit test comparing against PyTorch scaled_dot_product_attention
├── paged_bench.py                # Microbenchmark: Contiguous vs Paged vs Split-KV
└── scripts/
    ├── compile_paged_ops.sh      # Linux/Vast.ai NVCC build script for sm_80/sm_89/sm_120
    └── run_vast_bench.sh         # Automated benchmarking & NCU profiling script for Vast.ai
```

---

### Task 1: Virtual Memory Page Allocator (`paged_allocator.py`)

**Files:**
- Create: `03-cuda-llm-kernels/paged_attention/__init__.py`
- Create: `03-cuda-llm-kernels/paged_attention/paged_allocator.py`
- Test: `03-cuda-llm-kernels/paged_attention/test_allocator.py`

**Interfaces:**
- Produces: `PagedBlockAllocator(total_blocks: int, block_size: int = 16)`
  - `allocate(num_blocks: int) -> list[int]`
  - `free(block_ids: list[int]) -> None`
  - `get_num_free_blocks() -> int`
- Produces: `SequenceBlockTableManager(allocator: PagedBlockAllocator, block_size: int = 16)`
  - `allocate_sequence(seq_id: int, initial_tokens: int) -> None`
  - `append_token(seq_id: int) -> None`
  - `free_sequence(seq_id: int) -> None`
  - `build_block_table_tensor(seq_ids: list[int], max_blocks: int) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

```python
# 03-cuda-llm-kernels/paged_attention/test_allocator.py
import pytest
import torch
from paged_attention.paged_allocator import PagedBlockAllocator, SequenceBlockTableManager

def test_allocator_basic_lifecycle():
    allocator = PagedBlockAllocator(total_blocks=10, block_size=16)
    assert allocator.get_num_free_blocks() == 10
    
    blocks = allocator.allocate(3)
    assert len(blocks) == 3
    assert allocator.get_num_free_blocks() == 7
    
    allocator.free(blocks[:1])
    assert allocator.get_num_free_blocks() == 8

def test_sequence_manager_expansion():
    allocator = PagedBlockAllocator(total_blocks=100, block_size=16)
    manager = SequenceBlockTableManager(allocator, block_size=16)
    
    manager.allocate_sequence(seq_id=1, initial_tokens=16)
    assert len(manager.seq_to_blocks[1]) == 1
    
    # Appending 17th token triggers new block allocation
    manager.append_token(seq_id=1)
    assert len(manager.seq_to_blocks[1]) == 2
    assert manager.seq_lengths[1] == 17
    
    table_tensor = manager.build_block_table_tensor([1], max_blocks=4)
    assert table_tensor.shape == (1, 4)
    assert table_tensor.dtype == torch.int32
    assert table_tensor[0, 0].item() == manager.seq_to_blocks[1][0]
    assert table_tensor[0, 1].item() == manager.seq_to_blocks[1][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest 03-cuda-llm-kernels/paged_attention/test_allocator.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `paged_allocator.py` and `__init__.py`**

```python
# 03-cuda-llm-kernels/paged_attention/__init__.py
from .paged_allocator import PagedBlockAllocator, SequenceBlockTableManager

__all__ = ["PagedBlockAllocator", "SequenceBlockTableManager"]
```

```python
# 03-cuda-llm-kernels/paged_attention/paged_allocator.py
from typing import Dict, List
import torch

class PagedBlockAllocator:
    """
    Physical page frame allocator for KV cache blocks.
    Implements O(1) stack-based allocation and deallocation (Kwon et al., 2023).
    """
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(total_blocks - 1, -1, -1))

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self, num_blocks: int) -> List[int]:
        if num_blocks > len(self.free_blocks):
            raise MemoryError(
                f"Out of KV cache memory: requested {num_blocks} blocks, "
                f"only {len(self.free_blocks)} available."
            )
        allocated = [self.free_blocks.pop() for _ in range(num_blocks)]
        return allocated

    def free(self, block_ids: List[int]) -> None:
        for bid in block_ids:
            self.free_blocks.append(bid)


class SequenceBlockTableManager:
    """
    Virtual-to-physical block table manager for active inference sequences.
    """
    def __init__(self, allocator: PagedBlockAllocator, block_size: int = 16):
        self.allocator = allocator
        self.block_size = block_size
        self.seq_to_blocks: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, initial_tokens: int) -> None:
        num_blocks = (initial_tokens + self.block_size - 1) // self.block_size
        self.seq_to_blocks[seq_id] = self.allocator.allocate(num_blocks)
        self.seq_lengths[seq_id] = initial_tokens

    def append_token(self, seq_id: int) -> None:
        curr_len = self.seq_lengths[seq_id]
        if curr_len % self.block_size == 0:
            new_block = self.allocator.allocate(1)[0]
            self.seq_to_blocks[seq_id].append(new_block)
        self.seq_lengths[seq_id] = curr_len + 1

    def free_sequence(self, seq_id: int) -> None:
        if seq_id in self.seq_to_blocks:
            self.allocator.free(self.seq_to_blocks[seq_id])
            del self.seq_to_blocks[seq_id]
            del self.seq_lengths[seq_id]

    def build_block_table_tensor(self, seq_ids: List[int], max_blocks: int) -> torch.Tensor:
        batch_size = len(seq_ids)
        table = torch.full((batch_size, max_blocks), fill_value=-1, dtype=torch.int32)
        for i, sid in enumerate(seq_ids):
            blocks = self.seq_to_blocks[sid]
            n = min(len(blocks), max_blocks)
            table[i, :n] = torch.tensor(blocks[:n], dtype=torch.int32)
        return table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest 03-cuda-llm-kernels/paged_attention/test_allocator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add 03-cuda-llm-kernels/paged_attention/
git commit -m "feat(paged-attn): implement host virtual memory block allocator"
```

---

### Task 2: PagedAttention V1 CUDA Decode Kernel (`paged_cache.h` & `paged_decode.cu`)

**Files:**
- Create: `03-cuda-llm-kernels/paged_attention/paged_cache.h`
- Create: `03-cuda-llm-kernels/paged_attention/paged_decode.cu`

**Interfaces:**
- Produces: C function `paged_attention_v1_launcher`
  ```cpp
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
  ```

- [ ] **Step 1: Write header `paged_cache.h`**

```cpp
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
```

- [ ] **Step 2: Implement single-pass `paged_decode.cu`**
Implement kernel with:
1. `__shared__ half s_q[128]` for cooperative query caching via `float4`.
2. Iteration across logical blocks up to `context_lens[seq_idx]`.
3. Address calculation into `k_pool` using physical block ID from `block_tables`.
4. Online softmax tracking `m_running` and `l_running` (Milakov & Gimelshein 2018).
5. Output accumulator update and final write.

- [ ] **Step 3: Add launcher C API**
Include C-linkage launcher for integration with Python / PyTorch.

- [ ] **Step 4: Verify syntax and compile test object**
Compile with `nvcc -c paged_decode.cu -o paged_decode.o` on CUDA-capable runner.

- [ ] **Step 5: Commit**

```bash
git add 03-cuda-llm-kernels/paged_attention/paged_cache.h 03-cuda-llm-kernels/paged_attention/paged_decode.cu
git commit -m "feat(paged-attn): implement single-pass paged attention v1 decode kernel"
```

---

### Task 3: PyTorch C++ Extension & Unit Verification vs PyTorch SDPA

**Files:**
- Create: `03-cuda-llm-kernels/paged_attention/paged_ops.cpp`
- Create: `03-cuda-llm-kernels/paged_attention/test_paged_attention.py`

**Interfaces:**
- Produces: Python module `paged_ops.paged_attention_v1(...)`

- [ ] **Step 1: Write PyTorch C++ wrapper `paged_ops.cpp`**
Wrap `paged_attention_v1_launcher` into `torch::Tensor paged_attention_v1(...)` checking tensor contiguous state, dtypes, and shapes.

- [ ] **Step 2: Write verification test `test_paged_attention.py`**
Compare output of `paged_attention_v1` against unpaged `torch.nn.functional.scaled_dot_product_attention`.
Check:
- Max error $< 1e-3$
- Mean squared error $< 1e-5$
- Handles variable sequence lengths across the batch.

- [ ] **Step 3: Run test on GPU**
Run: `pytest 03-cuda-llm-kernels/paged_attention/test_paged_attention.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add 03-cuda-llm-kernels/paged_attention/paged_ops.cpp 03-cuda-llm-kernels/paged_attention/test_paged_attention.py
git commit -m "feat(paged-attn): add PyTorch C++ bindings and correctness verification suite"
```

---

### Task 4: FlashDecoding (Split-KV) Long-Context Kernels (`paged_split_kv.cu`)

**Files:**
- Create: `03-cuda-llm-kernels/paged_attention/paged_split_kv.cu`
- Modify: `03-cuda-llm-kernels/paged_attention/paged_ops.cpp`
- Modify: `03-cuda-llm-kernels/paged_attention/test_paged_attention.py`

**Interfaces:**
- Produces: C function `paged_attention_splitkv_launcher`
  - Stage 1: Computes partial outputs `tmp_out[batch, heads, splits, head_dim]` and `tmp_meta[batch, heads, splits, 2]`.
  - Stage 2: Merges splits into final `out[batch, heads, head_dim]`.

- [ ] **Step 1: Implement Stage 1 Split-KV kernel**
Partitions $L_s$ tokens across `num_splits` blocks. Each split processes interval $[k \cdot S, (k+1) \cdot S)$.

- [ ] **Step 2: Implement Stage 2 Log-Sum-Exp Reduction kernel**
Reads split partial results, computes global max $m_{\text{global}}$, rescales by $\exp(m_k - m_{\text{global}})$, and reduces to final output.

- [ ] **Step 3: Integrate with Python wrapper and test on $L=4096, 8192$**
Verify numerical equality against PyTorch SDPA on long sequences.

- [ ] **Step 4: Commit**

```bash
git add 03-cuda-llm-kernels/paged_attention/paged_split_kv.cu
git commit -m "feat(paged-attn): implement FlashDecoding Split-KV kernels for long-context decode"
```

---

### Task 5: Microbenchmark & Profiling Suite (`paged_bench.py`)

**Files:**
- Create: `03-cuda-llm-kernels/paged_bench.py`

**Interfaces:**
- Compares:
  1. Vanilla PyTorch SDPA (contiguous memory)
  2. PagedAttention V1 (single-pass)
  3. FlashDecoding Split-KV ($K=4, 8$)
- Outputs latency table ($\mu s$), memory bandwidth utilization (GB/s), and speedup factor across $L \in [128, 512, 2048, 8192, 16384]$.

- [ ] **Step 1: Implement benchmark script using CUDA events for precise timing**
- [ ] **Step 2: Run benchmark locally or on remote runner**
- [ ] **Step 3: Commit**

```bash
git add 03-cuda-llm-kernels/paged_bench.py
git commit -m "feat(paged-attn): add microbenchmark measuring latency and bandwidth across context lengths"
```

---

### Task 6: Vast.ai Deployment & Live Inference Integration

**Files:**
- Create: `scripts/compile_paged_ops.sh`
- Create: `scripts/run_vast_bench.sh`
- Modify: `03-cuda-llm-kernels/side_by_side_gen.py` (add PagedAttention mode)

- [ ] **Step 1: Write Linux compilation wrapper for sm_80 (A100), sm_89 (4090), sm_120**
- [ ] **Step 2: Write automated benchmark run script reporting NCU hardware metrics**
- [ ] **Step 3: Commit and Push**

```bash
git add scripts/compile_paged_ops.sh scripts/run_vast_bench.sh
git commit -m "feat(paged-attn): add Vast.ai build and profiling automation scripts"
```
