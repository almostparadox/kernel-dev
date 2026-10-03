"""Unit verification suite for PagedAttention decode vs PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn.functional as F

from paged_attention.ops import (
    BLOCK_SIZE,
    paged_attention_reference,
    paged_attention_v1,
)
from paged_attention.paged_allocator import (
    PagedBlockAllocator,
    SequenceBlockTableManager,
)


def _setup_paged_kv_data(
    batch_lens: List[int],
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    scramble_blocks: bool = False,
):
    """Helper to populate physical KV pool and build ground-truth unpaged tensors."""
    batch_size = len(batch_lens)
    allocator = PagedBlockAllocator(total_blocks=256, block_size=BLOCK_SIZE)

    if scramble_blocks:
        # Pre-allocate and release dummy blocks to scramble free list
        dummy = allocator.allocate(20)
        allocator.free(dummy[::2])

    mgr = SequenceBlockTableManager(allocator, block_size=BLOCK_SIZE)

    k_pool = torch.zeros(
        (allocator.total_blocks, num_heads, BLOCK_SIZE, head_dim),
        dtype=dtype,
        device=device,
    )
    v_pool = torch.zeros(
        (allocator.total_blocks, num_heads, BLOCK_SIZE, head_dim),
        dtype=dtype,
        device=device,
    )

    k_unpaged: List[torch.Tensor] = []
    v_unpaged: List[torch.Tensor] = []

    for sid, length in enumerate(batch_lens):
        mgr.allocate_sequence(sid, length)
        k_seq = torch.randn(num_heads, length, head_dim, dtype=dtype, device=device)
        v_seq = torch.randn(num_heads, length, head_dim, dtype=dtype, device=device)
        k_unpaged.append(k_seq)
        v_unpaged.append(v_seq)

        blocks = mgr.seq_to_blocks[sid]
        for b_idx, p_block in enumerate(blocks):
            start = b_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, length)
            tok_count = end - start
            if tok_count > 0:
                k_pool[p_block, :, :tok_count, :] = k_seq[:, start:end, :]
                v_pool[p_block, :, :tok_count, :] = v_seq[:, start:end, :]

    max_blocks = max((l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in batch_lens)
    block_tables = mgr.build_block_table_tensor(
        list(range(batch_size)), max_blocks
    ).to(device)
    context_lens = torch.tensor(batch_lens, dtype=torch.int32, device=device)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)

    return q, k_pool, v_pool, block_tables, context_lens, k_unpaged, v_unpaged


def _compute_unpaged_sdpa_reference(
    q: torch.Tensor,
    k_unpaged: List[torch.Tensor],
    v_unpaged: List[torch.Tensor],
    batch_lens: List[int],
    scale: float,
) -> torch.Tensor:
    """Compute PyTorch scaled_dot_product_attention per-sequence on unpaged tensors."""
    batch_size, num_heads, head_dim = q.shape[:3]
    ref_out = torch.zeros(
        (batch_size, num_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )

    for sid, length in enumerate(batch_lens):
        if length <= 0:
            continue
        # q: [1, num_heads, 1, head_dim]
        q_i = q[sid : sid + 1].unsqueeze(2)
        # k, v: [1, num_heads, length, head_dim]
        k_i = k_unpaged[sid].unsqueeze(0)
        v_i = v_unpaged[sid].unsqueeze(0)

        # PyTorch native scaled_dot_product_attention
        sdpa = F.scaled_dot_product_attention(q_i, k_i, v_i, scale=scale)
        ref_out[sid] = sdpa.squeeze(2).squeeze(0)

    return ref_out


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 4, 8])
def test_paged_attention_vs_sdpa_variable_lens(head_dim: int, num_heads: int):
    """Test variable sequence lengths [17, 35, 64] across batch vs PyTorch SDPA."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    batch_lens = [17, 35, 64]
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    out = paged_attention_v1(
        q, k_pool, v_pool, block_tables, context_lens, scale=scale, force_cpu=True
    )
    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )

    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()

    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3 for d={head_dim}, h={num_heads}"
    assert mse < 1e-5, f"MSE {mse} >= 1e-5 for d={head_dim}, h={num_heads}"


@pytest.mark.parametrize("batch_lens", [[16, 32, 64], [1, 15, 16]])
def test_paged_attention_block_boundaries(batch_lens: List[int]):
    """Test sequences exactly at block boundaries (multiples of 16) and short lengths."""
    torch.manual_seed(123)
    device = torch.device("cpu")
    head_dim = 64
    num_heads = 4
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    out = paged_attention_v1(
        q, k_pool, v_pool, block_tables, context_lens, scale=scale, force_cpu=True
    )
    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )

    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()

    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3 for lens {batch_lens}"
    assert mse < 1e-5, f"MSE {mse} >= 1e-5 for lens {batch_lens}"


def test_paged_attention_scattered_physical_blocks():
    """Test non-contiguous physical block allocations in the block table."""
    torch.manual_seed(999)
    device = torch.device("cpu")
    batch_lens = [25, 49, 13]
    head_dim = 128
    num_heads = 4
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(
        batch_lens, num_heads, head_dim, device, scramble_blocks=True
    )

    out = paged_attention_v1(
        q, k_pool, v_pool, block_tables, context_lens, scale=scale, force_cpu=True
    )
    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )

    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()

    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3 with scrambled blocks"
    assert mse < 1e-5, f"MSE {mse} >= 1e-5 with scrambled blocks"


def test_paged_attention_custom_scale():
    """Test custom scaling factor passed to paged_attention_v1."""
    torch.manual_seed(77)
    device = torch.device("cpu")
    batch_lens = [17, 32]
    head_dim = 64
    num_heads = 2
    custom_scale = 0.35

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    out = paged_attention_v1(
        q, k_pool, v_pool, block_tables, context_lens, scale=custom_scale, force_cpu=True
    )
    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=custom_scale
    )

    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()

    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3"
    assert mse < 1e-5, f"MSE {mse} >= 1e-5"


def test_paged_attention_preallocated_out():
    """Test in-place output accumulation with preallocated out tensor."""
    torch.manual_seed(88)
    device = torch.device("cpu")
    batch_lens = [19, 33]
    head_dim = 64
    num_heads = 2
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    out_buf = torch.empty_like(q)
    ret = paged_attention_v1(
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        scale=scale,
        out=out_buf,
        force_cpu=True,
    )

    assert ret is out_buf, "Function should return the same preallocated tensor"

    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )
    max_err = (out_buf - ref).abs().max().item()
    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3"


def test_paged_attention_zero_length_sequence():
    """Test batch containing a sequence with context_len = 0."""
    torch.manual_seed(55)
    device = torch.device("cpu")
    batch_lens = [0, 20, 0]
    head_dim = 64
    num_heads = 2
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    out = paged_attention_v1(
        q, k_pool, v_pool, block_tables, context_lens, scale=scale, force_cpu=True
    )

    # Empty sequences should have all zeros in output
    assert (out[0] == 0.0).all().item()
    assert (out[2] == 0.0).all().item()

    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )
    max_err = (out[1] - ref[1]).abs().max().item()
    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3 on active sequence"


def test_paged_attention_4d_query_input():
    """Test 4D query tensor shape [batch_size, 1, num_heads, head_dim]."""
    torch.manual_seed(10)
    device = torch.device("cpu")
    batch_lens = [22, 11]
    head_dim = 64
    num_heads = 4
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    q_4d = q.unsqueeze(1)  # [batch_size, 1, num_heads, head_dim]

    out = paged_attention_v1(
        q_4d, k_pool, v_pool, block_tables, context_lens, scale=scale, force_cpu=True
    )
    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )

    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-3, f"Max error {max_err} >= 1e-3"


def test_input_validation_errors():
    """Test that invalid inputs raise ValueError or TypeError."""
    device = torch.device("cpu")
    batch_lens = [16]
    head_dim = 64
    num_heads = 2

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        _,
        _,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    # 1. Invalid dtype
    with pytest.raises(ValueError, match="must be float16"):
        paged_attention_v1(q.float(), k_pool, v_pool, block_tables, context_lens)

    with pytest.raises(ValueError, match="must be int32"):
        paged_attention_v1(q, k_pool, v_pool, block_tables.long(), context_lens)

    # 2. Unsupported head_dim
    bad_q = torch.randn(1, num_heads, 96, dtype=torch.float16)
    bad_k = torch.randn(1, num_heads, BLOCK_SIZE, 96, dtype=torch.float16)
    bad_v = torch.randn(1, num_heads, BLOCK_SIZE, 96, dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim must be 64, 128, or 256"):
        paged_attention_v1(bad_q, bad_k, bad_v, block_tables, context_lens)

    # 3. Non-contiguous input
    non_contig_q = q.transpose(0, 1).contiguous().transpose(0, 1)
    # create truly non-contiguous tensor
    full_t = torch.randn(2, 2, num_heads, head_dim, dtype=torch.float16)
    sliced_q = full_t[:, 0, :, :]
    if not sliced_q.is_contiguous():
        with pytest.raises(ValueError, match="must be contiguous"):
            paged_attention_v1(sliced_q, k_pool, v_pool, block_tables, context_lens)

    # 4. Out of bounds block size
    bad_k_block = torch.randn(1, num_heads, 32, head_dim, dtype=torch.float16)
    bad_v_block = torch.randn(1, num_heads, 32, head_dim, dtype=torch.float16)
    with pytest.raises(ValueError, match="BLOCK_SIZE"):
        paged_attention_v1(q, bad_k_block, bad_v_block, block_tables, context_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_paged_attention_cuda_parity():
    """Test numerical parity on CUDA device when GPU is available."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    batch_lens = [17, 35, 64]
    head_dim = 128
    num_heads = 4
    scale = 1.0 / math.sqrt(head_dim)

    (
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        k_unpaged,
        v_unpaged,
    ) = _setup_paged_kv_data(batch_lens, num_heads, head_dim, device)

    out = paged_attention_v1(
        q, k_pool, v_pool, block_tables, context_lens, scale=scale
    )
    ref = _compute_unpaged_sdpa_reference(
        q, k_unpaged, v_unpaged, batch_lens, scale=scale
    )

    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()

    assert max_err < 1e-3, f"CUDA Max error {max_err} >= 1e-3"
    assert mse < 1e-5, f"CUDA MSE {mse} >= 1e-5"
