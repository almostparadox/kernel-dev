import math
import pytest
import torch
import torch.nn as nn

from paged_attention.paged_allocator import PagedBlockAllocator, SequenceBlockTableManager
from paged_attention.ops import BLOCK_SIZE, paged_attention_v1


class MockAttentionLayer(nn.Module):
    """Mock HuggingFace Attention layer with q/k/v/o projections."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.layer_idx = 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, past_key_values=None, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard SDPA
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.float()).to(q.dtype)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return self.o_proj(out), (k, v)


def test_custom_paged_attention_lifecycle():
    """Verify CustomPagedAttention correctly manages KV cache and matches unpaged attention."""
    torch.manual_seed(42)
    hidden_size = 256
    num_heads = 4
    head_dim = 64
    mock_attn = MockAttentionLayer(
        hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim
    ).to(torch.float16)
    mock_attn.eval()

    # Import CustomPagedAttention from side_by_side_gen
    from side_by_side_gen import CustomPagedAttention

    paged_attn = CustomPagedAttention(mock_attn, max_blocks=64)

    # 1. Prefill step: prompt of length 8
    prompt_len = 8
    hidden_prompt = torch.randn(1, prompt_len, hidden_size, dtype=torch.float16)
    out_native_prefill, kv_cache = mock_attn(hidden_prompt)

    out_paged_prefill, _ = paged_attn(
        hidden_prompt,
        past_key_values=[kv_cache],
    )

    # Prefill outputs must match native attention
    assert torch.allclose(out_native_prefill, out_paged_prefill, atol=1e-3)
    assert paged_attn.manager.seq_lengths[0] == prompt_len

    # 2. Decode step: 1 token
    token_hidden = torch.randn(1, 1, hidden_size, dtype=torch.float16)

    # Compute expected result via full unpaged attention across prompt + new token
    full_hidden = torch.cat([hidden_prompt, token_hidden], dim=1)
    full_native_out, _ = mock_attn(full_hidden)
    expected_token_out = full_native_out[:, -1:, :]

    # Run paged decode
    out_paged_token, _ = paged_attn(token_hidden)

    # Compare decode token output
    max_diff = (out_paged_token - expected_token_out).abs().max().item()
    assert max_diff < 1e-3, f"Paged decode diff {max_diff} exceeds tolerance 1e-3"
    assert paged_attn.manager.seq_lengths[0] == prompt_len + 1
