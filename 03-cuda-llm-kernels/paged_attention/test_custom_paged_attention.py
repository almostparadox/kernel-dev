from __future__ import annotations

import math

import torch
import torch.nn as nn


class MockCache:
    """Mock HuggingFace Cache supporting .update(key_states, value_states, layer_idx)."""

    def __init__(self, k: torch.Tensor | None = None, v: torch.Tensor | None = None):
        self.k = k
        self.v = v

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None or self.v is None:
            self.k = key_states
            self.v = value_states
        else:
            self.k = torch.cat([self.k, key_states], dim=2)
            self.v = torch.cat([self.v, value_states], dim=2)
        return self.k, self.v

    @property
    def key_cache(self) -> list[torch.Tensor]:
        return [self.k] if self.k is not None else []

    @property
    def value_cache(self) -> list[torch.Tensor]:
        return [self.v] if self.v is not None else []


class MockAttentionLayer(nn.Module):
    """Mock HuggingFace Attention layer with q/k/v/o projections matching 3-tuple return."""

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: MockCache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, MockCache]:
        pkv = past_key_value if past_key_value is not None else kwargs.get("past_key_values")
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if pkv is not None:
            k, v = pkv.update(k, v, self.layer_idx)
        else:
            pkv = MockCache(k, v)

        # Standard SDPA
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.float()).to(q.dtype)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return self.o_proj(out), None, pkv


def test_custom_paged_attention_lifecycle():
    """Verify CustomPagedAttention correctly manages KV cache and matches HuggingFace 3-tuple contract."""
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

    # 1. Prefill step: prompt of length 8 with singular past_key_value
    prompt_len = 8
    hidden_prompt = torch.randn(1, prompt_len, hidden_size, dtype=torch.float16)
    cache_native = MockCache()
    out_native_prefill, _, _ = mock_attn(hidden_prompt, past_key_value=cache_native)

    cache_paged = MockCache()
    out_paged_prefill, weights_prefill, present_kv_prefill = paged_attn(
        hidden_prompt,
        past_key_value=cache_paged,
    )

    # Verify 3-tuple return unpacking and parity
    assert weights_prefill is None
    assert present_kv_prefill is cache_paged
    assert torch.allclose(out_native_prefill, out_paged_prefill, atol=1e-3)
    assert paged_attn.manager.seq_lengths[0] == prompt_len

    # 2. Decode step: 1 token with singular past_key_value
    token_hidden = torch.randn(1, 1, hidden_size, dtype=torch.float16)

    # Compute expected result via full unpaged attention across prompt + new token
    full_hidden = torch.cat([hidden_prompt, token_hidden], dim=1)
    full_native_out, _, _ = mock_attn(full_hidden)
    expected_token_out = full_native_out[:, -1:, :]

    # Run paged decode using HuggingFace DecoderLayer calling convention:
    # hidden_states, self_attn_weights, present_key_value = self.self_attn(..., past_key_value=past_key_value)
    out_paged_token, attn_weights_decode, present_kv_decode = paged_attn(
        token_hidden,
        past_key_value=cache_paged,
    )

    # Verify exact 3-tuple return unpacking
    assert attn_weights_decode is None
    assert present_kv_decode is cache_paged

    # Compare decode token output
    max_diff = (out_paged_token - expected_token_out).abs().max().item()
    assert max_diff < 1e-3, f"Paged decode diff {max_diff} exceeds tolerance 1e-3"
    assert paged_attn.manager.seq_lengths[0] == prompt_len + 1
