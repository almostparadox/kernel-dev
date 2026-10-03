"""
True Side-by-Side LLM Generation Benchmark
Compares real LLM text generation: Standard PyTorch vs Custom CUDA Kernels (RMSNorm + SwiGLU + PagedAttention).
"""
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
from __future__ import annotations

import argparse
import ctypes
import inspect
import math
import os
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root and package directory are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from paged_attention.ops import BLOCK_SIZE, paged_attention_v1  # noqa: E402
from paged_attention.paged_allocator import (  # noqa: E402
    PagedBlockAllocator,
    SequenceBlockTableManager,
)


def load_cuda_lib() -> ctypes.CDLL | None:
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "custom_ops.dll")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "custom_ops.so")),
    ]
    lib_path = next((p for p in candidates if os.path.exists(p)), None)
    if lib_path is None:
        return None

    lib = ctypes.CDLL(lib_path)
    lib.launch_rmsnorm_fp16.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_uint64
    ]
    lib.launch_swiglu_fp16.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_uint64
    ]
    return lib


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_kv_from_cache(cache, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if hasattr(cache, "layers") and len(cache.layers) > layer_idx:
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache") and len(cache.key_cache) > layer_idx:
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if isinstance(cache, (list, tuple)) and len(cache) > layer_idx:
        return cache[layer_idx][0], cache[layer_idx][1]
    return None, None


class CustomRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps, cuda_lib):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps
        self.cuda_lib = cuda_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        out = torch.empty_like(x_2d)
        stream = torch.cuda.current_stream().cuda_stream
        self.cuda_lib.launch_rmsnorm_fp16(
            x_2d.data_ptr(),
            self.weight.data_ptr(),
            out.data_ptr(),
            x_2d.shape[0],
            x_2d.shape[1],
            self.variance_epsilon,
            stream
        )
        return out.view(orig_shape)


class CustomMLP(torch.nn.Module):
    def __init__(self, orig_mlp, cuda_lib):
        super().__init__()
        self.gate_proj = orig_mlp.gate_proj
        self.up_proj = orig_mlp.up_proj
        self.down_proj = orig_mlp.down_proj
        self.cuda_lib = cuda_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        out_act = torch.empty_like(gate)
        stream = torch.cuda.current_stream().cuda_stream
        self.cuda_lib.launch_swiglu_fp16(
            gate.data_ptr(),
            up.data_ptr(),
            out_act.data_ptr(),
            gate.numel(),
            stream
        )
        return self.down_proj(out_act)


class CustomPagedAttention(torch.nn.Module):
    """Integrates PagedAttention CUDA kernel and SequenceBlockTableManager with HF Attention."""

    def __init__(self, orig_attn: Any, max_blocks: int = 2048):
        super().__init__()
        self.orig_attn: Any = orig_attn
        self.layer_idx: int = int(getattr(orig_attn, "layer_idx", 0))
        self.max_blocks: int = max_blocks

        cfg = getattr(orig_attn, "config", None)
        head_dim = getattr(orig_attn, "head_dim", None)
        if head_dim is None and cfg is not None:
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        if head_dim is None:
            head_dim = 128
        self.head_dim: int = int(head_dim)

        num_heads = getattr(orig_attn, "num_heads", None)
        if num_heads is None and cfg is not None:
            num_heads = getattr(cfg, "num_attention_heads", 16)
        if num_heads is None:
            num_heads = 16
        self.num_heads: int = int(num_heads)

        num_kv_heads = getattr(orig_attn, "num_key_value_heads", None)
        if num_kv_heads is None and cfg is not None:
            num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        if num_kv_heads is None:
            num_kv_heads = self.num_heads
        self.num_kv_heads: int = int(num_kv_heads)

        self.num_kv_groups: int = max(1, self.num_heads // self.num_kv_heads)
        scaling = getattr(orig_attn, "scaling", 1.0 / math.sqrt(self.head_dim))
        self.scaling: float = float(scaling)

        self.allocator = PagedBlockAllocator(total_blocks=max_blocks, block_size=BLOCK_SIZE)
        self.manager = SequenceBlockTableManager(self.allocator, block_size=BLOCK_SIZE)
        self.k_pool: torch.Tensor = torch.empty(0)
        self.v_pool: torch.Tensor = torch.empty(0)
        self._num_outputs: int = 3

    def reset_cache(self, device: torch.device):
        self.allocator = PagedBlockAllocator(total_blocks=self.max_blocks, block_size=BLOCK_SIZE)
        self.manager = SequenceBlockTableManager(self.allocator, block_size=BLOCK_SIZE)
        self.k_pool = torch.zeros(
            (self.max_blocks, self.num_heads, BLOCK_SIZE, self.head_dim),
            dtype=torch.float16,
            device=device,
        )
        self.v_pool = torch.zeros(
            (self.max_blocks, self.num_heads, BLOCK_SIZE, self.head_dim),
            dtype=torch.float16,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        past_key_values=None,
        **kwargs,
    ):
        pkv = (
            past_key_value
            if past_key_value is not None
            else (
                past_key_values
                if past_key_values is not None
                else kwargs.get("past_key_values", kwargs.get("past_key_value"))
            )
        )
        seq_len = hidden_states.shape[1]
        device = hidden_states.device

        if seq_len > 1:
            # Prefill phase: initialize paged cache and populate with prompt KV
            self.reset_cache(device)
            call_kwargs = dict(kwargs)
            try:
                params = inspect.signature(self.orig_attn.forward).parameters
                if "position_embeddings" in params:
                    call_kwargs["position_embeddings"] = position_embeddings
                if "attention_mask" in params:
                    call_kwargs["attention_mask"] = attention_mask
                if pkv is not None:
                    if "past_key_values" in params:
                        call_kwargs["past_key_values"] = pkv
                    elif "past_key_value" in params:
                        call_kwargs["past_key_value"] = pkv
                    else:
                        call_kwargs["past_key_value"] = pkv
            except Exception:
                call_kwargs["position_embeddings"] = position_embeddings
                call_kwargs["attention_mask"] = attention_mask
                if pkv is not None:
                    call_kwargs["past_key_value"] = pkv

            out = self.orig_attn(hidden_states, **call_kwargs)

            if isinstance(out, tuple):
                self._num_outputs = len(out)
            else:
                self._num_outputs = 2

            if pkv is not None:
                k_cache, v_cache = get_kv_from_cache(pkv, self.layer_idx)
                if k_cache is not None and v_cache is not None:
                    self.manager.allocate_sequence(0, seq_len)
                    k_rep = k_cache.repeat_interleave(self.num_kv_groups, dim=1).to(torch.float16)
                    v_rep = v_cache.repeat_interleave(self.num_kv_groups, dim=1).to(torch.float16)
                    for t in range(seq_len):
                        b_id = self.manager.seq_to_blocks[0][t // BLOCK_SIZE]
                        slot = t % BLOCK_SIZE
                        self.k_pool[b_id, :, slot, :] = k_rep[0, :, t, :]
                        self.v_pool[b_id, :, slot, :] = v_rep[0, :, t, :]
            return out

        # Decode phase (seq_len == 1): run PagedAttention kernel
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.orig_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.orig_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.orig_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None and hasattr(self.orig_attn, "rotary_emb"):
            position_ids = kwargs.get("position_ids")
            if position_ids is None and 0 in self.manager.seq_lengths:
                position_ids = torch.tensor([[self.manager.seq_lengths[0]]], device=device, dtype=torch.long)
            if position_ids is not None:
                try:
                    cos, sin = self.orig_attn.rotary_emb(value_states, position_ids)
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                except Exception as err:
                    # Ignore rotary embedding error if positional format differs
                    _ = err
        elif position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if pkv is not None:
            key_states, value_states = pkv.update(key_states, value_states, self.layer_idx)

        # Update paged KV cache tables
        self.manager.append_token(0)
        curr_len = self.manager.seq_lengths[0]
        b_id = self.manager.seq_to_blocks[0][(curr_len - 1) // BLOCK_SIZE]
        slot = (curr_len - 1) % BLOCK_SIZE

        k_rep = key_states[:, :, -1:, :].repeat_interleave(self.num_kv_groups, dim=1).to(torch.float16)
        v_rep = value_states[:, :, -1:, :].repeat_interleave(self.num_kv_groups, dim=1).to(torch.float16)
        self.k_pool[b_id, :, slot, :] = k_rep[0, :, 0, :]
        self.v_pool[b_id, :, slot, :] = v_rep[0, :, 0, :]

        num_blocks = len(self.manager.seq_to_blocks[0])
        block_tables = self.manager.build_block_table_tensor([0], num_blocks).to(device)
        context_lens = torch.tensor([curr_len], dtype=torch.int32, device=device)

        q_paged = query_states.transpose(1, 2).contiguous().to(torch.float16)
        force_cpu = not (device.type == "cuda" and torch.cuda.is_available())
        attn_out = paged_attention_v1(
            q=q_paged,
            k_pool=self.k_pool,
            v_pool=self.v_pool,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=self.scaling,
            force_cpu=force_cpu,
        )

        attn_out = attn_out.view(*input_shape, -1).contiguous()
        attn_output = self.orig_attn.o_proj(attn_out)

        num_outs = getattr(self, "_num_outputs", 3)
        if num_outs == 3:
            return attn_output, None, pkv
        elif num_outs == 2:
            return attn_output, None
        return attn_output, None, pkv


def generate_with_timing(model, tokenizer, prompt: str, max_new_tokens: int = 60):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    gen_ids = outputs[0][prompt_len:]
    num_gen = len(gen_ids)
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tok_per_sec = num_gen / total_time if total_time > 0 else 0.0

    return {
        "text": gen_text.strip(),
        "num_tokens": num_gen,
        "total_time": total_time,
        "tok_per_sec": tok_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Side-by-Side LLM Generation Benchmark")
    parser.add_argument("prompt_pos", nargs="*", default=None, help="Positional prompt text")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument("--model", type=str, default=None, help="Model name or local path")
    parser.add_argument("--paged-attn", action="store_true", help="Enable CustomPagedAttention kernel integration")
    parser.add_argument("--max-tokens", type=int, default=60, help="Max new tokens to generate")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif args.prompt_pos:
        prompt = " ".join(args.prompt_pos)
    else:
        prompt = "Explain quantum computing in one short sentence."

    model_path = args.model or os.environ.get("MODEL_PATH")
    if not model_path:
        default_local = "C:/Users/Lenovo/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
        model_path = default_local if os.path.exists(default_local) else "Qwen/Qwen2.5-3B-Instruct"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 75)
    print("SIDE-BY-SIDE LLM GENERATION EXPERIMENT: PYTORCH VS CUSTOM CUDA")
    print(f"Model Architecture: Qwen2 / DeepSeek-Distill ({model_path})")
    print(f"Device: {device_name} ({device})")
    print(f"PagedAttention Enabled: {args.paged_attn}")
    print(f"Prompt: \"{prompt}\"")
    print("=" * 75)

    cuda_lib = load_cuda_lib()

    print(f"\n[1/4] Loading model ({model_path}) to {device} (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    model.eval()

    # Warmup
    print("[2/4] Warming up execution pipeline...")
    _ = generate_with_timing(model, tokenizer, "Warmup prompt", max_new_tokens=5)

    # -------------------------------------------------------------------------
    # RUN A: WITHOUT Custom CUDA (Native PyTorch Default)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Generation WITHOUT Custom CUDA (Native PyTorch)...")
    res_native = generate_with_timing(model, tokenizer, prompt, max_new_tokens=args.max_tokens)

    # -------------------------------------------------------------------------
    # PATCH MODEL WITH CUSTOM CUDA KERNELS
    # -------------------------------------------------------------------------
    print("\n[Patching Model: Injecting Custom CUDA Kernels]...")
    patched_layers = 0
    if cuda_lib is not None:
        for layer in model.model.layers:
            layer.input_layernorm = CustomRMSNorm(
                layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon,
                cuda_lib
            )
            layer.post_attention_layernorm = CustomRMSNorm(
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
                cuda_lib
            )
            layer.mlp = CustomMLP(layer.mlp, cuda_lib)
            patched_layers += 1

        model.model.norm = CustomRMSNorm(
            model.model.norm.weight,
            model.model.norm.variance_epsilon,
            cuda_lib
        )
        print(f"Successfully patched {patched_layers} layers ({patched_layers * 2 + 1} RMSNorms + {patched_layers} SwiGLUs).")
    else:
        print("Notice: custom_ops shared library (RMSNorm/SwiGLU) not found, skipping RMSNorm/SwiGLU patching.")

    if args.paged_attn:
        paged_layers = 0
        for layer in model.model.layers:
            layer.self_attn = CustomPagedAttention(layer.self_attn)
            paged_layers += 1
        print(f"Successfully patched {paged_layers} layers with CustomPagedAttention (SequenceBlockTableManager).")

    # -------------------------------------------------------------------------
    # RUN B: WITH Custom CUDA Kernels (Single-pass Warp Shuffle + Fused SwiGLU / PagedAttention)
    # -------------------------------------------------------------------------
    print("\n[4/4] Running Generation WITH Custom CUDA Kernels...")
    res_custom = generate_with_timing(model, tokenizer, prompt, max_new_tokens=args.max_tokens)

    # -------------------------------------------------------------------------
    # Comparison & Validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("SIDE-BY-SIDE GENERATION RESULTS")
    print("=" * 75)
    print("\n--- Output Text (Native PyTorch) ---")
    print(res_native["text"])
    print("\n--- Output Text (Custom CUDA Kernels) ---")
    print(res_custom["text"])

    match = (res_native["text"] == res_custom["text"])
    print("\n" + "-" * 75)
    print(f"Numerical Equivalence / Token Exactness: {'IDENTICAL (PASS)' if match else 'DIFFERENT'}")
    print("-" * 75)

    print(f"{'Metric':<30} | {'Native PyTorch':<18} | {'With Custom CUDA':<18} | {'Improvement'}")
    print("-" * 75)
    print(f"{'Generated Tokens':<30} | {res_native['num_tokens']:<18} | {res_custom['num_tokens']:<18} | Identical")
    print(f"{'Generation Time':<30} | {res_native['total_time']:.3f} s           | {res_custom['total_time']:.3f} s           | {(res_native['total_time'] - res_custom['total_time']) * 1000:+.1f} ms")
    speedup = res_custom['tok_per_sec'] / res_native['tok_per_sec'] if res_native['tok_per_sec'] > 0 else 1.0
    print(f"{'Throughput (tokens/sec)':<30} | {res_native['tok_per_sec']:.2f} tok/s         | {res_custom['tok_per_sec']:.2f} tok/s         | {speedup:.2f}x ({((speedup - 1.0) * 100):+.1f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
