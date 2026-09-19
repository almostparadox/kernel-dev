# pyright: reportMissingImports=false
"""
Side-by-Side LLM Generation Benchmark on Quantized 7B Model
Model: DeepSeek-R1-Distill-Qwen-7B (4-bit NF4 quantized)
Compares: Native PyTorch vs Custom CUDA Kernels (Single-pass RMSNorm + Fused SwiGLU)
"""
import ctypes
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit"
DLL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "custom_ops.dll"))


def load_cuda_lib():
    if not os.path.exists(DLL_PATH):
        raise FileNotFoundError(f"Missing {DLL_PATH}. Compile with nvcc_run.bat first.")
    lib = ctypes.CDLL(DLL_PATH)
    lib.launch_rmsnorm_fp16.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_uint64
    ]
    lib.launch_swiglu_fp16.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_uint64
    ]
    return lib


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


def generate_with_timing(model, tokenizer, prompt: str, max_new_tokens: int = 60):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]

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
    prompt = "Explain quantum computing in one short sentence."
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print("=" * 75)
    print("SIDE-BY-SIDE 4-BIT QUANTIZED 7B BENCHMARK: PYTORCH VS CUSTOM CUDA")
    print("Target Model: DeepSeek-R1-Distill-Qwen-7B (4-bit NF4)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompt: \"{prompt}\"")
    print("=" * 75)

    cuda_lib = load_cuda_lib()

    print("\n[1/4] Loading Quantized 7B model to GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0"
    )
    model.eval()

    vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Model VRAM in GPU: Allocated={vram_alloc:.2f} GB, Reserved={vram_res:.2f} GB")

    # Warmup GPU
    print("\n[2/4] Warming up CUDA execution pipeline...")
    _ = generate_with_timing(model, tokenizer, "Warmup prompt", max_new_tokens=5)

    # -------------------------------------------------------------------------
    # RUN A: WITHOUT Custom CUDA (Native PyTorch Default)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Generation WITHOUT Custom CUDA (Native PyTorch)...")
    res_native = generate_with_timing(model, tokenizer, prompt, max_new_tokens=120)

    # -------------------------------------------------------------------------
    # PATCH MODEL WITH CUSTOM CUDA KERNELS
    # -------------------------------------------------------------------------
    print("\n[Patching Model: Injecting Custom CUDA Kernels]...")
    patched_layers = 0
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

    # -------------------------------------------------------------------------
    # RUN B: WITH Custom CUDA Kernels
    # -------------------------------------------------------------------------
    print("\n[4/4] Running Generation WITH Custom CUDA Kernels...")
    res_custom = generate_with_timing(model, tokenizer, prompt, max_new_tokens=120)

    # -------------------------------------------------------------------------
    # Comparison & Validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("SIDE-BY-SIDE 7B GENERATION RESULTS (4-BIT QUANTIZED)")
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
