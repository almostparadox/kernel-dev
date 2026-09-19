import ctypes

import torch
import torch.nn.functional as F

dll_path = "D:/Active-Projects/learn-kernel-dev/03-cuda-llm-kernels/custom_ops.dll"
lib = ctypes.CDLL(dll_path)

lib.launch_rmsnorm_fp16.argtypes = [
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_uint64
]
lib.launch_swiglu_fp16.argtypes = [
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
    ctypes.c_int, ctypes.c_uint64
]

print("Testing RMSNorm FP16...")
rows, cols = 16, 2048
x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
w = torch.ones(cols, device="cuda", dtype=torch.float16)
out_custom = torch.empty_like(x)

# Native PyTorch RMSNorm reference
ref = w * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).half()

# Custom CUDA call
stream = torch.cuda.current_stream().cuda_stream
lib.launch_rmsnorm_fp16(
    x.data_ptr(), w.data_ptr(), out_custom.data_ptr(),
    rows, cols, 1e-6, stream
)
torch.cuda.synchronize()

err = (out_custom - ref).abs().max().item()
print(f"RMSNorm Max Diff vs PyTorch: {err:.6f} {'(PASS)' if err < 1e-2 else '(FAIL)'}")

print("Testing SwiGLU FP16...")
gate = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
up = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
out_swiglu = torch.empty_like(gate)

ref_swiglu = (F.silu(gate.float()) * up.float()).half()
lib.launch_swiglu_fp16(
    gate.data_ptr(), up.data_ptr(), out_swiglu.data_ptr(),
    rows * cols, stream
)
torch.cuda.synchronize()

err_swiglu = (out_swiglu - ref_swiglu).abs().max().item()
print(f"SwiGLU Max Diff vs PyTorch: {err_swiglu:.6f} {'(PASS)' if err_swiglu < 1e-2 else '(FAIL)'}")
