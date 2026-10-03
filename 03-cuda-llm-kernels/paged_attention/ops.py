"""PagedAttention operations: Python wrapper, C++ extension loader, ctypes bindings, and golden CPU reference."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import torch

BLOCK_SIZE = 16

_CPP_MODULE = None
_CTYPES_LAUNCHER = None


def load_cpp_extension(build_directory: str | None = None, verbose: bool = False):
    """Load or JIT-compile the PyTorch C++ extension for PagedAttention."""
    global _CPP_MODULE
    if _CPP_MODULE is not None:
        return _CPP_MODULE

    # 1. Check if already built and importable
    try:
        import paged_ops  # type: ignore

        _CPP_MODULE = paged_ops
        return _CPP_MODULE
    except ImportError:
        pass

    # 2. Try JIT-compilation via torch.utils.cpp_extension if CUDA is available
    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load

        src_dir = Path(__file__).resolve().parent
        sources = [
            str(src_dir / "paged_ops.cpp"),
            str(src_dir / "paged_decode.cu"),
        ]
        all_exist = all(Path(s).exists() for s in sources)
        if not all_exist:
            return None

        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
        ]
        extra_cflags = ["-O3", "-std=c++17", "-fPIC"]

        _CPP_MODULE = load(
            name="paged_ops",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=build_directory,
            verbose=verbose,
        )
        return _CPP_MODULE
    except Exception:
        return None


def load_ctypes_lib(path: str | None = None) -> ctypes.CDLL | None:
    """Load precompiled PagedAttention shared library via ctypes."""
    global _CTYPES_LAUNCHER
    if _CTYPES_LAUNCHER is not None and path is None:
        return _CTYPES_LAUNCHER

    search_paths = []
    if path is not None:
        search_paths.append(Path(path))

    curr_dir = Path(__file__).resolve().parent
    repo_root = curr_dir.parent

    for name in [
        "paged_ops.so",
        "paged_attention.so",
        "libpaged_ops.so",
        "custom_ops.dll",
        "custom_ops.so",
    ]:
        search_paths.append(curr_dir / name)
        search_paths.append(repo_root / name)

    lib = None
    for candidate in search_paths:
        if candidate.exists():
            try:
                lib = ctypes.CDLL(str(candidate))
                break
            except OSError:
                continue

    if lib is None:
        return None

    launcher = getattr(lib, "launch_paged_attention_v1", None) or getattr(
        lib, "paged_attention_v1_launcher", None
    )
    if launcher is not None:
        launcher.argtypes = [
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k_pool
            ctypes.c_void_p,  # v_pool
            ctypes.c_void_p,  # block_tables
            ctypes.c_void_p,  # context_lens
            ctypes.c_int,  # max_blocks_per_seq
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # num_heads
            ctypes.c_int,  # head_dim
            ctypes.c_float,  # scale
            ctypes.c_void_p,  # stream
        ]
        launcher.restype = None
        _CTYPES_LAUNCHER = launcher
        return lib

    return None


def is_cuda_extension_available() -> bool:
    """Check if compiled PyTorch C++ extension is loaded or available."""
    mod = load_cpp_extension()
    return mod is not None and hasattr(mod, "paged_attention_v1")


def is_ctypes_available() -> bool:
    """Check if ctypes shared library is loaded or available."""
    global _CTYPES_LAUNCHER
    if _CTYPES_LAUNCHER is not None:
        return True
    return load_ctypes_lib() is not None


def _validate_inputs(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    """Validate shapes, dtypes, contiguous layout, and device placement."""
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q)}")
    if not isinstance(k_pool, torch.Tensor):
        raise TypeError(f"k_pool must be a torch.Tensor, got {type(k_pool)}")
    if not isinstance(v_pool, torch.Tensor):
        raise TypeError(f"v_pool must be a torch.Tensor, got {type(v_pool)}")
    if not isinstance(block_tables, torch.Tensor):
        raise TypeError(f"block_tables must be a torch.Tensor, got {type(block_tables)}")
    if not isinstance(context_lens, torch.Tensor):
        raise TypeError(f"context_lens must be a torch.Tensor, got {type(context_lens)}")

    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if not k_pool.is_contiguous():
        raise ValueError("k_pool must be contiguous")
    if not v_pool.is_contiguous():
        raise ValueError("v_pool must be contiguous")
    if not block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous")
    if not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous")

    if q.dtype != torch.float16:
        raise ValueError(f"q must be float16, got {q.dtype}")
    if k_pool.dtype != torch.float16:
        raise ValueError(f"k_pool must be float16, got {k_pool.dtype}")
    if v_pool.dtype != torch.float16:
        raise ValueError(f"v_pool must be float16, got {v_pool.dtype}")
    if block_tables.dtype != torch.int32:
        raise ValueError(f"block_tables must be int32, got {block_tables.dtype}")
    if context_lens.dtype != torch.int32:
        raise ValueError(f"context_lens must be int32, got {context_lens.dtype}")

    device = q.device
    if k_pool.device != device:
        raise ValueError(f"k_pool device {k_pool.device} != q device {device}")
    if v_pool.device != device:
        raise ValueError(f"v_pool device {v_pool.device} != q device {device}")
    if block_tables.device != device:
        raise ValueError(f"block_tables device {block_tables.device} != q device {device}")
    if context_lens.device != device:
        raise ValueError(f"context_lens device {context_lens.device} != q device {device}")

    if q.dim() == 4:
        if q.shape[1] != 1:
            raise ValueError(f"q sequence length must be 1 for decode attention, got shape {q.shape}")
        batch_size, _, num_heads, head_dim = q.shape
    elif q.dim() == 3:
        batch_size, num_heads, head_dim = q.shape
    else:
        raise ValueError(f"q must have 3 or 4 dimensions, got shape {q.shape}")

    if head_dim not in (64, 128, 256):
        raise ValueError(f"head_dim must be 64, 128, or 256, got {head_dim}")

    if k_pool.dim() != 4:
        raise ValueError(f"k_pool must be 4D [num_blocks, num_heads, block_size, head_dim], got {k_pool.shape}")
    if v_pool.dim() != 4:
        raise ValueError(f"v_pool must be 4D [num_blocks, num_heads, block_size, head_dim], got {v_pool.shape}")

    if k_pool.shape != v_pool.shape:
        raise ValueError(f"k_pool shape {k_pool.shape} != v_pool shape {v_pool.shape}")

    if k_pool.shape[1] != num_heads:
        raise ValueError(f"k_pool num_heads ({k_pool.shape[1]}) != q num_heads ({num_heads})")
    if k_pool.shape[2] != BLOCK_SIZE:
        raise ValueError(f"k_pool block_size ({k_pool.shape[2]}) != BLOCK_SIZE ({BLOCK_SIZE})")
    if k_pool.shape[3] != head_dim:
        raise ValueError(f"k_pool head_dim ({k_pool.shape[3]}) != q head_dim ({head_dim})")

    if block_tables.dim() != 2:
        raise ValueError(f"block_tables must be 2D [batch_size, max_blocks_per_seq], got {block_tables.shape}")
    if block_tables.shape[0] != batch_size:
        raise ValueError(f"block_tables batch size ({block_tables.shape[0]}) != q batch size ({batch_size})")

    if context_lens.dim() != 1:
        raise ValueError(f"context_lens must be 1D [batch_size], got {context_lens.shape}")
    if context_lens.shape[0] != batch_size:
        raise ValueError(f"context_lens batch size ({context_lens.shape[0]}) != q batch size ({batch_size})")

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.dtype != torch.float16:
            raise ValueError(f"out must be float16, got {out.dtype}")
        if out.device != device:
            raise ValueError(f"out device {out.device} != q device {device}")
        expected_shape = (batch_size, num_heads, head_dim)
        if out.shape != expected_shape:
            raise ValueError(f"out shape {out.shape} != expected shape {expected_shape}")

    return batch_size, num_heads, head_dim


def paged_attention_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Golden reference simulator for PagedAttention single-pass decode.

    Computes mathematically identical scaled dot-product attention
    by gathering physical KV cache blocks per sequence into contiguous tensors.
    Runs on CPU or CUDA.
    """
    batch_size, num_heads, head_dim = _validate_inputs(
        q, k_pool, v_pool, block_tables, context_lens, out
    )

    if scale <= 0.0:
        scale = 1.0 / math.sqrt(head_dim)

    q_sq = q.squeeze(1) if q.dim() == 4 else q

    if out is None:
        out = torch.zeros(
            (batch_size, num_heads, head_dim),
            dtype=torch.float16,
            device=q.device,
        )
    else:
        out.zero_()

    for seq_idx in range(batch_size):
        ctx_len = int(context_lens[seq_idx].item())
        if ctx_len <= 0:
            continue

        num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        p_blocks = block_tables[seq_idx, :num_blocks].tolist()

        k_blocks = k_pool[p_blocks]
        v_blocks = v_pool[p_blocks]

        # Reshape physical blocks to [num_heads, total_tokens, head_dim]
        # k_blocks shape: [num_blocks, num_heads, BLOCK_SIZE, head_dim]
        # permute(1, 0, 2, 3) -> [num_heads, num_blocks, BLOCK_SIZE, head_dim]
        k_seq = k_blocks.permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)[:, :ctx_len, :]
        v_seq = v_blocks.permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)[:, :ctx_len, :]

        # Query vector: [num_heads, 1, head_dim]
        q_seq = q_sq[seq_idx].unsqueeze(1)

        # High-precision FP32 accumulation matching CUDA kernel's online softmax
        scores = torch.matmul(q_seq.float(), k_seq.float().transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_seq.float()).to(torch.float16)

        out[seq_idx] = attn_out.squeeze(1)

    return out


def paged_attention_v1(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float = 0.0,
    out: torch.Tensor | None = None,
    force_cpu: bool = False,
) -> torch.Tensor:
    """Execute PagedAttention V1 decode attention.

    Dispatches to compiled CUDA extension / ctypes shared library if on CUDA,
    or falls back to the golden reference simulator if on CPU or forced.
    Raises RuntimeError if CUDA tensor is passed and CUDA extension is not loaded.
    """
    batch_size, num_heads, head_dim = _validate_inputs(
        q, k_pool, v_pool, block_tables, context_lens, out
    )

    if scale <= 0.0:
        scale = 1.0 / math.sqrt(head_dim)

    is_cuda = q.is_cuda or q.device.type == "cuda"

    # 1. CPU execution or explicit force_cpu
    if force_cpu or not is_cuda:
        return paged_attention_reference(
            q, k_pool, v_pool, block_tables, context_lens, scale=scale, out=out
        )

    # 2. CUDA execution: try PyTorch C++ extension
    mod = load_cpp_extension()
    if mod is not None and hasattr(mod, "paged_attention_v1"):
        return mod.paged_attention_v1(
            q, k_pool, v_pool, block_tables, context_lens, scale, out
        )

    # 3. CUDA execution: try ctypes shared library
    _ = load_ctypes_lib()
    global _CTYPES_LAUNCHER
    if _CTYPES_LAUNCHER is not None:
        q_sq = q.squeeze(1) if q.dim() == 4 else q
        if out is None:
            out = torch.empty(
                (batch_size, num_heads, head_dim),
                dtype=torch.float16,
                device=q.device,
            )
        stream = torch.cuda.current_stream(q.device).cuda_stream
        max_blocks_per_seq = block_tables.shape[1]

        _CTYPES_LAUNCHER(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(q_sq.data_ptr()),
            ctypes.c_void_p(k_pool.data_ptr()),
            ctypes.c_void_p(v_pool.data_ptr()),
            ctypes.c_void_p(block_tables.data_ptr()),
            ctypes.c_void_p(context_lens.data_ptr()),
            ctypes.c_int(max_blocks_per_seq),
            ctypes.c_int(batch_size),
            ctypes.c_int(num_heads),
            ctypes.c_int(head_dim),
            ctypes.c_float(scale),
            ctypes.c_void_p(stream),
        )
        return out

    # 4. Strict CUDA enforcement: do NOT silently fallback to CPU reference simulator
    raise RuntimeError(
        "CUDA PagedAttention extension not available. Compile via scripts/compile_paged_ops.sh or load valid paged_attention.so"
    )
