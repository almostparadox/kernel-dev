#!/usr/bin/env bash
set -euo pipefail

# compile_paged_ops.sh
# Compiles paged attention CUDA kernels into paged_attention.so for Linux / Vast.ai

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAGED_DIR="$REPO_ROOT/03-cuda-llm-kernels/paged_attention"
OUTPUT_SO="$PAGED_DIR/paged_attention.so"

echo "======================================================================"
echo "Compiling PagedAttention & Split-KV Shared Library"
echo "======================================================================"

# 1. Locate CUDA Toolkit & nvcc
NVCC_BIN=""
if command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
    export CUDA_HOME="/usr/local/cuda"
else
    # Check for versioned /usr/local/cuda-* paths
    for candidate in /usr/local/cuda-*/bin/nvcc; do
        if [[ -x "$candidate" ]]; then
            NVCC_BIN="$candidate"
            export CUDA_HOME="$(dirname "$(dirname "$candidate")")"
            break
        fi
    done
fi

if [[ -z "$NVCC_BIN" ]]; then
    echo "ERROR: nvcc compiler not found in PATH or standard CUDA directories." >&2
    echo "Please set CUDA_HOME or add nvcc to PATH (e.g. export PATH=/usr/local/cuda/bin:\$PATH)." >&2
    exit 1
fi

echo "[1/3] Using nvcc compiler: $NVCC_BIN"
"$NVCC_BIN" --version | head -n 4

# 2. Determine target GPU architecture
GENCODE_FLAGS=""
FORCE_ALL_ARCH=0

TARGET_ARCH="${CUDA_ARCH:-}"
for arg in "$@"; do
    case "$arg" in
        --all-arch)
            FORCE_ALL_ARCH=1
            ;;
        --arch=*)
            TARGET_ARCH="${arg#*=}"
            ;;
    esac
done

if [[ -n "$TARGET_ARCH" ]]; then
    echo "[2/3] Using manually specified architecture: $TARGET_ARCH"
    ARCH_CLEAN="${TARGET_ARCH#compute_}"
    ARCH_CLEAN="${ARCH_CLEAN#sm_}"
    GENCODE_FLAGS="-gencode arch=compute_${ARCH_CLEAN},code=sm_${ARCH_CLEAN}"
elif [[ "$FORCE_ALL_ARCH" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "[2/3] Querying GPU architecture via nvidia-smi..."
    DETECTED_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
    if [[ -n "$DETECTED_CAP" ]]; then
        ARCH_NUM="${DETECTED_CAP//./}"
        echo "Detected GPU Compute Capability: $DETECTED_CAP -> sm_${ARCH_NUM}"
        GENCODE_FLAGS="-gencode arch=compute_${ARCH_NUM},code=sm_${ARCH_NUM}"
    fi
fi

# Fallback to multi-architecture compilation: sm_80, sm_89, sm_90
if [[ -z "$GENCODE_FLAGS" ]]; then
    echo "[2/3] No single active GPU detected; building multi-architecture binary:"
    echo "      Targets: sm_80 (A100), sm_89 (RTX 4090), sm_90 (H100)"
    GENCODE_FLAGS="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90"

    # Check if nvcc supports compute_120 (CUDA 12.8+)
    if "$NVCC_BIN" --help 2>&1 | grep -q "compute_120"; then
        echo "      Adding sm_120 target..."
        GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_120,code=sm_120"
    fi
fi

# 3. Source files
SRC_DECODE="$PAGED_DIR/paged_decode.cu"
SRC_SPLITKV="$PAGED_DIR/paged_split_kv.cu"

for src in "$SRC_DECODE" "$SRC_SPLITKV"; do
    if [[ ! -f "$src" ]]; then
        echo "ERROR: Missing source file: $src" >&2
        exit 1
    fi
done

echo "[3/3] Compiling shared library: $OUTPUT_SO"
echo "Command: $NVCC_BIN -O3 -std=c++17 --use_fast_math -Xcompiler -fPIC --shared $GENCODE_FLAGS -I\"$PAGED_DIR\" \"$SRC_DECODE\" \"$SRC_SPLITKV\" -o \"$OUTPUT_SO\""

"$NVCC_BIN" \
    -O3 \
    -std=c++17 \
    --use_fast_math \
    -Xcompiler -fPIC \
    --shared \
    $GENCODE_FLAGS \
    -I"$PAGED_DIR" \
    "$SRC_DECODE" \
    "$SRC_SPLITKV" \
    -o "$OUTPUT_SO"

echo "======================================================================"
echo "Successfully compiled: $OUTPUT_SO"
if [[ -f "$OUTPUT_SO" ]]; then
    ls -lh "$OUTPUT_SO"
fi
echo "======================================================================"
