#!/usr/bin/env bash
set -euo pipefail

# run_vast_bench.sh
# Automated one-command build, test, benchmark, and profiling runner on Vast.ai

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "Vast.ai Automated PagedAttention Benchmark & Profiling Pipeline"
echo "Host: $(hostname 2>/dev/null || uname -n)"
echo "Date: $(date)"
echo "Working directory: $REPO_ROOT"
echo "======================================================================"

# Determine Python command
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "ERROR: python3 or python not found in PATH." >&2
    exit 1
fi
echo "Python binary: $($PYTHON_BIN --version 2>&1) at $(command -v "$PYTHON_BIN")"

# Step A: Compile PagedAttention kernels
echo ""
echo "----------------------------------------------------------------------"
echo "Step A: Compiling PagedAttention CUDA Kernels"
echo "----------------------------------------------------------------------"
chmod +x "$SCRIPT_DIR/compile_paged_ops.sh"
"$SCRIPT_DIR/compile_paged_ops.sh" "$@"

# Step B: Run Unit Tests
echo ""
echo "----------------------------------------------------------------------"
echo "Step B: Running PagedAttention Unit Tests"
echo "----------------------------------------------------------------------"
if command -v pytest >/dev/null 2>&1; then
    pytest 03-cuda-llm-kernels/paged_attention/ -v
else
    "$PYTHON_BIN" -m pytest 03-cuda-llm-kernels/paged_attention/ -v
fi

# Step C: Run Microbenchmarks
echo ""
echo "----------------------------------------------------------------------"
echo "Step C: Running Full Microbenchmark Suite"
echo "----------------------------------------------------------------------"
mkdir -p "$REPO_ROOT/docs"
BENCH_MD="$REPO_ROOT/docs/paged_attention_benchmark_results.md"
BENCH_CSV="$REPO_ROOT/docs/paged_attention_benchmark_results.csv"

"$PYTHON_BIN" 03-cuda-llm-kernels/paged_bench.py \
    --markdown "$BENCH_MD" \
    --csv "$BENCH_CSV"

echo ""
echo "Benchmark outputs saved to:"
echo "  - Markdown report: $BENCH_MD"
echo "  - CSV results:     $BENCH_CSV"

# Step D: NCU Profiling (Memory Bandwidth & Roofline Analysis)
echo ""
echo "----------------------------------------------------------------------"
echo "Step D: NVIDIA Nsight Compute (NCU) Hardware Profiling"
echo "----------------------------------------------------------------------"
NCU_BIN=""
if command -v ncu >/dev/null 2>&1; then
    NCU_BIN="$(command -v ncu)"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/ncu" ]]; then
    NCU_BIN="${CUDA_HOME}/bin/ncu"
elif [[ -x "/usr/local/cuda/bin/ncu" ]]; then
    NCU_BIN="/usr/local/cuda/bin/ncu"
fi

if [[ -n "$NCU_BIN" ]]; then
    echo "Found NCU profiler at: $NCU_BIN"
    NCU_REPORT_DIR="$REPO_ROOT/docs"
    NCU_REPORT_BASE="$NCU_REPORT_DIR/paged_attention_ncu_profile"

    echo "Profiling memory bandwidth and roofline metrics on PagedAttention kernels..."
    "$NCU_BIN" \
        --target-processes all \
        --kernel-name regex:paged_attention \
        --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --export "$NCU_REPORT_BASE" \
        --force-overwrite \
        "$PYTHON_BIN" 03-cuda-llm-kernels/paged_bench.py \
            --context-lens 2048,8192 \
            --dim 128 \
            --warmup 2 \
            --iters 5 || true

    echo "NCU profiling finished. Report file: ${NCU_REPORT_BASE}.ncu-rep (if generated)."
else
    echo "Notice: NVIDIA Nsight Compute ('ncu') not detected in PATH or CUDA directory."
    echo "Skipping NCU hardware profiling step."
fi

echo ""
echo "======================================================================"
echo "Vast.ai Automated Pipeline Completed Successfully!"
echo "======================================================================"
