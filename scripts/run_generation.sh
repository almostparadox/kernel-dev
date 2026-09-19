#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PY="$(wslpath -w "$REPO_ROOT/03-cuda-llm-kernels/generate_bench.py")"
/mnt/c/Users/Lenovo/AppData/Local/Programs/Python/Python310/python.exe "$TARGET_PY" "$@"
