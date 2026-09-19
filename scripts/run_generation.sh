#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/mnt/c/Users/Lenovo/AppData/Local/Programs/Python/Python310/python.exe "$REPO_ROOT/03-cuda-llm-kernels/generate_bench.py" "$@"
