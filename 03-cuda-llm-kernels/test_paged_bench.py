"""Unit tests for PagedAttention microbenchmark & profiling suite."""

from __future__ import annotations

import os
import sys

import torch

# Ensure current directory is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import paged_bench


def test_paged_bench_bandwidth_calculation():
    """Verify bandwidth calculation formula against analytical arithmetic."""
    # batch=1, heads=32, length=2048, head_dim=128
    # bytes = 2 * 1 * 32 * 2048 * 128 * 2 = 33,554,432 bytes
    # at 100 us: (33554432 / 1e9) / (100 * 1e-6) = 335.54432 GB/s
    bw = paged_bench.compute_bandwidth_gbs(
        batch=1, heads=32, length=2048, head_dim=128, latency_us=100.0
    )
    expected = (33554432 / 1e9) / 1e-4
    assert abs(bw - expected) < 1e-4
    assert paged_bench.compute_bandwidth_gbs(1, 1, 1, 1, 0.0) == 0.0


def test_paged_bench_cli_parsing():
    """Verify CLI argument parsing in paged_bench."""
    parser = paged_bench.build_arg_parser()
    args = parser.parse_args([
        "--batch", "2",
        "--heads", "16",
        "--dim", "64",
        "--context-lens", "128,256",
        "--warmup", "1",
        "--iters", "2",
        "--simulated",
        "--verify",
        "--csv", "test_out.csv",
        "--markdown", "test_out.md",
    ])
    assert args.batch == 2
    assert args.heads == 16
    assert paged_bench.parse_dims(args.dim) == [64]
    assert paged_bench.parse_context_lens(args.context_lens) == [128, 256]
    assert args.warmup == 1
    assert args.iters == 2
    assert args.simulated is True
    assert args.verify is True
    assert args.csv == "test_out.csv"
    assert args.markdown == "test_out.md"
    assert paged_bench.parse_dims("all") == [64, 128]


def test_paged_bench_single_workload_execution(tmp_path):
    """Test full execution of benchmark workload with verify and export."""
    device = torch.device("cpu")
    row = paged_bench.benchmark_single_workload(
        batch=1,
        heads=2,
        head_dim=64,
        length=128,
        device=device,
        warmup=1,
        iters=2,
        verify=True,
    )
    assert row.sdpa.latency_us > 0.0
    assert row.paged_v1.latency_us > 0.0
    assert row.split_kv_4.latency_us > 0.0
    assert row.split_kv_8.latency_us > 0.0
    assert row.sdpa.bandwidth_gbs > 0.0

    csv_path = str(tmp_path / "bench.csv")
    paged_bench.write_csv_results([row], csv_path)
    assert os.path.exists(csv_path)

    md_table = paged_bench.format_markdown_table_for_dim(64, [row])
    assert "Benchmark Results" in md_table
    assert "Paged V1" in md_table

    summary = paged_bench.format_summary_breakdown([row], device, True, 1, 2)
    assert "Summary Breakdown" in summary
