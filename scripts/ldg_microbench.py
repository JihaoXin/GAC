#!/usr/bin/env python3
"""
Memory Alignment Microbenchmark

This script isolates the pure memory load performance to verify
L2 cache sector alignment effects on bandwidth.

Key insight (updated based on PTX analysis):
- Triton generates scalar ld.global.b16 loads, not vectorized LDG.128
- The real constraint is L2 cache sector alignment (32 bytes = 16 fp16)
- K % 16 = 0 required for optimal bandwidth (not K % 8 = 0)
- Misalignment causes partial sector reads with wasted bandwidth

This benchmark:
1. Tests pure load throughput with different alignments
2. Measures effective bandwidth to quantify the penalty
3. Verifies L2 cache sector alignment is the dominant factor
"""

import argparse
import json
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Error: Triton required for this benchmark")
    exit(1)


# =============================================================================
# Triton Kernels for LDG Benchmarking
# =============================================================================

@triton.jit
def ldg_row_kernel(
    ptr,
    out_ptr,
    N,        # number of rows
    K,        # row width (contiguous dimension)
    stride,   # stride between rows (= K for contiguous)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Load rows of data to benchmark row-major access patterns.

    This simulates GeMV weight matrix access:
    - Each program loads BLOCK_N rows
    - Each row has K elements
    - If K % 8 != 0, row starts are misaligned
    """
    pid = tl.program_id(0)

    # Which rows this program handles
    row_start = pid * BLOCK_N
    row_idx = row_start + tl.arange(0, BLOCK_N)
    row_mask = row_idx < N

    # Accumulator to prevent optimization
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Load each row in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Compute pointers: ptr + row * stride + col
        # If stride (=K) is not aligned, each row starts at misaligned address
        ptrs = ptr + row_idx[:, None] * stride + k_idx[None, :]
        mask = row_mask[:, None] & k_mask[None, :]

        # Load - THIS IS WHERE ALIGNMENT MATTERS
        vals = tl.load(ptrs, mask=mask, other=0.0)

        # Accumulate to prevent dead code elimination
        acc += tl.sum(vals, axis=1)

    # Store result
    tl.store(out_ptr + row_idx, acc.to(tl.float16), mask=row_mask)


@triton.jit
def ldg_contiguous_kernel(
    ptr,
    out_ptr,
    N,        # total elements
    BLOCK: tl.constexpr,
):
    """
    Pure contiguous load benchmark.

    Tests raw memory bandwidth for sequential access.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    # Load
    vals = tl.load(ptr + offset, mask=mask, other=0.0)

    # Store (prevents optimization)
    tl.store(out_ptr + offset, vals, mask=mask)


@triton.jit
def ldg_strided_kernel(
    ptr,
    out_ptr,
    N,           # number of elements to load
    stride,      # stride between elements
    BLOCK: tl.constexpr,
):
    """
    Strided load benchmark.

    Tests memory bandwidth when accessing every Nth element.
    This isolates the effect of non-coalesced access.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N

    # Strided access
    ptrs = ptr + idx * stride
    vals = tl.load(ptrs, mask=mask, other=0.0)

    tl.store(out_ptr + idx, vals, mask=mask)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_row_load(
    n_rows: int,
    row_width: int,
    block_n: int = 32,
    block_k: int = 64,
    dtype=torch.float16,
    warmup: int = 50,
    repeats: int = 200,
) -> dict:
    """Benchmark row-major load pattern (simulates GeMV weight access)."""

    # Allocate matrix
    data = torch.randn(n_rows, row_width, dtype=dtype, device='cuda')
    out = torch.empty(n_rows, dtype=dtype, device='cuda')

    # Grid
    grid = (triton.cdiv(n_rows, block_n),)

    # Warmup
    for _ in range(warmup):
        ldg_row_kernel[grid](
            data, out, n_rows, row_width, row_width,
            BLOCK_N=block_n, BLOCK_K=block_k
        )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        ldg_row_kernel[grid](
            data, out, n_rows, row_width, row_width,
            BLOCK_N=block_n, BLOCK_K=block_k
        )
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Statistics
    import numpy as np
    times = np.array(times)
    mean_ms = float(np.mean(times))

    # Bandwidth: read N*K elements + write N elements
    bytes_read = n_rows * row_width * 2  # fp16
    bytes_write = n_rows * 2
    total_bytes = bytes_read + bytes_write
    bandwidth_gbs = (total_bytes / 1e9) / (mean_ms / 1000)

    return {
        "type": "row_load",
        "n_rows": n_rows,
        "row_width": row_width,
        "row_width_mod8": row_width % 8,
        "is_aligned": row_width % 8 == 0,
        "mean_ms": mean_ms,
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "bandwidth_gbs": bandwidth_gbs,
        "count": len(times),
    }


def benchmark_contiguous_load(
    n_elements: int,
    offset: int = 0,  # byte offset to test alignment
    block_size: int = 1024,
    dtype=torch.float16,
    warmup: int = 50,
    repeats: int = 200,
) -> dict:
    """Benchmark pure contiguous load."""

    # Allocate with extra space for offset
    total_elements = n_elements + offset // 2 + 16
    data = torch.randn(total_elements, dtype=dtype, device='cuda')
    out = torch.empty(total_elements, dtype=dtype, device='cuda')

    # Apply offset (in elements, since we can't do byte offset easily)
    elem_offset = offset // 2
    data_ptr = data[elem_offset:]
    out_ptr = out[elem_offset:]

    grid = (triton.cdiv(n_elements, block_size),)

    # Warmup
    for _ in range(warmup):
        ldg_contiguous_kernel[grid](data_ptr, out_ptr, n_elements, BLOCK=block_size)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        ldg_contiguous_kernel[grid](data_ptr, out_ptr, n_elements, BLOCK=block_size)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    times = np.array(times)
    mean_ms = float(np.mean(times))

    # Bandwidth: read + write
    total_bytes = n_elements * 2 * 2  # read + write, fp16
    bandwidth_gbs = (total_bytes / 1e9) / (mean_ms / 1000)

    return {
        "type": "contiguous_load",
        "n_elements": n_elements,
        "offset_bytes": offset,
        "mean_ms": mean_ms,
        "std_ms": float(np.std(times)),
        "bandwidth_gbs": bandwidth_gbs,
        "count": len(times),
    }


# =============================================================================
# Experiments
# =============================================================================

def experiment_row_alignment(args) -> dict:
    """
    Experiment 1: Row width alignment effect

    Tests different row widths around alignment boundaries.
    This directly measures the K-dimension alignment effect.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Row Width Alignment (K dimension)")
    print("=" * 60)

    results = []
    n_rows = 1024  # Fixed number of rows

    # Test row widths around alignment boundaries
    test_widths = []
    for base in [1024, 2048, 4096]:
        test_widths.extend(range(base - 8, base + 9))

    for width in test_widths:
        print(f"\n  Row width K={width}, K%8={width%8}", end="")

        result = benchmark_row_load(
            n_rows=n_rows,
            row_width=width,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(result)

        aligned_marker = " [ALIGNED]" if width % 8 == 0 else ""
        print(f" -> {result['bandwidth_gbs']:.1f} GB/s, {result['mean_ms']:.4f} ms{aligned_marker}")

    return {"row_alignment": results}


def experiment_ldg_comparison(args) -> dict:
    """
    Experiment 2: Direct LDG.128 vs LDG.32 comparison

    Compares aligned (mod8=0) vs misaligned (mod8!=0) for same data size.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Aligned vs Misaligned Direct Comparison")
    print("=" * 60)

    results = []
    n_rows = 1024

    test_cases = [
        # (row_width, label)
        (4096, "aligned_4096"),
        (4095, "misaligned_4095"),
        (4094, "misaligned_4094"),
        (4093, "misaligned_4093"),
        (4092, "misaligned_4092"),
        (4091, "misaligned_4091"),
        (4090, "misaligned_4090"),
        (4089, "misaligned_4089"),
        (4088, "aligned_4088"),
    ]

    for width, label in test_cases:
        print(f"\n  {label}: K={width}, K%8={width%8}")

        result = benchmark_row_load(
            n_rows=n_rows,
            row_width=width,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result["label"] = label
        results.append(result)

        print(f"    Bandwidth: {result['bandwidth_gbs']:.1f} GB/s")
        print(f"    Latency:   {result['mean_ms']:.4f} ms")

    # Compute penalties
    baseline = next(r for r in results if r["label"] == "aligned_4096")
    baseline_bw = baseline["bandwidth_gbs"]

    print("\n  Alignment Penalties (vs aligned_4096):")
    for r in results:
        penalty = (baseline_bw - r["bandwidth_gbs"]) / baseline_bw * 100
        print(f"    {r['label']:20s}: {penalty:+6.1f}% bandwidth loss")

    return {"ldg_comparison": results}


def experiment_scaling(args) -> dict:
    """
    Experiment 3: Scaling with different sizes

    Tests how alignment penalty scales with matrix size.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Size Scaling")
    print("=" * 60)

    results = []

    # Different matrix sizes
    sizes = [
        (256, 1024),
        (256, 4096),
        (512, 1024),
        (512, 4096),
        (1024, 1024),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
    ]

    for n_rows, row_width in sizes:
        print(f"\n  Size: {n_rows} x {row_width}")

        # Aligned
        result_aligned = benchmark_row_load(
            n_rows=n_rows,
            row_width=row_width,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        # Misaligned (K-1)
        result_misaligned = benchmark_row_load(
            n_rows=n_rows,
            row_width=row_width - 1,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        penalty = (result_aligned["bandwidth_gbs"] - result_misaligned["bandwidth_gbs"]) / result_aligned["bandwidth_gbs"] * 100

        results.append({
            "n_rows": n_rows,
            "row_width_aligned": row_width,
            "row_width_misaligned": row_width - 1,
            "aligned": result_aligned,
            "misaligned": result_misaligned,
            "penalty_pct": penalty,
        })

        print(f"    Aligned (K={row_width}):     {result_aligned['bandwidth_gbs']:.1f} GB/s")
        print(f"    Misaligned (K={row_width-1}): {result_misaligned['bandwidth_gbs']:.1f} GB/s")
        print(f"    Penalty: {penalty:.1f}%")

    return {"scaling": results}


# =============================================================================
# Visualization
# =============================================================================

def plot_results(all_results: dict, output_dir: Path):
    """Generate plots for LDG benchmark results."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Figure 1: Row alignment sweep
    if "row_alignment" in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        data = all_results["row_alignment"]
        widths = [d["row_width"] for d in data]
        bandwidths = [d["bandwidth_gbs"] for d in data]
        latencies = [d["mean_ms"] for d in data]
        aligned = [d["is_aligned"] for d in data]

        colors = ["green" if a else "red" for a in aligned]

        # Bandwidth plot
        ax = axes[0]
        ax.scatter(widths, bandwidths, c=colors, alpha=0.7, s=40)
        ax.set_xlabel("Row Width (K dimension)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title("Memory Bandwidth vs Row Width\n(Green=Aligned, Red=Misaligned)")
        ax.grid(True, alpha=0.3)

        # Add vertical lines at mod8 boundaries
        for base in [1024, 2048, 4096]:
            ax.axvline(x=base, color='blue', linestyle='--', alpha=0.3, label=f'K={base}' if base == 1024 else '')

        # Latency plot
        ax = axes[1]
        ax.scatter(widths, latencies, c=colors, alpha=0.7, s=40)
        ax.set_xlabel("Row Width (K dimension)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Load Latency vs Row Width")
        ax.grid(True, alpha=0.3)

        for base in [1024, 2048, 4096]:
            ax.axvline(x=base, color='blue', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "ldg_row_alignment.png", dpi=150)
        plt.close()

    # Figure 2: Direct comparison bar chart
    if "ldg_comparison" in all_results:
        fig, ax = plt.subplots(figsize=(12, 6))

        data = all_results["ldg_comparison"]
        labels = [d["label"] for d in data]
        bandwidths = [d["bandwidth_gbs"] for d in data]
        colors = ["green" if d["is_aligned"] else "red" for d in data]

        bars = ax.bar(labels, bandwidths, color=colors, alpha=0.8)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title("LDG Bandwidth: Aligned vs Misaligned\n(Green=mod8=0, Red=mod8≠0)")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, bw in zip(bars, bandwidths):
            ax.annotate(f'{bw:.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, bw),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        # Add baseline reference line
        baseline_bw = next(d["bandwidth_gbs"] for d in data if d["label"] == "aligned_4096")
        ax.axhline(y=baseline_bw, color='green', linestyle='--', alpha=0.5, label=f'Aligned baseline: {baseline_bw:.0f} GB/s')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "ldg_comparison.png", dpi=150)
        plt.close()

    # Figure 3: Scaling with penalty
    if "scaling" in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        data = all_results["scaling"]
        labels = [f"{d['n_rows']}x{d['row_width_aligned']}" for d in data]
        penalties = [d["penalty_pct"] for d in data]

        bars = ax.bar(labels, penalties, color='coral', alpha=0.8)
        ax.set_xlabel("Matrix Size (N x K)")
        ax.set_ylabel("Alignment Penalty (%)")
        ax.set_title("LDG Alignment Penalty vs Matrix Size")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, p in zip(bars, penalties):
            ax.annotate(f'{p:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, p),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / "ldg_scaling.png", dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def print_summary(all_results: dict):
    """Print summary of key findings."""
    print("\n" + "=" * 60)
    print("SUMMARY: LDG Alignment Effect")
    print("=" * 60)

    if "ldg_comparison" in all_results:
        data = all_results["ldg_comparison"]
        aligned = next(d for d in data if d["label"] == "aligned_4096")
        misaligned = next(d for d in data if d["label"] == "misaligned_4095")

        aligned_bw = aligned["bandwidth_gbs"]
        misaligned_bw = misaligned["bandwidth_gbs"]
        penalty = (aligned_bw - misaligned_bw) / aligned_bw * 100

        print(f"\nKey Finding: K-dimension alignment effect")
        print(f"  Aligned (K=4096, mod8=0):     {aligned_bw:.1f} GB/s")
        print(f"  Misaligned (K=4095, mod8=7):  {misaligned_bw:.1f} GB/s")
        print(f"  Bandwidth Loss:               {penalty:.1f}%")
        print(f"\nInterpretation:")
        print(f"  - Aligned access enables LDG.128 (16-byte vectorized loads)")
        print(f"  - Misaligned access falls back to LDG.32/64 (smaller loads)")
        print(f"  - This is a pure hardware effect, no library heuristics involved")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LDG Instruction Microbenchmark")
    parser.add_argument("--output", type=str, default="results/ldg_microbench",
                       help="Output directory")
    parser.add_argument("--warmup", type=int, default=50,
                       help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=200,
                       help="Measurement iterations")
    parser.add_argument("--experiments", type=str, default="all",
                       help="Experiments to run: all, row, compare, scaling")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LDG Instruction Microbenchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Output: {output_dir}")

    all_results = {}
    experiments = args.experiments.split(",") if args.experiments != "all" else ["row", "compare", "scaling"]

    if "row" in experiments:
        result = experiment_row_alignment(args)
        all_results.update(result)
        with open(output_dir / "row_alignment.json", "w") as f:
            json.dump(result, f, indent=2)

    if "compare" in experiments:
        result = experiment_ldg_comparison(args)
        all_results.update(result)
        with open(output_dir / "ldg_comparison.json", "w") as f:
            json.dump(result, f, indent=2)

    if "scaling" in experiments:
        result = experiment_scaling(args)
        all_results.update(result)
        with open(output_dir / "scaling.json", "w") as f:
            json.dump(result, f, indent=2)

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot and summarize
    plot_results(all_results, output_dir)
    print_summary(all_results)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
