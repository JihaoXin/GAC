#!/usr/bin/env python3
"""
GeMV (M=1 GEMM) Alignment Benchmark

This script benchmarks GeMV operations to analyze the impact of dimension
alignment on memory-bound operations (decode phase of LLMs).

Key differences from GEMM:
- M is small (1, 4, 8) instead of large batch
- Operation is memory-bound, not compute-bound
- Alignment impact expected to be smaller

Usage:
    python scripts/benchmark_gemv.py --output results/gemv_alignment
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_statistics(values: List[float]) -> Dict:
    """Compute statistics for a list of values."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "count": len(values),
    }


def compute_gemv_bandwidth(m: int, n: int, k: int, dtype: torch.dtype, time_s: float) -> float:
    """
    Compute effective memory bandwidth for GeMV in GB/s.

    GeMV: y[m,n] = x[m,k] @ W[k,n]^T, where W is stored as [n,k]
    Memory access: read W (n*k), read x (m*k), write y (m*n)
    For M=1: dominated by reading W (n*k elements)
    """
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = (n * k + m * k + m * n) * dtype_size
    bandwidth_gbs = total_bytes / (time_s * 1e9)
    return bandwidth_gbs


def benchmark_gemv(
    n: int,
    k: int,
    m: int = 1,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    warmup: int = 50,
    repeats: int = 200,
) -> Dict:
    """
    Benchmark GeMV operation: y = F.linear(x, W)

    This is equivalent to: y[m,n] = x[m,k] @ W[k,n]^T
    where W is stored as [n, k] (PyTorch convention)

    Args:
        n: Output dimension (hidden_dim in LLM)
        k: Input dimension (intermediate_dim in LLM)
        m: Batch size (1 for pure decode, small for micro-batch)
        dtype: Data type
        device: CUDA device
        warmup: Warmup iterations
        repeats: Measurement iterations

    Returns:
        Dictionary with timing and performance metrics
    """
    # Allocate tensors: W[n,k], x[m,k] -> y[m,n]
    W = torch.randn(n, k, dtype=dtype, device=device)
    x = torch.randn(m, k, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        y = F.linear(x, W)
    torch.cuda.synchronize(device)

    # Measurement with CUDA events
    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)
        start.record()
        y = F.linear(x, W)
        end.record()
        torch.cuda.synchronize(device)

        times_ms.append(start.elapsed_time(end))

    # Compute statistics
    stats = compute_statistics(times_ms)

    # Compute bandwidth
    mean_time_s = stats["mean"] / 1000.0
    bandwidth = compute_gemv_bandwidth(m, n, k, dtype, mean_time_s)

    # Alignment info
    n_mod8 = n % 8
    k_mod8 = k % 8
    is_aligned = (n_mod8 == 0) and (k_mod8 == 0)

    return {
        "m": m,
        "n": n,
        "k": k,
        "n_mod8": n_mod8,
        "k_mod8": k_mod8,
        "is_aligned": is_aligned,
        "dtype": str(dtype),
        "mean_ms": stats["mean"],
        "std_ms": stats["std"],
        "p50_ms": stats["p50"],
        "p90_ms": stats["p90"],
        "bandwidth_gbs": bandwidth,
        "count": stats["count"],
    }


def run_dimension_sweep(
    n_values: List[int],
    k_values: List[int],
    m_values: List[int],
    dtype: torch.dtype,
    device: str,
    warmup: int,
    repeats: int,
) -> List[Dict]:
    """Run GeMV benchmark across dimension combinations."""
    results = []
    total = len(n_values) * len(k_values) * len(m_values)
    count = 0

    for k in k_values:
        for m in m_values:
            for n in n_values:
                count += 1
                if count % 50 == 0:
                    print(f"  Progress: {count}/{total} (n={n}, k={k}, m={m})")

                try:
                    res = benchmark_gemv(n, k, m, dtype, device, warmup, repeats)
                    results.append(res)
                except Exception as e:
                    print(f"  Error at n={n}, k={k}, m={m}: {e}")

                # Clear cache periodically
                if count % 100 == 0:
                    torch.cuda.empty_cache()

    return results


def run_gemm_vs_gemv_comparison(
    test_dims: List[tuple],
    m_gemm: int,
    m_gemv: int,
    dtype: torch.dtype,
    device: str,
    warmup: int,
    repeats: int,
) -> List[Dict]:
    """Compare GEMM (large M) vs GeMV (M=1) for same dimensions."""
    results = []

    for n, k in test_dims:
        print(f"  Testing n={n}, k={k}")

        # GeMV (M=1)
        gemv_res = benchmark_gemv(n, k, m_gemv, dtype, device, warmup, repeats)
        gemv_res["type"] = "gemv"
        results.append(gemv_res)

        # GEMM (large M)
        gemm_res = benchmark_gemv(n, k, m_gemm, dtype, device, warmup, repeats)
        gemm_res["type"] = "gemm"
        results.append(gemm_res)

        torch.cuda.empty_cache()

    return results


def run_llm_shapes(
    dtype: torch.dtype,
    device: str,
    warmup: int,
    repeats: int,
) -> List[Dict]:
    """Benchmark with actual LLM shapes from Llama-3-8B compression."""
    # Llama-3-8B MLP dimensions after LLM-Pruner compression
    llm_shapes = [
        {"name": "baseline", "n": 14336, "k": 4096},
        {"name": "pruned_5931", "n": 5931, "k": 4096},   # mod8=7
        {"name": "pruned_5936", "n": 5936, "k": 4096},   # mod8=0 (aligned)
        {"name": "pruned_6054", "n": 6054, "k": 4096},   # mod8=6
        {"name": "pruned_6056", "n": 6056, "k": 4096},   # mod8=0 (aligned)
        {"name": "pruned_10450", "n": 10450, "k": 4096}, # mod8=2
        {"name": "pruned_10456", "n": 10456, "k": 4096}, # mod8=0 (aligned)
    ]

    results = []
    for shape in llm_shapes:
        print(f"  Testing {shape['name']}: n={shape['n']}, k={shape['k']}")

        for m in [1, 4, 8]:
            res = benchmark_gemv(shape["n"], shape["k"], m, dtype, device, warmup, repeats)
            res["name"] = shape["name"]
            results.append(res)

        torch.cuda.empty_cache()

    return results


def generate_plots(results: Dict, output_dir: Path):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Plot 1: N dimension sweep (latency vs N)
    sweep_results = results.get("dimension_sweep", [])
    if sweep_results:
        # Filter for M=1, K=4096
        m1_k4096 = [r for r in sweep_results if r["m"] == 1 and r["k"] == 4096]
        if m1_k4096:
            n_vals = [r["n"] for r in m1_k4096]
            lat_vals = [r["mean_ms"] for r in m1_k4096]
            aligned = [r["is_aligned"] for r in m1_k4096]

            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["green" if a else "red" for a in aligned]
            ax.scatter(n_vals, lat_vals, c=colors, s=8, alpha=0.6)
            ax.set_xlabel("N dimension")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("GeMV Latency vs N (M=1, K=4096)")
            ax.grid(True, alpha=0.3)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Aligned (mod8=0)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Misaligned'),
            ]
            ax.legend(handles=legend_elements)

            fig.tight_layout()
            fig.savefig(plot_dir / "gemv_latency_vs_n.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: {plot_dir / 'gemv_latency_vs_n.png'}")

    # Plot 2: Alignment penalty comparison
    comparison = results.get("gemm_vs_gemv", [])
    if comparison:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Group by type
        gemv_data = [r for r in comparison if r["type"] == "gemv"]
        gemm_data = [r for r in comparison if r["type"] == "gemm"]

        # Latency comparison
        ax = axes[0]
        x = range(len(gemv_data))
        labels = [f"n={r['n']}" for r in gemv_data]
        gemv_lat = [r["mean_ms"] for r in gemv_data]
        gemm_lat = [r["mean_ms"] for r in gemm_data]

        width = 0.35
        ax.bar([i - width/2 for i in x], gemv_lat, width, label='GeMV (M=1)', color='steelblue')
        ax.bar([i + width/2 for i in x], gemm_lat, width, label='GEMM (M=512)', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("GEMM vs GeMV Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bandwidth comparison
        ax = axes[1]
        gemv_bw = [r["bandwidth_gbs"] for r in gemv_data]
        gemm_bw = [r["bandwidth_gbs"] for r in gemm_data]

        ax.bar([i - width/2 for i in x], gemv_bw, width, label='GeMV (M=1)', color='steelblue')
        ax.bar([i + width/2 for i in x], gemm_bw, width, label='GEMM (M=512)', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title("GEMM vs GeMV Effective Bandwidth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_dir / "gemm_vs_gemv_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {plot_dir / 'gemm_vs_gemv_comparison.png'}")

    # Plot 3: LLM shapes
    llm_results = results.get("llm_shapes", [])
    if llm_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by M
        for m in [1, 4, 8]:
            data = [r for r in llm_results if r["m"] == m]
            names = [r["name"] for r in data]
            latencies = [r["mean_ms"] for r in data]
            colors = ["green" if r["is_aligned"] else "red" for r in data]

            x = range(len(names))
            offset = (m - 4) * 0.25
            ax.bar([i + offset for i in x], latencies, 0.2, label=f'M={m}', alpha=0.8)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel("Latency (ms)")
        ax.set_title("GeMV Latency for Llama-3-8B Shapes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_dir / "gemv_llm_shapes.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {plot_dir / 'gemv_llm_shapes.png'}")


def main():
    parser = argparse.ArgumentParser(description="GeMV Alignment Benchmark")
    parser.add_argument("--output", type=str, default="results/gemv_alignment",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip dimension sweep (fast mode)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float16

    print("=" * 70)
    print("GeMV Alignment Benchmark")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Dtype: {dtype}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print("=" * 70)

    results = {
        "config": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "warmup": args.warmup,
            "repeats": args.repeats,
        }
    }

    # Experiment 1: Dimension sweep
    if not args.skip_sweep:
        print("\n[Experiment 1] Dimension Sweep")
        print("-" * 40)

        # N values: dense near boundaries, sparse for large values
        n_values = list(range(64, 512, 4)) + list(range(512, 2048, 8)) + list(range(2048, 4096, 16))
        k_values = [1024, 2048, 4096]
        m_values = [1, 4, 8]

        print(f"  N values: {len(n_values)} points ({min(n_values)} to {max(n_values)})")
        print(f"  K values: {k_values}")
        print(f"  M values: {m_values}")

        sweep_results = run_dimension_sweep(
            n_values, k_values, m_values, dtype, str(device), args.warmup, args.repeats
        )
        results["dimension_sweep"] = sweep_results
        print(f"  Completed: {len(sweep_results)} measurements")

        # Save intermediate
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2)

    # Experiment 2: GEMM vs GeMV comparison
    print("\n[Experiment 2] GEMM vs GeMV Comparison")
    print("-" * 40)

    test_dims = [
        (1023, 4096),  # misaligned N
        (1024, 4096),  # aligned N
        (2047, 4096),  # misaligned N
        (2048, 4096),  # aligned N
        (1024, 4095),  # misaligned K
        (1024, 4096),  # aligned K
    ]

    comparison_results = run_gemm_vs_gemv_comparison(
        test_dims, m_gemm=512, m_gemv=1, dtype=dtype, device=str(device),
        warmup=args.warmup, repeats=args.repeats
    )
    results["gemm_vs_gemv"] = comparison_results
    print(f"  Completed: {len(comparison_results)} measurements")

    # Experiment 3: LLM shapes
    print("\n[Experiment 3] LLM Shapes (Llama-3-8B)")
    print("-" * 40)

    llm_results = run_llm_shapes(dtype, str(device), args.warmup, args.repeats)
    results["llm_shapes"] = llm_results
    print(f"  Completed: {len(llm_results)} measurements")

    # Compute alignment penalty summary
    print("\n" + "=" * 70)
    print("ALIGNMENT PENALTY SUMMARY")
    print("=" * 70)

    # From comparison results
    for n, k in [(1023, 4096), (1024, 4096)]:
        gemv = [r for r in comparison_results if r["n"] == n and r["k"] == k and r["type"] == "gemv"]
        gemm = [r for r in comparison_results if r["n"] == n and r["k"] == k and r["type"] == "gemm"]
        if gemv and gemm:
            print(f"n={n}, k={k}:")
            print(f"  GeMV (M=1):   {gemv[0]['mean_ms']:.4f} ms")
            print(f"  GEMM (M=512): {gemm[0]['mean_ms']:.4f} ms")

    # Compute penalty: (misaligned - aligned) / aligned
    aligned_gemv = [r for r in comparison_results if r["n"] == 1024 and r["type"] == "gemv"]
    misaligned_gemv = [r for r in comparison_results if r["n"] == 1023 and r["type"] == "gemv"]
    if aligned_gemv and misaligned_gemv:
        penalty = (misaligned_gemv[0]["mean_ms"] - aligned_gemv[0]["mean_ms"]) / aligned_gemv[0]["mean_ms"] * 100
        print(f"\nGeMV Alignment Penalty (n=1023 vs 1024): {penalty:.1f}%")

    aligned_gemm = [r for r in comparison_results if r["n"] == 1024 and r["type"] == "gemm"]
    misaligned_gemm = [r for r in comparison_results if r["n"] == 1023 and r["type"] == "gemm"]
    if aligned_gemm and misaligned_gemm:
        penalty = (misaligned_gemm[0]["mean_ms"] - aligned_gemm[0]["mean_ms"]) / aligned_gemm[0]["mean_ms"] * 100
        print(f"GEMM Alignment Penalty (n=1023 vs 1024): {penalty:.1f}%")

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'all_results.json'}")

    # Generate plots
    print("\n[Plotting]")
    print("-" * 40)
    generate_plots(results, output_dir)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
