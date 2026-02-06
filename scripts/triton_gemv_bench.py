#!/usr/bin/env python3
"""
Triton GeMV Kernel Benchmark

This script implements a custom Triton kernel for GeMV (General Matrix-Vector multiply)
to verify hardware-level memory alignment effects without cuBLAS kernel heuristics.

Key objectives:
1. Quantify pure hardware alignment penalty (L2 cache sector alignment)
2. Verify K % 16 = 0 is required (not K % 8 = 0) for FP16
3. Confirm GeMV is memory-bound (load-intensive)

Note: PTX analysis shows Triton uses scalar ld.global.b16 loads.
The performance difference is due to L2 cache sector (32 bytes) alignment,
not LDG instruction width differences.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available. Only cuBLAS benchmarks will run.")


# =============================================================================
# Triton GeMV Kernel
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def gemv_kernel(
        W_ptr, x_ptr, y_ptr,
        N, K,
        stride_wn, stride_wk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        GeMV kernel: y[n] = sum_k(W[n,k] * x[k])

        Each program handles BLOCK_N output elements.
        The K dimension is processed in chunks of BLOCK_K.

        Memory access pattern:
        - x[k]: Coalesced reads, reused across all N
        - W[n,k]: Row-major layout, alignment affects LDG efficiency
        """
        pid = tl.program_id(0)
        n_offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offset < N

        # Accumulator in fp32 for numerical stability
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Process K dimension in blocks
        for k_start in range(0, K, BLOCK_K):
            k_offset = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offset < K

            # Load x[k] - broadcasted to all N elements
            x_vals = tl.load(x_ptr + k_offset, mask=k_mask, other=0.0)

            # Load W[n, k] - this is where alignment matters
            # For row-major W[N, K], stride_wn = K, stride_wk = 1
            # If K % 8 != 0, loads will be misaligned
            w_ptrs = W_ptr + n_offset[:, None] * stride_wn + k_offset[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Compute partial dot product: acc += W[n,:] dot x[:]
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # Store result (cast back to fp16)
        tl.store(y_ptr + n_offset, acc.to(tl.float16), mask=n_mask)


    def triton_gemv(W: torch.Tensor, x: torch.Tensor, block_n: int = 64, block_k: int = 64) -> torch.Tensor:
        """
        Execute Triton GeMV kernel.

        Args:
            W: Weight matrix [N, K] in fp16
            x: Input vector [1, K] or [K] in fp16
            block_n: Block size for N dimension
            block_k: Block size for K dimension

        Returns:
            y: Output vector [N] in fp16
        """
        if x.dim() == 2:
            x = x.squeeze(0)

        N, K = W.shape
        y = torch.empty(N, dtype=torch.float16, device=W.device)

        # Grid: one program per BLOCK_N elements
        grid = (triton.cdiv(N, block_n),)

        gemv_kernel[grid](
            W, x, y,
            N, K,
            W.stride(0), W.stride(1),  # stride_wn, stride_wk
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )

        return y


# =============================================================================
# Benchmark Utilities
# =============================================================================

def compute_statistics(times: list) -> dict:
    """Compute timing statistics from a list of measurements."""
    import numpy as np
    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p99_ms": float(np.percentile(times, 99)),
        "count": len(times),
    }


def compute_bandwidth(n: int, k: int, time_ms: float, dtype=torch.float16) -> float:
    """
    Compute effective memory bandwidth in GB/s.

    For GeMV y = W @ x:
    - Read W: N * K elements
    - Read x: K elements
    - Write y: N elements
    Total bytes = (N*K + K + N) * bytes_per_element
    """
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    total_bytes = (n * k + k + n) * bytes_per_elem
    time_s = time_ms / 1000
    bandwidth_gbs = (total_bytes / 1e9) / time_s
    return bandwidth_gbs


def benchmark_triton_gemv(
    n: int, k: int,
    block_n: int = 64,
    block_k: int = 64,
    dtype=torch.float16,
    device: str = "cuda",
    warmup: int = 50,
    repeats: int = 200,
) -> dict:
    """
    Benchmark Triton GeMV kernel.
    """
    if not HAS_TRITON:
        return {"error": "Triton not available"}

    W = torch.randn(n, k, dtype=dtype, device=device)
    x = torch.randn(k, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        y = triton_gemv(W, x, block_n=block_n, block_k=block_k)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = triton_gemv(W, x, block_n=block_n, block_k=block_k)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    stats = compute_statistics(times)
    bandwidth = compute_bandwidth(n, k, stats["mean_ms"], dtype)

    return {
        "kernel": "triton",
        "n": n,
        "k": k,
        "block_n": block_n,
        "block_k": block_k,
        "n_mod8": n % 8,
        "k_mod8": k % 8,
        "is_aligned": (n % 8 == 0) and (k % 8 == 0),
        "dtype": str(dtype),
        **stats,
        "bandwidth_gbs": bandwidth,
    }


def benchmark_cublas_gemv(
    n: int, k: int,
    dtype=torch.float16,
    device: str = "cuda",
    warmup: int = 50,
    repeats: int = 200,
) -> dict:
    """
    Benchmark cuBLAS GeMV (via F.linear) for comparison.
    """
    W = torch.randn(n, k, dtype=dtype, device=device)
    x = torch.randn(1, k, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        y = F.linear(x, W)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = F.linear(x, W)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    stats = compute_statistics(times)
    bandwidth = compute_bandwidth(n, k, stats["mean_ms"], dtype)

    return {
        "kernel": "cublas",
        "n": n,
        "k": k,
        "n_mod8": n % 8,
        "k_mod8": k % 8,
        "is_aligned": (n % 8 == 0) and (k % 8 == 0),
        "dtype": str(dtype),
        **stats,
        "bandwidth_gbs": bandwidth,
    }


# =============================================================================
# Experiments
# =============================================================================

def run_alignment_sweep(args) -> dict:
    """
    Experiment 1: Sweep N dimension to measure alignment effect.

    Tests N from N_aligned-7 to N_aligned+8 to see the mod8 boundary effect.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Alignment Sweep")
    print("=" * 60)

    results = {"triton": [], "cublas": []}
    k = args.k_fixed

    # Test around alignment boundaries
    n_values = []
    for base in [256, 512, 1024, 2048]:
        n_values.extend(range(base - 7, base + 9))  # -7 to +8 around boundary

    for n in n_values:
        print(f"\n  N={n}, K={k}, N%8={n%8}")

        # Triton
        result_triton = benchmark_triton_gemv(
            n, k,
            block_n=args.block_n,
            block_k=args.block_k,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["triton"].append(result_triton)
        print(f"    Triton: {result_triton['mean_ms']:.4f} ms, {result_triton['bandwidth_gbs']:.1f} GB/s")

        # cuBLAS for comparison
        result_cublas = benchmark_cublas_gemv(
            n, k,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results["cublas"].append(result_cublas)
        print(f"    cuBLAS: {result_cublas['mean_ms']:.4f} ms, {result_cublas['bandwidth_gbs']:.1f} GB/s")

    return results


def run_triton_vs_cublas(args) -> dict:
    """
    Experiment 2: Compare Triton vs cuBLAS on same dimensions.

    Tests aligned and misaligned configurations to see kernel selection effects.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Triton vs cuBLAS Comparison")
    print("=" * 60)

    test_dims = [
        (1024, 4096, "aligned"),
        (1023, 4096, "N_misaligned"),
        (1024, 4095, "K_misaligned"),
        (1023, 4095, "both_misaligned"),
        (2048, 4096, "aligned_2048"),
        (2047, 4096, "N_misaligned_2047"),
    ]

    results = []

    for n, k, label in test_dims:
        print(f"\n  {label}: N={n}, K={k}")

        triton_result = benchmark_triton_gemv(
            n, k,
            block_n=args.block_n,
            block_k=args.block_k,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        cublas_result = benchmark_cublas_gemv(
            n, k,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        results.append({
            "label": label,
            "n": n,
            "k": k,
            "triton": triton_result,
            "cublas": cublas_result,
        })

        print(f"    Triton: {triton_result['mean_ms']:.4f} ms ({triton_result['bandwidth_gbs']:.1f} GB/s)")
        print(f"    cuBLAS: {cublas_result['mean_ms']:.4f} ms ({cublas_result['bandwidth_gbs']:.1f} GB/s)")

    return results


def run_block_size_sweep(args) -> dict:
    """
    Experiment 3: Sweep Triton block sizes to understand tuning effects.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Block Size Sweep")
    print("=" * 60)

    block_configs = [
        (32, 32),
        (32, 64),
        (32, 128),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 32),
        (128, 64),
        (128, 128),
    ]

    test_dims = [
        (1024, 4096),  # aligned
        (1023, 4096),  # misaligned
    ]

    results = []

    for n, k in test_dims:
        for block_n, block_k in block_configs:
            print(f"\n  N={n}, K={k}, BLOCK_N={block_n}, BLOCK_K={block_k}")

            result = benchmark_triton_gemv(
                n, k,
                block_n=block_n,
                block_k=block_k,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results.append(result)

            print(f"    {result['mean_ms']:.4f} ms, {result['bandwidth_gbs']:.1f} GB/s")

    return results


def run_llm_shapes(args) -> dict:
    """
    Experiment 4: Test real LLM dimensions from compression.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: LLM Compression Shapes")
    print("=" * 60)

    # Llama-3-8B dimensions after different compression methods
    llm_shapes = [
        {"name": "baseline", "n": 14336, "k": 4096},
        {"name": "pruned_5931", "n": 5931, "k": 4096},  # mod8=7, misaligned
        {"name": "pruned_5936", "n": 5936, "k": 4096},  # mod8=0, aligned (GAC)
        {"name": "pruned_8000", "n": 8000, "k": 4096},  # mod8=0
        {"name": "pruned_7997", "n": 7997, "k": 4096},  # mod8=5
    ]

    results = []

    for shape in llm_shapes:
        n, k = shape["n"], shape["k"]
        print(f"\n  {shape['name']}: N={n}, K={k}, N%8={n%8}")

        triton_result = benchmark_triton_gemv(
            n, k,
            block_n=args.block_n,
            block_k=args.block_k,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        cublas_result = benchmark_cublas_gemv(
            n, k,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        results.append({
            "name": shape["name"],
            "n": n,
            "k": k,
            "triton": triton_result,
            "cublas": cublas_result,
        })

        print(f"    Triton: {triton_result['mean_ms']:.4f} ms ({triton_result['bandwidth_gbs']:.1f} GB/s)")
        print(f"    cuBLAS: {cublas_result['mean_ms']:.4f} ms ({cublas_result['bandwidth_gbs']:.1f} GB/s)")

    return results


# =============================================================================
# Analysis & Visualization
# =============================================================================

def analyze_results(all_results: dict, output_dir: Path):
    """Analyze and visualize benchmark results."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Figure 1: Alignment sweep comparison
    if "alignment_sweep" in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sweep = all_results["alignment_sweep"]

        for idx, kernel in enumerate(["triton", "cublas"]):
            ax = axes[idx]
            data = sweep[kernel]

            n_vals = [d["n"] for d in data]
            latencies = [d["mean_ms"] for d in data]
            bandwidths = [d["bandwidth_gbs"] for d in data]

            # Color by alignment
            colors = ["green" if d["is_aligned"] else "red" for d in data]

            ax.scatter(n_vals, latencies, c=colors, alpha=0.7, s=30)
            ax.set_xlabel("N dimension")
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"{kernel.upper()} GeMV Latency")
            ax.grid(True, alpha=0.3)

            # Add vertical lines at mod8 boundaries
            for base in [256, 512, 1024, 2048]:
                ax.axvline(x=base, color='blue', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "alignment_sweep.png", dpi=150)
        plt.close()

    # Figure 2: Triton vs cuBLAS bar chart
    if "triton_vs_cublas" in all_results:
        fig, ax = plt.subplots(figsize=(12, 6))

        comparison = all_results["triton_vs_cublas"]
        labels = [d["label"] for d in comparison]
        triton_latencies = [d["triton"]["mean_ms"] for d in comparison]
        cublas_latencies = [d["cublas"]["mean_ms"] for d in comparison]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, triton_latencies, width, label='Triton', color='steelblue')
        bars2 = ax.bar(x + width/2, cublas_latencies, width, label='cuBLAS', color='coral')

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Triton vs cuBLAS GeMV Latency')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "triton_vs_cublas.png", dpi=150)
        plt.close()

    # Figure 3: Block size heatmap
    if "block_size_sweep" in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sweep = all_results["block_size_sweep"]

        # Group by (n, k)
        aligned_data = [d for d in sweep if d["is_aligned"]]
        misaligned_data = [d for d in sweep if not d["is_aligned"]]

        for idx, (data, title) in enumerate([(aligned_data, "Aligned (N=1024)"),
                                              (misaligned_data, "Misaligned (N=1023)")]):
            if not data:
                continue

            ax = axes[idx]

            block_ns = sorted(set(d["block_n"] for d in data))
            block_ks = sorted(set(d["block_k"] for d in data))

            heatmap = np.zeros((len(block_ns), len(block_ks)))
            for d in data:
                i = block_ns.index(d["block_n"])
                j = block_ks.index(d["block_k"])
                heatmap[i, j] = d["bandwidth_gbs"]

            im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(block_ks)))
            ax.set_yticks(range(len(block_ns)))
            ax.set_xticklabels(block_ks)
            ax.set_yticklabels(block_ns)
            ax.set_xlabel("BLOCK_K")
            ax.set_ylabel("BLOCK_N")
            ax.set_title(f"{title}\nBandwidth (GB/s)")

            # Add text annotations
            for i in range(len(block_ns)):
                for j in range(len(block_ks)):
                    ax.text(j, i, f"{heatmap[i, j]:.0f}",
                           ha="center", va="center", fontsize=8)

            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output_dir / "block_size_sweep.png", dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def print_summary(all_results: dict):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "triton_vs_cublas" in all_results:
        print("\nTriton vs cuBLAS Alignment Penalties:")
        comparison = all_results["triton_vs_cublas"]

        # Find aligned baseline
        aligned = next((d for d in comparison if d["label"] == "aligned"), None)
        if aligned:
            triton_base = aligned["triton"]["mean_ms"]
            cublas_base = aligned["cublas"]["mean_ms"]

            for d in comparison:
                if d["label"] != "aligned":
                    triton_penalty = (d["triton"]["mean_ms"] - triton_base) / triton_base * 100
                    cublas_penalty = (d["cublas"]["mean_ms"] - cublas_base) / cublas_base * 100
                    print(f"  {d['label']:20s}: Triton {triton_penalty:+6.1f}%, cuBLAS {cublas_penalty:+6.1f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Triton GeMV Alignment Benchmark")
    parser.add_argument("--output", type=str, default="results/triton_gemv",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--warmup", type=int, default=50,
                       help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=200,
                       help="Measurement iterations")
    parser.add_argument("--block-n", type=int, default=64,
                       help="Triton BLOCK_N")
    parser.add_argument("--block-k", type=int, default=64,
                       help="Triton BLOCK_K")
    parser.add_argument("--k-fixed", type=int, default=4096,
                       help="Fixed K dimension for sweeps")
    parser.add_argument("--experiments", type=str, default="all",
                       help="Experiments to run: all, sweep, compare, block, llm")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Triton GeMV Alignment Benchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"Output: {output_dir}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print(f"Block sizes: BLOCK_N={args.block_n}, BLOCK_K={args.block_k}")

    all_results = {}
    experiments = args.experiments.split(",") if args.experiments != "all" else ["sweep", "compare", "block", "llm"]

    # Run experiments
    if "sweep" in experiments:
        all_results["alignment_sweep"] = run_alignment_sweep(args)
        with open(output_dir / "alignment_sweep.json", "w") as f:
            json.dump(all_results["alignment_sweep"], f, indent=2)

    if "compare" in experiments:
        all_results["triton_vs_cublas"] = run_triton_vs_cublas(args)
        with open(output_dir / "triton_vs_cublas.json", "w") as f:
            json.dump(all_results["triton_vs_cublas"], f, indent=2)

    if "block" in experiments:
        all_results["block_size_sweep"] = run_block_size_sweep(args)
        with open(output_dir / "block_size_sweep.json", "w") as f:
            json.dump(all_results["block_size_sweep"], f, indent=2)

    if "llm" in experiments:
        all_results["llm_shapes"] = run_llm_shapes(args)
        with open(output_dir / "llm_shapes.json", "w") as f:
            json.dump(all_results["llm_shapes"], f, indent=2)

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Analyze and visualize
    analyze_results(all_results, output_dir)
    print_summary(all_results)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
