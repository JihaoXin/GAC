"""Profile cuBLAS kernel selection across various alignment boundaries.

Tests dimensions around key alignment points (8, 16, 32, 64, 128)
to reveal how cuBLAS reacts to irregular dimensions.

Usage:
    python scripts/profile_alignment.py              # original experiments
    python scripts/profile_alignment.py --sweep      # fine-grained 64-128 sweep, output CSV
"""

import argparse
import csv
import os
import torch
from torch.profiler import profile, ProfilerActivity


def get_kernel_name(M, N, K, dtype=torch.float16):
    """Profile a GEMM and return the kernel name chosen by cuBLAS."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(5):
        torch.mm(A, B)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        torch.mm(A, B)
        torch.cuda.synchronize()

    kernel = ""
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            kernel = evt.key
            break
    return kernel


def time_gemm(M, N, K, dtype=torch.float16, warmup=10, repeats=200):
    """Time a GEMM and return average ms."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeats


def classify_kernel(name):
    """Extract key properties from kernel name."""
    props = []

    # Source: cuBLAS native vs CUTLASS
    if "cutlass" in name.lower():
        props.append("CUTLASS")
        if "cutlass_75" in name:
            props.append("sm75")
        elif "cutlass_80" in name:
            props.append("sm80")
    elif "xmma" in name or "ampere" in name:
        props.append("cuBLAS-native")
        props.append("sm80")

    # Alignment
    if "align1" in name:
        props.append("align1")
    elif "align2" in name:
        props.append("align2")
    elif "align4" in name:
        props.append("align4")
    elif "align8" in name:
        props.append("align8")

    # Tensor Core type
    if "tensor16x8x16" in name:
        props.append("mma.m16n8k16")
    elif "wmma" in name or "s161616" in name:
        props.append("wmma")
    elif "s16816" in name:
        props.append("mma.m16n8k16")

    # CTA tile
    import re
    tile_match = re.search(r'tilesize(\d+)x(\d+)x(\d+)', name)
    if tile_match:
        props.append(f"CTA={tile_match.group(1)}x{tile_match.group(2)}")

    tile_match2 = re.search(r'_(\d+)x(\d+)_(\d+)x', name)
    if not tile_match and tile_match2:
        props.append(f"CTA={tile_match2.group(1)}x{tile_match2.group(2)}")

    return " | ".join(props) if props else name[:60]


def run_sweep(output_csv="results/alignment_sweep.csv"):
    """Sweep dim 64-128 for M, N, K independently. Output CSV.

    Baseline shape: M=2048, N=2048, K=128
    Sweep ranges:
      M: 1024-2048 (other dims: N=2048, K=128)
      N: 1024-2048 (other dims: M=2048, K=128)
      K: 64-128    (other dims: M=2048, N=2048)
    """
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SMs: {props.multi_processor_count}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    sweeps = [
        ("N", range(1024, 2049), lambda d: (2048, d, 128)),
        ("K", range(64, 129),    lambda d: (2048, 2048, d)),
        ("M", range(1024, 2049), lambda d: (d, 2048, 128)),
    ]

    rows = []
    for dim_name, dim_range, make_shape in sweeps:
        lo, hi = dim_range[0], dim_range[-1]
        print(f"\nSweeping {dim_name} from {lo} to {hi}...")
        for d in dim_range:
            M, N, K = make_shape(d)
            kernel = get_kernel_name(M, N, K)
            t_ms = time_gemm(M, N, K)
            t_us = t_ms * 1000
            flops = 2 * M * N * K
            tflops = flops / (t_ms * 1e-3) / 1e12
            kclass = classify_kernel(kernel)

            rows.append({
                "dim_name": dim_name,
                "dim_value": d,
                "time_us": round(t_us, 2),
                "tflops": round(tflops, 2),
                "kernel": kclass,
            })
            print(f"  {dim_name}={d:>4d}  {t_us:>8.1f} us  {tflops:>8.2f} TFLOPS  {kclass}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim_name", "dim_value", "time_us", "tflops", "kernel"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to {output_csv} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="Run fine-grained 64-128 sweep and output CSV")
    parser.add_argument("--sweep-csv", default="results/alignment_sweep.csv",
                        help="Output CSV path for sweep mode")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.sweep_csv)
        return

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SMs: {props.multi_processor_count}")
    print()

    # =========================================================
    # Experiment 1: Vary N around alignment boundaries
    # Fix M=K=1024 (large enough to saturate), sweep N
    # =========================================================
    print("=" * 100)
    print("  Experiment 1: Fix M=K=1024, vary N (alignment sensitivity)")
    print("=" * 100)
    print(f"{'N':>6} {'%8':>4} {'%16':>4} {'%32':>4} {'%64':>4} "
          f"{'Time(us)':>10} {'TFLOPS':>8} {'Kernel Properties'}")
    print("-" * 100)

    test_ns = [
        # Odd / prime numbers
        31, 33, 47, 49, 63, 65,
        # Around 8-boundary
        95, 96, 97,
        # Around 16-boundary
        111, 112, 113,
        # Around 32-boundary
        127, 128, 129,
        # Around 64-boundary
        191, 192, 193,
        # Around 128-boundary
        255, 256, 257,
        # Typical compressed dims
        107, 214, 321,
        # Typical aligned dims
        64, 128, 256, 512,
    ]
    test_ns = sorted(set(test_ns))

    M = K = 1024
    for N in test_ns:
        kernel = get_kernel_name(M, N, K)
        t_ms = time_gemm(M, N, K)
        t_us = t_ms * 1000
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12

        print(f"{N:>6} {N%8:>4} {N%16:>4} {N%32:>4} {N%64:>4} "
              f"{t_us:>10.1f} {tflops:>8.2f}   {classify_kernel(kernel)}")

    # =========================================================
    # Experiment 2: M=N=K (square), fine-grained sweep
    # =========================================================
    print()
    print("=" * 100)
    print("  Experiment 2: M=N=K (square), fine-grained sweep near boundaries")
    print("=" * 100)
    print(f"{'MNK':>6} {'%8':>4} {'%16':>4} {'%32':>4} {'%64':>4} "
          f"{'Time(us)':>10} {'TFLOPS':>8} {'Kernel Properties'}")
    print("-" * 100)

    test_sizes = sorted(set([
        # Small irregular
        31, 32, 33, 47, 48, 49, 63, 64, 65,
        # Around 96
        95, 96, 97,
        # Around 107 (PaLU compressed dim)
        105, 106, 107, 108, 110, 112,
        # Around 128
        126, 127, 128, 129, 130,
        # Larger
        192, 224, 255, 256, 257,
        384, 512,
    ]))

    for size in test_sizes:
        M = N = K = size
        kernel = get_kernel_name(M, N, K)
        t_ms = time_gemm(M, N, K)
        t_us = t_ms * 1000
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12

        print(f"{size:>6} {size%8:>4} {size%16:>4} {size%32:>4} {size%64:>4} "
              f"{t_us:>10.1f} {tflops:>8.2f}   {classify_kernel(kernel)}")

    # =========================================================
    # Experiment 3: Only vary K (reduction dimension)
    # Fix M=N=1024, sweep K to see K-alignment effects
    # =========================================================
    print()
    print("=" * 100)
    print("  Experiment 3: Fix M=N=1024, vary K (reduction dim alignment)")
    print("=" * 100)
    print(f"{'K':>6} {'%8':>4} {'%16':>4} {'%32':>4} {'%64':>4} "
          f"{'Time(us)':>10} {'TFLOPS':>8} {'Kernel Properties'}")
    print("-" * 100)

    M = N = 1024
    test_ks = sorted(set([
        31, 32, 33, 47, 48, 49, 63, 64, 65,
        95, 96, 97, 107, 112, 127, 128, 129,
        191, 192, 193, 255, 256, 257,
        512, 1024,
    ]))

    for K in test_ks:
        kernel = get_kernel_name(M, N, K)
        t_ms = time_gemm(M, N, K)
        t_us = t_ms * 1000
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12

        print(f"{K:>6} {K%8:>4} {K%16:>4} {K%32:>4} {K%64:>4} "
              f"{t_us:>10.1f} {tflops:>8.2f}   {classify_kernel(kernel)}")

    # =========================================================
    # Experiment 4: Only vary M (token/batch dimension)
    # Fix N=K=1024, sweep M to see M-alignment effects
    # Corresponds to: Token Eviction compressing sequence length
    # =========================================================
    print()
    print("=" * 100)
    print("  Experiment 4: Fix N=K=1024, vary M (token dim alignment)")
    print("=" * 100)
    print(f"{'M':>6} {'%8':>4} {'%16':>4} {'%32':>4} {'%64':>4} "
          f"{'Time(us)':>10} {'TFLOPS':>8} {'Kernel Properties'}")
    print("-" * 100)

    N = K = 1024
    test_ms = sorted(set([
        31, 32, 33, 47, 48, 49, 63, 64, 65,
        95, 96, 97, 107, 112, 127, 128, 129,
        191, 192, 193, 255, 256, 257,
        512, 1024,
    ]))

    for M in test_ms:
        kernel = get_kernel_name(M, N, K)
        t_ms = time_gemm(M, N, K)
        t_us = t_ms * 1000
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12

        print(f"{M:>6} {M%8:>4} {M%16:>4} {M%32:>4} {M%64:>4} "
              f"{t_us:>10.1f} {tflops:>8.2f}   {classify_kernel(kernel)}")

    # =========================================================
    # Summary table: aligned vs misaligned for each dimension
    # =========================================================
    print()
    print("=" * 100)
    print("  Summary: 107 vs 112 vs 128 for each dimension independently")
    print("  (other two dims fixed at 1024)")
    print("=" * 100)
    print(f"{'Scenario':<40} {'Time(us)':>10} {'TFLOPS':>8} {'Kernel Properties'}")
    print("-" * 100)

    scenarios = [
        # SVD: K is irregular
        ("SVD  (K=107, M=N=1024)", 1024, 1024, 107),
        ("SVD  (K=112, M=N=1024)", 1024, 1024, 112),
        ("SVD  (K=128, M=N=1024)", 1024, 1024, 128),
        # Pruning: N is irregular
        ("Prune (N=107, M=K=1024)", 1024, 107, 1024),
        ("Prune (N=112, M=K=1024)", 1024, 112, 1024),
        ("Prune (N=128, M=K=1024)", 1024, 128, 1024),
        # Token eviction: M is irregular
        ("Token (M=107, N=K=1024)", 107, 1024, 1024),
        ("Token (M=112, N=K=1024)", 112, 1024, 1024),
        ("Token (M=128, N=K=1024)", 128, 1024, 1024),
    ]

    for label, M, N, K in scenarios:
        kernel = get_kernel_name(M, N, K)
        t_ms = time_gemm(M, N, K)
        t_us = t_ms * 1000
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12
        print(f"{label:<40} {t_us:>10.1f} {tflops:>8.2f}   {classify_kernel(kernel)}")


if __name__ == "__main__":
    main()
