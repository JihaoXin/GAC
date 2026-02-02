"""Profile a 128x128x128 FP16 GEMM to see how it maps to GPU hardware.

Shows: kernel name, grid dimensions, block dimensions, and timing.
This reveals the CTA tiling strategy chosen by cuBLAS.

Usage (via srun or sbatch):
    python scripts/profile_gemm_128.py
"""

import argparse
import json
import torch
from torch.profiler import profile, ProfilerActivity


def profile_gemm(M, N, K, dtype=torch.float16):
    """Profile a single GEMM and extract kernel launch info."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(10):
        torch.mm(A, B)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            torch.mm(A, B)
        torch.cuda.synchronize()

    return prof


def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SMs: {props.multi_processor_count}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Max threads/SM: {props.max_threads_per_multi_processor}")
    print()
    return props


def analyze_tiling(M, N, K, sm_count):
    """Analyze possible tiling strategies."""
    print(f"--- Tiling analysis for M={M} N={N} K={K} ---")

    # Common CTA tile sizes used by cuBLAS/CUTLASS
    cta_tiles = [(32, 32), (64, 32), (32, 64), (64, 64),
                 (128, 64), (64, 128), (128, 128), (256, 64),
                 (256, 128), (128, 256)]

    for tm, tn in cta_tiles:
        grid_m = (M + tm - 1) // tm
        grid_n = (N + tn - 1) // tn
        total = grid_m * grid_n
        waves = total / sm_count
        print(f"  CTA tile {tm:3d}x{tn:3d} → grid {grid_m}x{grid_n} = "
              f"{total:4d} CTAs, {waves:.3f} waves, "
              f"SM util {min(total, sm_count)}/{sm_count} = "
              f"{min(total, sm_count)/sm_count*100:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[128],
                        help="M=N=K sizes to profile")
    args = parser.parse_args()

    torch.cuda.init()
    props = print_gpu_info()
    sm_count = props.multi_processor_count

    for size in args.sizes:
        M = N = K = size
        print(f"{'='*60}")
        print(f"  GEMM M={M} N={N} K={K} FP16")
        print(f"{'='*60}")

        # Theoretical
        flops = 2 * M * N * K
        print(f"FLOPs: {flops:,}")
        print()

        # MMA-level analysis
        # A100: mma.sync.aligned.m16n8k16.f16
        mma_m, mma_n, mma_k = 16, 8, 16
        mma_tiles_m = M // mma_m
        mma_tiles_n = N // mma_n
        mma_tiles_k = K // mma_k
        total_mma = mma_tiles_m * mma_tiles_n * mma_tiles_k
        print(f"MMA instruction shape (A100 FP16): m{mma_m}n{mma_n}k{mma_k}")
        print(f"MMA tiles: {mma_tiles_m} x {mma_tiles_n} x {mma_tiles_k} = "
              f"{total_mma} total mma ops")
        print()

        # Tiling analysis
        analyze_tiling(M, N, K, sm_count)

        # Profile with PyTorch
        prof = profile_gemm(M, N, K)

        # Print kernel-level events
        print("--- CUDA Kernel Events ---")
        events = prof.key_averages()
        for evt in events:
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                print(f"  Kernel: {evt.key}")
                print(f"    Count: {evt.count}")
                print(f"    CPU time avg: {evt.cpu_time_total / max(evt.count, 1):.1f} us")
                print()

        # Timing
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        for _ in range(10):
            torch.mm(A, B)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / 100
        tflops = flops / (avg_ms * 1e-3) / 1e12
        print(f"Average time: {avg_ms:.4f} ms")
        print(f"Throughput: {tflops:.2f} TFLOPS")
        print(f"Peak utilization: {tflops/312*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
