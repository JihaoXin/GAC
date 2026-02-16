#!/usr/bin/env python3
"""
C2.3 Hardware Layer Analysis Experiment

Analyzes hardware-level factors affecting GEMM/SDPA performance for non-aligned dimensions:
1. Tensor Core utilization vs dimension alignment
2. L2 cache sector efficiency
3. Memory bandwidth efficiency

Hypothesis to validate:
- H1: Tensor Core requires K % 16 == 0 for FP16/BF16
- H2: L2 cache sector (32 bytes) causes over-fetch for non-aligned strides
- H3: Non-aligned dims cause bandwidth waste (< theoretical peak)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.measurement import benchmark_kernel, compute_gemm_tflops, compute_gemm_bandwidth
from src.utils import set_deterministic, get_dtype, compute_statistics, allocate_tensors
from src.environment import collect_environment


# Theoretical peaks
GPU_SPECS = {
    "A100": {"FP16_TFLOPS": 312, "MEM_BW_GBS": 2039},  # 80GB
    "H100": {"FP16_TFLOPS": 989, "MEM_BW_GBS": 3350},  # SXM5 (approx)
    "DEFAULT": {"FP16_TFLOPS": 312, "MEM_BW_GBS": 2039},
}

CURRENT_GPU_SPECS = GPU_SPECS["DEFAULT"]

def set_gpu_specs(device: str):
    global CURRENT_GPU_SPECS
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device)
            print(f"Detected GPU: {gpu_name}")
            if "H100" in gpu_name:
                CURRENT_GPU_SPECS = GPU_SPECS["H100"]
            elif "A100" in gpu_name:
                CURRENT_GPU_SPECS = GPU_SPECS["A100"]
            else:
                print(f"Unknown GPU {gpu_name}, using default (A100) specs.")
                CURRENT_GPU_SPECS = GPU_SPECS["DEFAULT"]
    except Exception as e:
        print(f"Error detecting GPU: {e}, using default specs.")



def analyze_tensor_core_utilization(device: str = "cuda:0", seed: int = 42) -> dict:
    """
    H1: Analyze Tensor Core utilization for different K alignments.

    Tensor Core requirements for A100:
    - FP16: M, N, K must be multiples of 8 (but 16 preferred for efficiency)
    - Optimal tile sizes: 16x16, 32x32

    Test: Compare TFLOPs for K = 112, 113, 114, ..., 128 (holding M, N constant)
    """
    print("\n" + "="*60)
    print("H1: Tensor Core Utilization vs K Alignment")
    print("="*60)

    set_deterministic(seed)
    dtype = torch.float16

    # Large M, N to ensure compute-bound (not memory-bound)
    M, N = 8192, 8192

    # Test K values around the PaLU typical range (114-125)
    K_values = list(range(104, 129))  # 104-128 inclusive

    results = {
        "description": "Tensor Core utilization vs K dimension alignment",
        "config": {"M": M, "N": N, "dtype": "fp16"},
        "measurements": [],
        "peak_tflops": CURRENT_GPU_SPECS["FP16_TFLOPS"],
    }

    warmup = 30
    measure = 100

    for K in K_values:
        a, b = allocate_tensors((M, K), (K, N), dtype=dtype, device=device, seed=seed)

        def kernel_fn():
            torch.matmul(a, b)

        stats = benchmark_kernel(kernel_fn, warmup, measure, device)

        mean_time_s = stats["mean"] / 1000.0
        tflops = compute_gemm_tflops(M, N, K, mean_time_s)
        utilization = (tflops / CURRENT_GPU_SPECS["FP16_TFLOPS"]) * 100

        # Compute alignment properties
        mod_8 = K % 8 == 0
        mod_16 = K % 16 == 0
        mod_32 = K % 32 == 0

        result = {
            "K": K,
            "mod_8": mod_8,
            "mod_16": mod_16,
            "mod_32": mod_32,
            "latency_ms": stats["mean"],
            "latency_std_ms": stats["std"],
            "tflops": tflops,
            "tc_utilization_pct": utilization,
        }
        results["measurements"].append(result)

        align_str = "✓16" if mod_16 else ("✓8" if mod_8 else "✗")
        print(f"  K={K:3d} [{align_str:>3}]: {stats['mean']:.3f} ms, {tflops:.1f} TFLOPS ({utilization:.1f}% TC util)")

        del a, b
        torch.cuda.empty_cache()

    # Compute summary statistics
    aligned_16 = [m for m in results["measurements"] if m["mod_16"]]
    aligned_8_only = [m for m in results["measurements"] if m["mod_8"] and not m["mod_16"]]
    non_aligned = [m for m in results["measurements"] if not m["mod_8"]]

    results["summary"] = {
        "aligned_16_avg_tflops": np.mean([m["tflops"] for m in aligned_16]) if aligned_16 else 0,
        "aligned_8_only_avg_tflops": np.mean([m["tflops"] for m in aligned_8_only]) if aligned_8_only else 0,
        "non_aligned_avg_tflops": np.mean([m["tflops"] for m in non_aligned]) if non_aligned else 0,
        "aligned_16_avg_latency_ms": np.mean([m["latency_ms"] for m in aligned_16]) if aligned_16 else 0,
        "aligned_8_only_avg_latency_ms": np.mean([m["latency_ms"] for m in aligned_8_only]) if aligned_8_only else 0,
        "non_aligned_avg_latency_ms": np.mean([m["latency_ms"] for m in non_aligned]) if non_aligned else 0,
    }

    if aligned_16 and non_aligned:
        slowdown = results["summary"]["non_aligned_avg_latency_ms"] / results["summary"]["aligned_16_avg_latency_ms"]
        results["summary"]["non_aligned_slowdown_vs_16"] = slowdown
        print(f"\n  Summary: Non-aligned is {(slowdown-1)*100:.1f}% slower than 16-aligned")

    return results


def analyze_l2_cache_efficiency(device: str = "cuda:0", seed: int = 42) -> dict:
    """
    H2: Analyze L2 cache sector efficiency.

    A100 L2 cache:
    - 40 MB L2 cache
    - 128 byte cache line
    - 32 byte sector (minimum fetch unit)

    For FP16 (2 bytes/elem):
    - 16 elements = 32 bytes = 1 sector
    - head_dim=107 → 214 bytes = 7 sectors (224 bytes) → 4.5% waste
    - head_dim=112 → 224 bytes = 7 sectors (224 bytes) → 0% waste
    - head_dim=113 → 226 bytes = 8 sectors (256 bytes) → 13.3% waste

    Test: Measure effective bandwidth for various head_dim values
    """
    print("\n" + "="*60)
    print("H2: L2 Cache Sector Efficiency Analysis")
    print("="*60)

    set_deterministic(seed)
    dtype = torch.float16
    elem_size = 2  # bytes
    sector_size = 32  # bytes

    # Smaller M, N to be memory-bound (showcase L2 effects)
    M = 2048

    # Test head_dims in PaLU range
    head_dims = list(range(104, 129))

    results = {
        "description": "L2 cache sector efficiency for different head_dim",
        "config": {"M": M, "dtype": "fp16", "sector_size_bytes": sector_size},
        "measurements": [],
    }

    warmup = 30
    measure = 100

    for head_dim in head_dims:
        # Memory access pattern: read Q (M x head_dim), K (M x head_dim), V (M x head_dim)
        # This simulates the memory access pattern of attention

        # For simplicity, use a GEMM that simulates the memory pattern
        # (M, head_dim) @ (head_dim, M) - this reads rows of head_dim elements
        K = head_dim
        N = M

        a, b = allocate_tensors((M, K), (K, N), dtype=dtype, device=device, seed=seed)

        def kernel_fn():
            torch.matmul(a, b)

        stats = benchmark_kernel(kernel_fn, warmup, measure, device)

        mean_time_s = stats["mean"] / 1000.0
        bandwidth = compute_gemm_bandwidth(M, N, K, dtype, mean_time_s)

        # Calculate L2 sector waste
        row_bytes = head_dim * elem_size
        sectors_needed = (row_bytes + sector_size - 1) // sector_size
        bytes_fetched = sectors_needed * sector_size
        waste_pct = ((bytes_fetched - row_bytes) / row_bytes) * 100

        # Efficiency relative to peak
        bw_efficiency = (bandwidth / CURRENT_GPU_SPECS["MEM_BW_GBS"]) * 100

        mod_8 = head_dim % 8 == 0
        mod_16 = head_dim % 16 == 0

        result = {
            "head_dim": head_dim,
            "mod_8": mod_8,
            "mod_16": mod_16,
            "row_bytes": row_bytes,
            "sectors_needed": sectors_needed,
            "bytes_fetched": bytes_fetched,
            "sector_waste_pct": waste_pct,
            "latency_ms": stats["mean"],
            "bandwidth_gbs": bandwidth,
            "bandwidth_efficiency_pct": bw_efficiency,
        }
        results["measurements"].append(result)

        align_str = "✓16" if mod_16 else ("✓8" if mod_8 else "✗")
        print(f"  D={head_dim:3d} [{align_str:>3}]: {stats['mean']:.3f} ms, {bandwidth:.1f} GB/s ({bw_efficiency:.1f}% peak), L2 waste: {waste_pct:.1f}%")

        del a, b
        torch.cuda.empty_cache()

    # Summary
    aligned_16 = [m for m in results["measurements"] if m["mod_16"]]
    non_aligned = [m for m in results["measurements"] if not m["mod_8"]]

    results["summary"] = {
        "aligned_16_avg_bw_gbs": np.mean([m["bandwidth_gbs"] for m in aligned_16]) if aligned_16 else 0,
        "non_aligned_avg_bw_gbs": np.mean([m["bandwidth_gbs"] for m in non_aligned]) if non_aligned else 0,
        "avg_sector_waste_non_aligned_pct": np.mean([m["sector_waste_pct"] for m in non_aligned]) if non_aligned else 0,
    }

    return results


def analyze_memory_bandwidth_efficiency(device: str = "cuda:0", seed: int = 42) -> dict:
    """
    H3: Analyze memory bandwidth efficiency for different strides.

    Test: Compare achieved bandwidth for coalesced vs non-coalesced access patterns
    Simulate the effect of non-aligned head_dim on memory access efficiency.
    """
    print("\n" + "="*60)
    print("H3: Memory Bandwidth Efficiency Analysis")
    print("="*60)

    set_deterministic(seed)
    dtype = torch.float16

    # Test different stride patterns
    # Simulate: seq_len x head_dim matrix with varying head_dim (stride)
    seq_len = 4096
    head_dims = [104, 107, 108, 112, 113, 114, 115, 116, 120, 121, 124, 128]

    results = {
        "description": "Memory bandwidth efficiency for different head_dim strides",
        "config": {"seq_len": seq_len, "dtype": "fp16"},
        "measurements": [],
    }

    warmup = 30
    measure = 100

    for head_dim in head_dims:
        # Create tensor with specific stride
        # This simulates reading rows of head_dim elements
        tensor = torch.randn(seq_len, head_dim, dtype=dtype, device=device)

        # Simple copy/reduce operation to measure pure memory bandwidth
        def kernel_fn():
            # Sum along head_dim axis - reads all elements
            return tensor.sum(dim=1)

        stats = benchmark_kernel(kernel_fn, warmup, measure, device)

        mean_time_s = stats["mean"] / 1000.0
        total_bytes = seq_len * head_dim * tensor.element_size()
        bandwidth = total_bytes / (mean_time_s * 1e9)  # GB/s
        efficiency = (bandwidth / CURRENT_GPU_SPECS["MEM_BW_GBS"]) * 100

        mod_8 = head_dim % 8 == 0
        mod_16 = head_dim % 16 == 0

        result = {
            "head_dim": head_dim,
            "mod_8": mod_8,
            "mod_16": mod_16,
            "total_bytes": total_bytes,
            "latency_ms": stats["mean"],
            "bandwidth_gbs": bandwidth,
            "bandwidth_efficiency_pct": efficiency,
        }
        results["measurements"].append(result)

        align_str = "✓16" if mod_16 else ("✓8" if mod_8 else "✗")
        print(f"  D={head_dim:3d} [{align_str:>3}]: {stats['mean']:.4f} ms, {bandwidth:.1f} GB/s ({efficiency:.1f}% peak)")

        del tensor
        torch.cuda.empty_cache()

    # Also test SDPA directly to see memory bandwidth effects
    print("\n  --- SDPA Memory Pattern ---")

    batch, n_heads = 4, 32

    for head_dim in [112, 113, 120, 121]:
        torch.manual_seed(seed)
        query = torch.randn(batch, n_heads, seq_len // 2, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, n_heads, seq_len // 2, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, n_heads, seq_len // 2, head_dim, dtype=dtype, device=device)

        def kernel_fn():
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)

        stats = benchmark_kernel(kernel_fn, warmup, measure, device)

        # Estimate memory traffic (very rough)
        qkv_bytes = 3 * batch * n_heads * (seq_len // 2) * head_dim * 2  # Q, K, V
        output_bytes = batch * n_heads * (seq_len // 2) * head_dim * 2
        total_bytes = qkv_bytes + output_bytes

        mean_time_s = stats["mean"] / 1000.0
        bandwidth = total_bytes / (mean_time_s * 1e9)

        mod_8 = head_dim % 8 == 0
        align_str = "✓8" if mod_8 else "✗"

        results["measurements"].append({
            "operation": "SDPA",
            "head_dim": head_dim,
            "mod_8": mod_8,
            "latency_ms": stats["mean"],
            "estimated_bandwidth_gbs": bandwidth,
        })

        print(f"  SDPA D={head_dim:3d} [{align_str:>2}]: {stats['mean']:.3f} ms, ~{bandwidth:.1f} GB/s")

        del query, key, value
        torch.cuda.empty_cache()

    return results


def analyze_vectorized_load_patterns(device: str = "cuda:0", seed: int = 42) -> dict:
    """
    Bonus: Analyze vectorized load pattern efficiency.

    CUDA vectorized loads:
    - float4 (16 bytes) - 4x float or 8x half
    - float2 (8 bytes) - 2x float or 4x half
    - scalar (4 bytes for float, 2 bytes for half)

    For FP16:
    - head_dim % 8 == 0: can use float4 loads (8 halfs = 16 bytes)
    - head_dim % 4 == 0: can use float2 loads (4 halfs = 8 bytes)
    - else: scalar loads
    """
    print("\n" + "="*60)
    print("H4 (Bonus): Vectorized Load Pattern Analysis")
    print("="*60)

    set_deterministic(seed)
    dtype = torch.float16

    # Test specific head_dims that show vectorization boundaries
    test_cases = [
        (104, "8x", "float4"),
        (105, "✗", "scalar"),
        (106, "✗", "scalar"),
        (107, "✗", "scalar"),
        (108, "4x", "float2"),
        (112, "16x", "float4"),
        (116, "4x", "float2"),
        (120, "8x", "float4"),
        (124, "4x", "float2"),
        (128, "16x", "float4"),
    ]

    M, N = 4096, 4096
    warmup = 30
    measure = 100

    results = {
        "description": "Vectorized load pattern efficiency",
        "measurements": [],
    }

    for head_dim, align_type, expected_load in test_cases:
        a, b = allocate_tensors((M, head_dim), (head_dim, N), dtype=dtype, device=device, seed=seed)

        def kernel_fn():
            torch.matmul(a, b)

        stats = benchmark_kernel(kernel_fn, warmup, measure, device)

        mean_time_s = stats["mean"] / 1000.0
        tflops = compute_gemm_tflops(M, N, head_dim, mean_time_s)

        result = {
            "K": head_dim,
            "alignment": align_type,
            "expected_load_type": expected_load,
            "mod_4": head_dim % 4 == 0,
            "mod_8": head_dim % 8 == 0,
            "mod_16": head_dim % 16 == 0,
            "latency_ms": stats["mean"],
            "tflops": tflops,
        }
        results["measurements"].append(result)

        print(f"  K={head_dim:3d} [{align_type:>3}] {expected_load:>6}: {stats['mean']:.3f} ms, {tflops:.1f} TFLOPS")

        del a, b
        torch.cuda.empty_cache()

    return results


def run_all_analyses(output_dir: Path, device: str = "cuda:0", seed: int = 42) -> dict:
    """Run all hardware layer analyses."""
    print("="*70)
    print("C2.3 Hardware Layer Analysis Experiment")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().isoformat()}")

    env = collect_environment()

    results = {
        "experiment": "C23_hardware_layer_analysis",
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "analyses": {},
    }

    # H1: Tensor Core utilization
    results["analyses"]["H1_tensor_core"] = analyze_tensor_core_utilization(device, seed)

    # H2: L2 cache efficiency
    results["analyses"]["H2_l2_cache"] = analyze_l2_cache_efficiency(device, seed)

    # H3: Memory bandwidth efficiency
    results["analyses"]["H3_bandwidth"] = analyze_memory_bandwidth_efficiency(device, seed)

    # H4 (bonus): Vectorized load patterns
    results["analyses"]["H4_vectorized_loads"] = analyze_vectorized_load_patterns(device, seed)

    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    h1 = results["analyses"]["H1_tensor_core"]["summary"]
    print(f"\nH1 (Tensor Core):")
    print(f"  - 16-aligned avg: {h1['aligned_16_avg_tflops']:.1f} TFLOPS")
    print(f"  - 8-aligned avg:  {h1['aligned_8_only_avg_tflops']:.1f} TFLOPS")
    print(f"  - Non-aligned avg: {h1['non_aligned_avg_tflops']:.1f} TFLOPS")
    if "non_aligned_slowdown_vs_16" in h1:
        print(f"  - Slowdown: {(h1['non_aligned_slowdown_vs_16']-1)*100:.1f}%")

    h2 = results["analyses"]["H2_l2_cache"]["summary"]
    print(f"\nH2 (L2 Cache):")
    print(f"  - 16-aligned avg bandwidth: {h2['aligned_16_avg_bw_gbs']:.1f} GB/s")
    print(f"  - Non-aligned avg bandwidth: {h2['non_aligned_avg_bw_gbs']:.1f} GB/s")
    print(f"  - Avg sector waste (non-aligned): {h2['avg_sector_waste_non_aligned_pct']:.1f}%")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "experiment": "C23_hardware_layer_analysis",
            "device": device,
            "seed": seed,
        }, f, indent=2)

    raw_path = output_dir / "raw.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    env_path = output_dir / "env.json"
    with open(env_path, 'w') as f:
        json.dump(env, f, indent=2)

    # Summary JSON
    summary = {
        "experiment": "C23_hardware_layer_analysis",
        "timestamp": results["timestamp"],
        "findings": {
            "H1_tensor_core": {
                "hypothesis": "Tensor Core requires K % 16 == 0 for optimal FP16 performance",
                "aligned_16_tflops": h1["aligned_16_avg_tflops"],
                "aligned_8_only_tflops": h1["aligned_8_only_avg_tflops"],
                "non_aligned_tflops": h1["non_aligned_avg_tflops"],
                "slowdown_pct": (h1.get("non_aligned_slowdown_vs_16", 1) - 1) * 100,
            },
            "H2_l2_cache": {
                "hypothesis": "L2 cache 32-byte sector causes over-fetch for non-aligned strides",
                "aligned_16_bandwidth_gbs": h2["aligned_16_avg_bw_gbs"],
                "non_aligned_bandwidth_gbs": h2["non_aligned_avg_bw_gbs"],
                "avg_sector_waste_pct": h2["avg_sector_waste_non_aligned_pct"],
            },
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="C2.3 Hardware Layer Analysis")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/C23/<timestamp>)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/C23/{timestamp}_C23_hardware_layer")
    else:
        output_dir = Path(args.output_dir)

    set_gpu_specs(args.device)
    run_all_analyses(output_dir, args.device, args.seed)

    # Print output path for sbatch script
    print(output_dir)


if __name__ == "__main__":
    main()
