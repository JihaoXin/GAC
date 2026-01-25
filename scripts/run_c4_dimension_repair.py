#!/usr/bin/env python3
"""
C4 Experiment: Validate Dimension Repair Algorithm

This script validates that the dimension repair algorithm can:
1. Correctly identify misaligned dimensions in PaLU-compressed models
2. Apply padding to restore alignment
3. Achieve performance recovery comparable to manual padding (P1 experiment)

Key metrics:
- Memory overhead: Should be <10% for minimal strategy
- Performance recovery: Should match P1 results (~30-34% faster)
- Correctness: Repaired dimensions must satisfy alignment constraints

Reference experiments:
- P1: Padding 107→112 gives 30.5% speedup with 4.7% memory overhead
- C23: 8-aligned enables float4 loads, 16-aligned enables optimal Tensor Core
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gcompress_bench.dimension_repair import (
    AlignmentStrategy,
    DimensionRepairer,
    RepairResult,
    ShapeContract,
    repair_dimension,
)


def test_repair_dimension_correctness() -> Dict[str, bool]:
    """Test repair_dimension function with known inputs."""
    tests = {}

    # Test minimal strategy (pad to 8)
    tests["minimal_107_to_112"] = repair_dimension(107, "minimal") == 112
    tests["minimal_113_to_120"] = repair_dimension(113, "minimal") == 120
    tests["minimal_120_unchanged"] = repair_dimension(120, "minimal") == 120

    # Test optimal strategy (pad to 16)
    tests["optimal_107_to_112"] = repair_dimension(107, "optimal") == 112
    tests["optimal_113_to_128"] = repair_dimension(113, "optimal") == 128
    tests["optimal_120_to_128"] = repair_dimension(120, "optimal") == 128
    tests["optimal_128_unchanged"] = repair_dimension(128, "optimal") == 128

    # Test predefined strategy
    tests["predefined_107_to_112"] = repair_dimension(107, "predefined") == 112
    tests["predefined_100_to_112"] = repair_dimension(100, "predefined") == 112
    tests["predefined_130_to_160"] = repair_dimension(130, "predefined") == 160

    # Test tradeoff strategy with different thresholds
    # 107→112 has 4.7% overhead, should be acceptable at 5%
    tests["tradeoff_107_5pct"] = repair_dimension(107, "tradeoff", max_overhead_pct=5.0) == 112
    # 107→128 has 19.6% overhead, should require higher threshold
    tests["tradeoff_107_1pct"] = repair_dimension(107, "tradeoff", max_overhead_pct=1.0) == 112

    # Test PaLU-typical dimensions (114-125 range)
    palu_dims = [114, 116, 117, 118, 120, 121, 122, 123, 124, 125]
    for dim in palu_dims:
        repaired = repair_dimension(dim, "minimal")
        aligned = repaired % 8 == 0
        tests[f"palu_dim_{dim}_aligned"] = aligned

    return tests


def test_shape_contract() -> Dict[str, bool]:
    """Test ShapeContract alignment checking."""
    contract = ShapeContract()
    tests = {}

    # Test alignment levels
    tests["is_8_aligned_120"] = contract.is_aligned(120, "minimal")
    tests["is_not_8_aligned_107"] = not contract.is_aligned(107, "minimal")
    tests["is_16_aligned_128"] = contract.is_aligned(128, "optimal")
    tests["is_not_16_aligned_120"] = not contract.is_aligned(120, "optimal")
    tests["is_predefined_128"] = contract.is_aligned(128, "predefined")
    tests["is_not_predefined_120"] = not contract.is_aligned(120, "predefined")

    # Test memory overhead calculation
    overhead = contract.memory_overhead(107, 112)
    tests["overhead_107_to_112"] = abs(overhead - 4.67) < 0.1

    overhead = contract.memory_overhead(107, 128)
    tests["overhead_107_to_128"] = abs(overhead - 19.63) < 0.1

    return tests


def analyze_palu_dimensions() -> Dict[str, any]:
    """
    Analyze PaLU compressed model dimensions and compute repair statistics.

    Returns statistics about:
    - Distribution of misaligned dimensions
    - Expected memory overhead for each strategy
    - Number of layers requiring repair
    """
    # PaLU Llama-3-8B r=0.8 dimension distribution (from C1 experiments)
    palu_dims_distribution = {
        114: 51,   # count of heads with this dim
        116: 89,
        117: 42,
        118: 68,
        120: 16,   # 8-aligned
        121: 73,
        122: 58,
        123: 47,
        124: 52,
        125: 16,
    }

    total_heads = sum(palu_dims_distribution.values())
    aligned_8 = sum(count for dim, count in palu_dims_distribution.items() if dim % 8 == 0)

    results = {
        "total_heads": total_heads,
        "aligned_8_original": aligned_8,
        "aligned_8_pct": 100.0 * aligned_8 / total_heads,
        "misaligned_pct": 100.0 * (total_heads - aligned_8) / total_heads,
    }

    # Compute repair statistics for each strategy
    for strategy in ["minimal", "optimal", "predefined"]:
        repaired_dims = {}
        total_original = 0
        total_repaired = 0

        for dim, count in palu_dims_distribution.items():
            target = repair_dimension(dim, strategy)
            repaired_dims[dim] = target
            total_original += dim * count
            total_repaired += target * count

        memory_overhead = 100.0 * (total_repaired - total_original) / total_original

        results[f"{strategy}_overhead_pct"] = round(memory_overhead, 2)
        results[f"{strategy}_repairs"] = repaired_dims

    return results


def benchmark_repair_strategies(
    dims: List[int],
    batch_size: int = 4,
    seq_len: int = 2048,
    num_heads: int = 32,
    warmup: int = 50,
    measure: int = 200,
) -> Dict[str, Dict[int, float]]:
    """
    Benchmark SDPA latency for original vs repaired dimensions.

    Args:
        dims: List of head_dim values to test
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        warmup: Warmup iterations
        measure: Measurement iterations

    Returns:
        Dict mapping strategy -> {dim: latency_ms}
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {}

    device = torch.device("cuda")
    results = {"original": {}, "minimal": {}, "optimal": {}}

    for dim in dims:
        # Test original dimension
        q = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(warmup):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(measure):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()

        latency_ms = start.elapsed_time(end) / measure
        results["original"][dim] = round(latency_ms, 3)

        # Test repaired dimensions for each strategy
        for strategy in ["minimal", "optimal"]:
            repaired_dim = repair_dimension(dim, strategy)
            if repaired_dim == dim:
                results[strategy][dim] = results["original"][dim]
                continue

            # Pad tensors to repaired dimension
            pad_size = repaired_dim - dim
            q_pad = torch.nn.functional.pad(q, (0, pad_size))
            k_pad = torch.nn.functional.pad(k, (0, pad_size))
            v_pad = torch.nn.functional.pad(v, (0, pad_size))

            # Warmup
            for _ in range(warmup):
                _ = torch.nn.functional.scaled_dot_product_attention(q_pad, k_pad, v_pad)
            torch.cuda.synchronize()

            # Measure
            start.record()
            for _ in range(measure):
                _ = torch.nn.functional.scaled_dot_product_attention(q_pad, k_pad, v_pad)
            end.record()
            torch.cuda.synchronize()

            latency_ms = start.elapsed_time(end) / measure
            results[strategy][dim] = round(latency_ms, 3)

        # Clean up
        del q, k, v
        torch.cuda.empty_cache()

    return results


def run_validation(args):
    """Run full validation suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{timestamp}_C4_dimension_repair"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "experiment": "C4_dimension_repair",
        "tests": {},
    }

    print("=" * 60)
    print("C4 Dimension Repair Validation")
    print("=" * 60)

    # 1. Test repair_dimension correctness
    print("\n[1/4] Testing repair_dimension() correctness...")
    tests = test_repair_dimension_correctness()
    passed = sum(tests.values())
    total = len(tests)
    print(f"  Passed: {passed}/{total}")
    for name, result in tests.items():
        status = "✓" if result else "✗"
        print(f"    {status} {name}")
    results["tests"]["repair_dimension"] = tests

    # 2. Test ShapeContract
    print("\n[2/4] Testing ShapeContract...")
    tests = test_shape_contract()
    passed = sum(tests.values())
    total = len(tests)
    print(f"  Passed: {passed}/{total}")
    for name, result in tests.items():
        status = "✓" if result else "✗"
        print(f"    {status} {name}")
    results["tests"]["shape_contract"] = tests

    # 3. Analyze PaLU dimensions
    print("\n[3/4] Analyzing PaLU dimension distribution...")
    palu_analysis = analyze_palu_dimensions()
    print(f"  Total heads: {palu_analysis['total_heads']}")
    print(f"  Originally 8-aligned: {palu_analysis['aligned_8_pct']:.1f}%")
    print(f"  Misaligned: {palu_analysis['misaligned_pct']:.1f}%")
    print(f"  Memory overhead (minimal): {palu_analysis['minimal_overhead_pct']:.2f}%")
    print(f"  Memory overhead (optimal): {palu_analysis['optimal_overhead_pct']:.2f}%")
    print(f"  Memory overhead (predefined): {palu_analysis['predefined_overhead_pct']:.2f}%")
    results["palu_analysis"] = palu_analysis

    # 4. Benchmark if GPU available and requested
    if args.benchmark and torch.cuda.is_available():
        print("\n[4/4] Benchmarking repair strategies on GPU...")
        test_dims = [107, 114, 117, 120, 121, 125]
        benchmark_results = benchmark_repair_strategies(
            dims=test_dims,
            warmup=args.warmup,
            measure=args.measure,
        )

        print("\n  SDPA Latency (ms):")
        print(f"  {'Dim':>6} | {'Original':>10} | {'Minimal':>10} | {'Optimal':>10} | {'Speedup':>10}")
        print("  " + "-" * 55)
        for dim in test_dims:
            orig = benchmark_results["original"].get(dim, 0)
            minimal = benchmark_results["minimal"].get(dim, 0)
            optimal = benchmark_results["optimal"].get(dim, 0)
            speedup = (orig - minimal) / orig * 100 if orig > 0 else 0
            print(f"  {dim:>6} | {orig:>10.3f} | {minimal:>10.3f} | {optimal:>10.3f} | {speedup:>9.1f}%")

        results["benchmark"] = benchmark_results
    else:
        print("\n[4/4] Skipping GPU benchmark (use --benchmark flag)")
        results["benchmark"] = None

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir}")

    # Summary
    all_tests = {}
    all_tests.update(results["tests"]["repair_dimension"])
    all_tests.update(results["tests"]["shape_contract"])
    total_passed = sum(all_tests.values())
    total_tests = len(all_tests)

    print(f"\nSummary: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("✓ All tests passed - dimension repair algorithm is correct")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="C4 Experiment: Validate Dimension Repair Algorithm"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/C4",
        help="Output directory for results",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run GPU benchmark (requires CUDA)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=200,
        help="Measurement iterations for benchmark",
    )

    args = parser.parse_args()
    return run_validation(args)


if __name__ == "__main__":
    sys.exit(main())
