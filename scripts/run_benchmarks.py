"""Main CLI entrypoint for running benchmarks."""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, GEMMConfig, SDPAConfig
from src.benchmark_gemm import run_gemm_benchmark, save_gemm_results
from src.benchmark_sdpa import run_sdpa_benchmark, save_sdpa_results
from src.environment import save_environment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU Irregular Dimensions Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["gemm_projection", "gemm_reduction", "sdpa_backend", "all"],
        default="all",
        help="Which experiment to run"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16"],
        help="Data types to test"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measurement iterations"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # GEMM-specific options
    parser.add_argument(
        "--gemm-m-values",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384],
        help="M values for GEMM (batch × tokens)"
    )
    
    parser.add_argument(
        "--gemm-k-values",
        nargs="+",
        type=int,
        default=[96, 104, 107, 112, 120, 128],
        help="K values for GEMM (dimension to vary)"
    )
    
    parser.add_argument(
        "--gemm-n-values",
        nargs="+",
        type=int,
        default=[96, 104, 107, 112, 120, 128],
        help="N values for GEMM (head_dim-like)"
    )
    
    parser.add_argument(
        "--gemm-reduction-k",
        action="store_true",
        help="Run reduction dimension test (K as reduction dim)"
    )
    
    # SDPA-specific options
    parser.add_argument(
        "--sdpa-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8],
        help="Batch sizes for SDPA"
    )
    
    parser.add_argument(
        "--sdpa-seq-lengths",
        nargs="+",
        type=int,
        default=[1024, 4096],
        help="Sequence lengths for SDPA"
    )
    
    parser.add_argument(
        "--sdpa-n-heads",
        type=int,
        default=32,
        help="Number of attention heads"
    )
    
    parser.add_argument(
        "--sdpa-head-dims",
        nargs="+",
        type=int,
        default=[96, 104, 107, 112, 120, 128],
        help="Head dimensions to test"
    )
    
    return parser.parse_args()


def main():
    """Main entrypoint."""
    args = parse_args()
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment if args.experiment != "all" else "full_suite"
    exp_dir = args.output_dir / experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GPU Irregular Dimensions Benchmark Suite")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Output directory: {exp_dir}")
    print(f"Device: {args.device}")
    print(f"Dtypes: {args.dtype}")
    print("=" * 60)
    print()
    
    # Save environment metadata
    print("Collecting environment metadata...")
    env_path = exp_dir / "env.json"
    save_environment(env_path)
    print(f"Environment metadata saved to {env_path}")
    print()
    
    # Create experiment config
    exp_config = ExperimentConfig(output_dir=exp_dir)
    
    # Configure GEMM
    if args.experiment in ["gemm_projection", "gemm_reduction", "all"]:
        gemm_config = GEMMConfig(
            experiment_name="gemm_sweep",
            device=args.device,
            dtypes=args.dtype,
            warmup_iterations=args.warmup,
            measurement_iterations=args.iterations,
            seed=args.seed,
            m_values=args.gemm_m_values,
            k_values=args.gemm_k_values,
            n_values=args.gemm_n_values,
            reduction_k=args.gemm_reduction_k or args.experiment == "gemm_reduction",
            test_projection_qkv=True,
            test_projection_output=False,
        )
        exp_config.gemm_config = gemm_config
        exp_config.run_gemm = True
    else:
        exp_config.run_gemm = False
    
    # Configure SDPA
    if args.experiment in ["sdpa_backend", "all"]:
        sdpa_config = SDPAConfig(
            experiment_name="sdpa_backend",
            device=args.device,
            dtypes=args.dtype,
            warmup_iterations=args.warmup,
            measurement_iterations=args.iterations,
            seed=args.seed,
            batch_sizes=args.sdpa_batch_sizes,
            seq_lengths=args.sdpa_seq_lengths,
            n_heads=args.sdpa_n_heads,
            head_dims=args.sdpa_head_dims,
        )
        exp_config.sdpa_config = sdpa_config
        exp_config.run_sdpa = True
    else:
        exp_config.run_sdpa = False
    
    # Save configuration
    config_path = exp_dir / "config.json"
    exp_config.save(config_path)
    print(f"Configuration saved to {config_path}")
    print()
    
    # Run GEMM benchmark
    if exp_config.run_gemm:
        print("=" * 60)
        print("Running GEMM Benchmark")
        print("=" * 60)
        gemm_results = run_gemm_benchmark(exp_config.gemm_config, exp_dir)
        gemm_path = exp_dir / "gemm_results.json"
        save_gemm_results(gemm_results, gemm_path)
        print(f"\nGEMM results saved to {gemm_path}")
        print()
    
    # Run SDPA benchmark
    if exp_config.run_sdpa:
        print("=" * 60)
        print("Running SDPA Benchmark")
        print("=" * 60)
        sdpa_results = run_sdpa_benchmark(exp_config.sdpa_config, exp_dir)
        sdpa_path = exp_dir / "sdpa_results.json"
        save_sdpa_results(sdpa_results, sdpa_path)
        print(f"\nSDPA results saved to {sdpa_path}")
        print()
    
    print("=" * 60)
    print("✅ Benchmark suite completed!")
    print(f"Results directory: {exp_dir}")
    print("=" * 60)
    
    # Suggest next steps
    print("\nNext steps:")
    print(f"  1. Review results: {exp_dir}")
    print(f"  2. Generate plots: python -m scripts.plot_results {exp_dir}")
    print()


if __name__ == "__main__":
    main()
