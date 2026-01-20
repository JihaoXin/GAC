"""CLI entrypoint for running experiments from spec files."""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_runner import load_experiment_spec, run_experiment
from src.environment import collect_environment


def generate_run_id(exp_name: str) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{exp_name}"


def save_results(results: dict, output_dir: Path, run_id: str):
    """Save experiment results in the required structure."""
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(results["config"], f, indent=2)
    
    # Save raw results
    raw_path = run_dir / "raw.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary = {
        "experiment": results["experiment"],
        "num_measurements": len(results.get("measurements", [])),
        "successful": len([m for m in results.get("measurements", []) if "error" not in m]),
        "failed": len([m for m in results.get("measurements", []) if "error" in m]),
    }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save environment
    env_path = run_dir / "env.json"
    env = collect_environment()
    with open(env_path, 'w') as f:
        json.dump(env, f, indent=2)
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment from spec file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["run"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to experiment spec YAML file"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name from spec file"
    )
    
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory for results"
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (default: auto-generated)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Load experiment spec
        exp_spec = load_experiment_spec(args.spec, args.name)
        
        # Generate run ID
        run_id = args.run_id if args.run_id else generate_run_id(args.name)
        
        # Determine output directory (use experiment name as group)
        exp_group = args.name.split("_")[0]  # S1, S2, G3, etc.
        output_dir = args.results_root / exp_group
        
        print("=" * 70)
        print(f"Running experiment: {args.name}")
        print(f"Run ID: {run_id}")
        print(f"Output directory: {output_dir / run_id}")
        print("=" * 70)
        print()
        
        # Run experiment
        results = run_experiment(exp_spec, device=args.device, seed=args.seed)
        results["metadata"] = {
            "run_id": run_id,
            "experiment_name": args.name,
            "device": args.device,
            "seed": args.seed,
        }
        
        # Save results
        run_dir = save_results(results, output_dir, run_id)
        
        print("=" * 70)
        print("✅ Experiment completed!")
        print(f"Results saved to: {run_dir}")
        print("=" * 70)
        print()
        
        # Print final results path (for Slurm scripts)
        print(run_dir.absolute())


if __name__ == "__main__":
    main()
