"""Validate that night sweep setup is complete and ready."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_file(path, description):
    """Check if file exists."""
    p = Path(path)
    if p.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} MISSING")
        return False

def check_import(module, description):
    """Check if module can be imported."""
    try:
        __import__(module)
        print(f"✓ {description}: {module}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module} - {e}")
        return False

print("=" * 70)
print("Night Sweep Setup Validation")
print("=" * 70)
print()

all_ok = True

# Check files
print("Files:")
all_ok &= check_file("experiments/night_sweep.yaml", "Experiment spec")
all_ok &= check_file("slurm/run_experiment.sbatch", "Slurm script")
all_ok &= check_file("slurm/launch_all.sh", "Launch script")
all_ok &= check_file("scripts/run_experiment.py", "CLI entrypoint")
all_ok &= check_file("scripts/plot_night_sweep.py", "Plotting script")
print()

# Check imports
print("Python modules:")
all_ok &= check_import("yaml", "PyYAML")
all_ok &= check_import("src.experiment_runner", "Experiment runner")
all_ok &= check_import("src.utils", "Utils")
all_ok &= check_import("src.measurement", "Measurement")
print()

# Check experiment spec
print("Experiment specs:")
try:
    from src.experiment_runner import load_experiment_spec, EXPERIMENT_RUNNERS
    
    experiments = [
        "S1_sdpa_dense_sweep",
        "S2_sdpa_backend_forced",
        "G3_gemm_k_dense",
        "G4_gemm_n_dense_projectionlike",
        "P1_padding_rescue",
        "HET1_head_hetero_batching_penalty"
    ]
    
    for exp_name in experiments:
        try:
            spec = load_experiment_spec("experiments/night_sweep.yaml", exp_name)
            exp_type = spec["type"]
            if exp_type in EXPERIMENT_RUNNERS:
                print(f"✓ {exp_name}: type={exp_type}")
            else:
                print(f"✗ {exp_name}: type={exp_type} (no runner)")
                all_ok = False
        except Exception as e:
            print(f"✗ {exp_name}: {e}")
            all_ok = False
except Exception as e:
    print(f"✗ Error checking specs: {e}")
    all_ok = False

print()
print("=" * 70)
if all_ok:
    print("✅ All checks passed! Ready to run experiments.")
else:
    print("❌ Some checks failed. Please fix issues above.")
print("=" * 70)

sys.exit(0 if all_ok else 1)
