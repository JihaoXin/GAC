#!/bin/bash
#SBATCH --job-name=NightSweep
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa

set -e  # Exit on error

# ============================================
# USER: Edit environment setup below
# ============================================
# Activate mamba environment
source ~/.bashrc
mamba activate gc
cd $SLURM_SUBMIT_DIR

# Set reproducibility environment variables
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
# Optional: For deterministic GEMM (may reduce performance)
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Parse arguments
EXP_NAME=""
RESULTS_ROOT="results"
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)
            EXP_NAME="$2"
            shift 2
            ;;
        --results_root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    echo "Error: --exp argument required"
    exit 1
fi

echo "=========================================="
echo "Running experiment: $EXP_NAME"
echo "Results root: $RESULTS_ROOT"
echo "Run ID: $RUN_ID"
echo "=========================================="

# Run experiment and capture output directory
OUTPUT_DIR=$(python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name "$EXP_NAME" \
    --results-root "$RESULTS_ROOT" \
    ${RUN_ID:+--run-id "$RUN_ID"} \
    --device cuda:0 2>&1 | tail -1)

# Generate plots
if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "Generating plots..."
    python -m scripts.plot_night_sweep "$OUTPUT_DIR"
fi

echo "=========================================="
echo "✅ Experiment completed!"
echo "=========================================="
