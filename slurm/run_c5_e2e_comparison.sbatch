#!/bin/bash
#SBATCH --job-name=C5_e2e
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_C5_e2e_comparison.out
#SBATCH --error=slurm_logs/%j_C5_e2e_comparison.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

# C5 End-to-End LLM Inference Comparison
# Compares: Baseline vs PaLU vs PaLU+Repair
# Requires 80GB GPU for full LLM experiments

set -e

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

# Set up Python path
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR/third_party/palu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# Create output directories
mkdir -p results/C5
mkdir -p slurm_logs

# Parse arguments
SMOKE=""
REPAIR_STRATEGY="minimal"
SKIP_BASELINE=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE="--smoke"; shift;;
    --repair-strategy) REPAIR_STRATEGY="$2"; shift 2;;
    --skip-baseline) SKIP_BASELINE="--skip-baseline"; shift;;
    --run-id) RUN_ID="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=============================================="
echo "C5 End-to-End LLM Inference Comparison"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo "Repair Strategy: $REPAIR_STRATEGY"
echo "Smoke Test: ${SMOKE:-no}"
echo "=============================================="

# Run the comparison experiment
if [[ -n "$RUN_ID" ]]; then
  RUN_ID_ARG="--run-id $RUN_ID"
else
  RUN_ID_ARG=""
fi

python scripts/run_c5_e2e_comparison.py \
    --out results/C5 \
    --device cuda:0 \
    --dtype float16 \
    --repair-strategy "$REPAIR_STRATEGY" \
    $SMOKE \
    $SKIP_BASELINE \
    $RUN_ID_ARG

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "Results saved to: results/C5/"
echo "=============================================="
