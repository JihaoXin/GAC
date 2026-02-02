#!/bin/bash
#SBATCH --job-name=c5_e2e_v2
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_c5_e2e_v2.out
#SBATCH --error=slurm_logs/%j_c5_e2e_v2.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

set -e
source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "C5 End-to-End LLM Inference Comparison v2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "This version includes:"
echo "  - Fixed analyze_palu_dimensions() to detect per-head ranks"
echo "  - Fixed DimensionRepairer for PaLU's HeadwiseLowRankModule"
echo "  - Fixed memory leak by deleting palu_model before palu_repair"
echo ""

# Run C5 experiment
python scripts/run_c5_e2e_comparison.py \
    --out results/C5 \
    --repair-strategy minimal \
    --run-id "$(date +%Y%m%d_%H%M%S)_C5_e2e_v2_fixed"

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
