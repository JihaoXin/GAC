#!/bin/bash
#SBATCH --job-name=C4_DimRepair
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j_C4_dim_repair.out
#SBATCH --error=slurm_logs/%j_C4_dim_repair.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa

set -e

source ~/.bashrc
mamba activate gc
cd $SLURM_SUBMIT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "C4 Experiment: Dimension Repair Validation"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run validation with benchmark
python scripts/run_c4_dimension_repair.py \
    --output results/C4 \
    --benchmark \
    --warmup 50 \
    --measure 200

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
