#!/bin/bash
#SBATCH --job-name=triton_inspect
#SBATCH --output=slurm_logs/triton_inspect_%j.out
#SBATCH --error=slurm_logs/triton_inspect_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=spot

set -e

echo "================================================"
echo "Triton Kernel Assembly Inspection"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo ""

cd /home/xinj/GAC

# Activate environment
source ~/.bashrc
conda activate gac

# Clear Triton cache to get fresh compilation
rm -rf ~/.triton/cache/*

echo ""
echo "Running Triton assembly inspector..."
echo ""

python scripts/triton_inspect_asm.py

echo ""
echo "================================================"
echo "Inspection complete"
echo "================================================"
