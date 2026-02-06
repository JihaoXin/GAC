#!/bin/bash
#SBATCH --job-name=gemv_fine
#SBATCH --output=slurm_logs/gemv_fine_%j.out
#SBATCH --error=slurm_logs/gemv_fine_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --mem=32G

source /home/xinj/miniforge3/etc/profile.d/conda.sh
conda activate svdllm
cd /home/xinj/GAC

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python scripts/gemv_fine_sweep.py

echo "Done: $(date)"
