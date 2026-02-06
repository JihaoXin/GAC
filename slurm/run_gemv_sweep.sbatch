#!/bin/bash
#SBATCH --job-name=gemv_sweep
#SBATCH --output=slurm_logs/gemv_sweep_%j.out
#SBATCH --error=slurm_logs/gemv_sweep_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "============================================="
echo "GEMV Dimension Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo "============================================="

source /home/xinj/miniforge3/etc/profile.d/conda.sh
conda activate svdllm

cd /home/xinj/GAC
mkdir -p results slurm_logs

python scripts/gemv_dim_sweep.py --repeats 100

echo ""
echo "============================================="
echo "Complete: $(date)"
echo "============================================="
