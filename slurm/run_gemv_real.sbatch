#!/bin/bash
#SBATCH --job-name=gemv_real
#SBATCH --output=slurm_logs/gemv_real_%j.out
#SBATCH --error=slurm_logs/gemv_real_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "============================================="
echo "GEMV Benchmark with Real ASVD Dimensions"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo "============================================="

source /home/xinj/miniforge3/etc/profile.d/conda.sh
conda activate svdllm

cd /home/xinj/GAC
mkdir -p results slurm_logs

python scripts/gemv_real_dims.py --param_ratio 0.85 --repeats 200

echo ""
echo "============================================="
echo "Complete: $(date)"
echo "============================================="
