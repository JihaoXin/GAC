#!/bin/bash
#SBATCH --job-name=asvd_decode
#SBATCH --output=slurm_logs/asvd_decode_%j.out
#SBATCH --error=slurm_logs/asvd_decode_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "============================================="
echo "ASVD Decode Latency Re-test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo "============================================="

source /home/xinj/miniforge3/etc/profile.d/conda.sh
conda activate svdllm

cd /home/xinj/GAC
mkdir -p results/asvd_decode_retest slurm_logs

python scripts/asvd_decode_retest.py

echo ""
echo "============================================="
echo "Complete: $(date)"
echo "============================================="
