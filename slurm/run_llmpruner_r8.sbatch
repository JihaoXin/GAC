#!/bin/bash
#SBATCH --job-name=llmpruner_r8
#SBATCH --output=slurm_logs/llmpruner_r8_%j.out
#SBATCH --error=slurm_logs/llmpruner_r8_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================="
echo "LLM-Pruner pruned_r8 Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "============================================="

source /home/xinj/miniforge3/etc/profile.d/conda.sh
conda activate svdllm

cd /home/xinj/GAC
mkdir -p results/unified_eval slurm_logs

python scripts/eval_llmpruner_r8.py

echo ""
echo "============================================="
echo "Complete: $(date)"
echo "============================================="
