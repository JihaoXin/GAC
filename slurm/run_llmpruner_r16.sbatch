#!/bin/bash
#SBATCH --job-name=llmprune_r16
#SBATCH --output=slurm_logs/llmprune_r16_%j.out
#SBATCH --error=slurm_logs/llmprune_r16_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

# Use llama env python directly (avoids conda activate issues in batch)
PYTHON=/home/xinj/miniforge3/envs/llama/bin/python
export LD_LIBRARY_PATH="/home/xinj/miniforge3/envs/llama/lib:$LD_LIBRARY_PATH"
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd $SLURM_SUBMIT_DIR

echo "============================================="
echo "LLM-Pruner + GAC: round_to=8 vs round_to=16"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================="
echo ""
nvidia-smi
echo ""

# Run LLM-Pruner experiment with all 4 strategies:
# baseline, pruned (no rounding), pruned_r8, pruned_r16
$PYTHON scripts/llmpruner_gac_experiment.py \
    --pruning-ratio 0.25 \
    --output results/llmpruner_llama3_r16 \
    --device cuda \
    --eval-accuracy \
    --accuracy-tasks piqa,hellaswag \
    --accuracy-limit 200 \
    --eval-latency \
    --seq-lens 128,256,512,1024 \
    --prefill-warmup 5 \
    --prefill-repeats 30 \
    --decode-prompt-len 128 \
    --decode-gen-tokens 64 \
    --decode-warmup 3 \
    --decode-repeats 10 \
    2>&1

echo ""
echo "============================================="
echo "LLM-Pruner r16 Experiment Complete: $(date)"
echo "============================================="
