#!/bin/bash
#SBATCH --job-name=rank_scores
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j_rank_%a.out
#SBATCH --error=slurm_logs/%j_rank_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot
#SBATCH --array=0-1

set -e

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

export PYTORCH_ALLOC_CONF=expandable_segments:True

MODELS=("meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-v0.3")
NAMES=("llama3_8b" "mistral_7b")

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Task $SLURM_ARRAY_TASK_ID: $MODEL ==="

# Step 1: Compute scores
python scripts/compute_rank_scores.py \
    --model "$MODEL" \
    --output "results/rank_scores/${NAME}.json" \
    --device cuda \
    --nsamples 32 \
    --seqlen 1024

echo "Scores computed for $NAME"

# Step 2: Plot (only on task 0 after all tasks done, or run separately)
# python scripts/plot_dimension_distribution.py \
#     --scores results/rank_scores/llama3_8b.json results/rank_scores/mistral_7b.json \
#     --ratios 0.5 0.6 0.7 0.8 0.9 \
#     --out results/rank_distribution/plots

echo "=== Done: $NAME ==="
