#!/bin/bash
#SBATCH --job-name=gac_eval
#SBATCH --output=slurm_logs/gac_eval_%j.out
#SBATCH --error=slurm_logs/gac_eval_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=06:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

echo "============================================="
echo "GAC Recompression & Perplexity Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================="
echo ""

# Install fast_hadamard_transform if needed (for PaLU checkpoint loading)
pip install fast_hadamard_transform 2>/dev/null || echo "fast_hadamard_transform install skipped (not critical)"

# Step 1: Run GAC rank allocation (generates rank configs, no GPU needed)
echo ">>> Step 1: Running GAC rank allocation..."
python3 scripts/gac_rank_allocation.py --output results/gac_allocation 2>&1
echo ""

# Step 2: Run recompression + perplexity evaluation for each strategy
# Each strategy loads the base model, compresses with SVD, evaluates PPL
# Note: --include-palu may fail if fast_hadamard_transform isn't available
echo ">>> Step 2: Running recompression + perplexity evaluation..."
python3 scripts/gac_recompress_eval.py \
    --rank-dir results/gac_allocation \
    --output results/gac_eval \
    --strategies unaligned,round8,gac_dp \
    --include-baseline \
    --include-palu \
    --device cuda:0 2>&1

echo ""
echo "============================================="
echo "GAC Experiment Complete"
echo "============================================="
