#!/bin/bash
#SBATCH --job-name=unified_eval
#SBATCH --output=slurm_logs/unified_eval_%j.out
#SBATCH --error=slurm_logs/unified_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================="
echo "Unified Evaluation: Accuracy + Decode Latency"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "============================================="

# Activate conda environment
source /home/xinj/miniforge3/etc/profile.d/conda.sh
conda activate svdllm

# Fix lm_eval library conflict
export LD_LIBRARY_PATH=/home/xinj/miniforge3/envs/svdllm/lib:$LD_LIBRARY_PATH

# Verify lm_eval works
python -c "import lm_eval; print('lm_eval OK')"

cd /home/xinj/GAC

# Create output directory
mkdir -p results/unified_eval
mkdir -p slurm_logs

echo ""
echo "============================================="
echo "Running unified evaluation..."
echo "============================================="

# Run evaluation for all methods
# Can customize which to evaluate with --eval flag
# Skip baseline (already completed in job 18844)
python scripts/eval_all.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --output results/unified_eval \
    --eval asvd_unaligned asvd_aligned llmpruner_pruned llmpruner_pruned_r8

echo ""
echo "============================================="
echo "Experiment Complete: $(date)"
echo "============================================="
