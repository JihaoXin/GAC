#!/bin/bash
#SBATCH --job-name=asvd_ppl
#SBATCH --output=slurm_logs/asvd_ppl_%j.out
#SBATCH --error=slurm_logs/asvd_ppl_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ASVD + GAC Alignment Experiment with PPL-based sensitivity
# This uses ASVD's recommended PPL-based sensitivity calibration
# Compares rank_align=1 (unaligned) vs rank_align=8 (aligned)

set -e

echo "============================================="
echo "ASVD + GAC Alignment Experiment (PPL Sensitivity)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================="

# Setup environment
source ~/.bashrc
conda activate svdllm

# Show environment
echo ""
echo "Environment:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi

cd /home/xinj/GAC

# Create output directory
mkdir -p results/asvd_ppl_15pct
mkdir -p slurm_logs
mkdir -p cache

# Run experiment with 15% compression using PPL-based sensitivity
python scripts/asvd_alignment_experiment.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --param_ratio_target 0.85 \
    --n_calib_samples 32 \
    --calib_dataset wikitext2 \
    --scaling_method abs_mean \
    --sensitivity_metric ppl \
    --act_aware \
    --alpha 0.5 \
    --sigma_fuse UV \
    --seed 42 \
    --seq_len 512 \
    --eval_limit 200 \
    --output results/asvd_ppl_15pct

echo ""
echo "============================================="
echo "Experiment Complete: $(date)"
echo "============================================="
