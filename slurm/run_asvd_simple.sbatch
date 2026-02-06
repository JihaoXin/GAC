#!/bin/bash
#SBATCH --job-name=asvd_sim
#SBATCH --output=slurm_logs/asvd_simple_%j.out
#SBATCH --error=slurm_logs/asvd_simple_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ASVD Simple - Uses original ASVD code structure
# Compares rank_align=1 vs rank_align=8

set -e

echo "============================================="
echo "ASVD Simple Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================="

source ~/.bashrc
conda activate svdllm

echo ""
echo "Environment:"
python --version
nvidia-smi

cd /home/xinj/GAC/third_party/ASVD4LLM
mkdir -p cache

# Parameters
MODEL="meta-llama/Meta-Llama-3-8B"
RATIO=0.85
CALIB_SAMPLES=32
ALPHA=0.5

echo ""
echo "============================================="
echo "Step 1: Evaluate baseline model"
echo "============================================="
python asvd_simple.py \
    --model_id $MODEL \
    --raw_model \
    --eval_ppl wikitext2 \
    --seq_len 512 \
    --output /home/xinj/GAC/results/asvd_simple/baseline

echo ""
echo "============================================="
echo "Step 2: ASVD with rank_align=1 (unaligned)"
echo "============================================="
python asvd_simple.py \
    --model_id $MODEL \
    --param_ratio_target $RATIO \
    --n_calib_samples $CALIB_SAMPLES \
    --calib_dataset wikitext2 \
    --scaling_method abs_mean \
    --sensitivity_metric ppl \
    --act_aware \
    --alpha $ALPHA \
    --sigma_fuse UV \
    --rank_align 1 \
    --use_cache \
    --eval_ppl wikitext2 \
    --seq_len 512 \
    --output /home/xinj/GAC/results/asvd_simple/unaligned

echo ""
echo "============================================="
echo "Step 3: ASVD with rank_align=8 (aligned)"
echo "============================================="
python asvd_simple.py \
    --model_id $MODEL \
    --param_ratio_target $RATIO \
    --n_calib_samples $CALIB_SAMPLES \
    --calib_dataset wikitext2 \
    --scaling_method abs_mean \
    --sensitivity_metric ppl \
    --act_aware \
    --alpha $ALPHA \
    --sigma_fuse UV \
    --rank_align 8 \
    --use_cache \
    --eval_ppl wikitext2 \
    --seq_len 512 \
    --output /home/xinj/GAC/results/asvd_simple/aligned

echo ""
echo "============================================="
echo "Experiment Complete: $(date)"
echo "============================================="

# Summarize results
echo ""
echo "Results Summary:"
cat /home/xinj/GAC/results/asvd_simple/baseline/results.json 2>/dev/null || echo "No baseline results"
cat /home/xinj/GAC/results/asvd_simple/unaligned/results.json 2>/dev/null || echo "No unaligned results"
cat /home/xinj/GAC/results/asvd_simple/aligned/results.json 2>/dev/null || echo "No aligned results"
