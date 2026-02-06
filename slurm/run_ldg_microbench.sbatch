#!/bin/bash
#SBATCH --job-name=ldg_bench
#SBATCH --output=slurm_logs/ldg_bench_%j.out
#SBATCH --error=slurm_logs/ldg_bench_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate llama
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd $SLURM_SUBMIT_DIR

echo "============================================="
echo "LDG Instruction Microbenchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================="
echo ""

nvidia-smi
echo ""

python scripts/ldg_microbench.py \
    --output results/ldg_microbench \
    --warmup 50 \
    --repeats 200 \
    --experiments all \
    2>&1

echo ""
echo "============================================="
echo "Benchmark Complete: $(date)"
echo "============================================="
