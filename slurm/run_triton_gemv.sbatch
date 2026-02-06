#!/bin/bash
#SBATCH --job-name=triton_gemv
#SBATCH --output=slurm_logs/triton_gemv_%j.out
#SBATCH --error=slurm_logs/triton_gemv_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate llama
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd $SLURM_SUBMIT_DIR

echo "============================================="
echo "Triton GeMV Alignment Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================="
echo ""

nvidia-smi
echo ""

# Check Triton installation
python -c "import triton; print(f'Triton version: {triton.__version__}')" 2>&1 || echo "Warning: Triton not available"

echo ""
echo "Running Triton GeMV benchmark..."
echo ""

python scripts/triton_gemv_bench.py \
    --output results/triton_gemv \
    --warmup 50 \
    --repeats 200 \
    --block-n 64 \
    --block-k 64 \
    --experiments all \
    2>&1

echo ""
echo "============================================="
echo "Benchmark Complete: $(date)"
echo "============================================="
