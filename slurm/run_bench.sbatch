#!/bin/bash
#SBATCH --job-name=GCompressBench
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j_bench.out
#SBATCH --error=slurm_logs/%j_bench.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa

set -e  # Exit on error

# Activate mamba environment
source ~/.bashrc
mamba activate gc
cd $SLURM_SUBMIT_DIR

# Set CUDA memory allocation settings
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run benchmark suite
# Options:
#   --experiment: gemm_projection, gemm_reduction, sdpa_backend, or all
#   --dtype: float16, bfloat16 (can specify multiple)
#   --warmup: number of warmup iterations (default: 10)
#   --iterations: number of measurement iterations (default: 100)

python -m scripts.run_benchmarks \
    --experiment all \
    --output-dir results/ \
    --dtype float16 bfloat16 \
    --warmup 10 \
    --iterations 100 \
    --device cuda:0

echo "=========================================="
echo "✅ Benchmark suite completed!"
echo "=========================================="

# Optional: Generate plots after benchmark completes
# python -m scripts.plot_results results/<experiment_name>/<timestamp>/
