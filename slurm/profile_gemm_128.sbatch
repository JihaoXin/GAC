#!/bin/bash
#SBATCH --job-name=profile_gemm128
#SBATCH --output=slurm_logs/profile_gemm128_%j.out
#SBATCH --error=slurm_logs/profile_gemm128_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

# Activate environment
source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

echo "=== Step 1: Basic timing + tile analysis ==="
python scripts/profile_gemm_128.py --sizes 128

echo ""
echo "=== Step 2: nsys profile for grid/block dimensions ==="
# Use nsys to capture kernel launch configs
nsys profile --stats=true -o /tmp/gemm128_profile \
    python scripts/profile_gemm_128.py --sizes 128 2>/dev/null || \
    echo "nsys not available, skipping"

echo ""
echo "=== Step 3: Irregular vs aligned dimensions ==="
python scripts/profile_gemm_128.py --sizes 107 112 128

echo ""
echo "=== Step 4: Comparison across sizes ==="
python scripts/profile_gemm_128.py --sizes 64 128 256 512 1024 2048 4096
