#!/bin/bash
#SBATCH --job-name=ncu_profile
#SBATCH --output=slurm_logs/ncu_profile_%j.out
#SBATCH --error=slurm_logs/ncu_profile_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

# Profile key M transition points: 1088 (Kernel A), 1089 (Kernel B), 1153 (Kernel C), 1728, 1729
# Collect: grid size, block size, registers, shared memory, kernel name
METRICS="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_driver"

for M in 1024 1088 1089 1152 1153 1280 1536 1728 1729 2048; do
    echo "========================================"
    echo "  M=$M  N=2048  K=128"
    echo "========================================"
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        --kernel-name regex:'gemm|cutlass|xmma' \
        python scripts/ncu_profile_kernels.py --M $M --N 2048 --K 128 2>&1
    echo ""
done
