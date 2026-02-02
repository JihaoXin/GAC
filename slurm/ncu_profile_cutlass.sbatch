#!/bin/bash
#SBATCH --job-name=ncu_cutlass
#SBATCH --output=slurm_logs/ncu_cutlass_%j.out
#SBATCH --error=slurm_logs/ncu_cutlass_%j.err
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

METRICS="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_driver"

# No kernel-name filter — capture ALL kernels for misaligned dims
# N misaligned
for N in 1025 1026 1033 1034; do
    echo "========================================"
    echo "  M=2048  N=$N  K=128  (N misaligned)"
    echo "========================================"
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        python scripts/ncu_profile_kernels.py --M 2048 --N $N --K 128 2>&1
    echo ""
done

# K misaligned
for K in 65 66 73 74 105 106 113 114 127; do
    echo "========================================"
    echo "  M=2048  N=2048  K=$K  (K misaligned)"
    echo "========================================"
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        python scripts/ncu_profile_kernels.py --M 2048 --N 2048 --K $K 2>&1
    echo ""
done
