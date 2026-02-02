#!/bin/bash
#SBATCH --job-name=profile_align
#SBATCH --output=slurm_logs/profile_align_%j.out
#SBATCH --error=slurm_logs/profile_align_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

python scripts/profile_alignment.py --sweep --sweep-csv results/alignment_sweep.csv
