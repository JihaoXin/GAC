#!/bin/bash
#SBATCH --job-name=l2_dense
#SBATCH --output=slurm_logs/l2_dense_%j.out
#SBATCH --error=slurm_logs/l2_dense_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=spot

cd /home/xinj/GAC
source ~/.bashrc
conda activate gac

python scripts/l2_sweep_dense.py
