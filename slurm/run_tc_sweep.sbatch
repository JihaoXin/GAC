#!/bin/bash
#SBATCH --job-name=tc_sweep
#SBATCH --output=slurm_logs/tc_sweep_%j.out
#SBATCH --error=slurm_logs/tc_sweep_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=spot

cd /home/xinj/GAC
source ~/.bashrc
conda activate gac

python scripts/tensor_core_alignment_sweep.py
