#!/bin/bash
#SBATCH --job-name=prof_rank
#SBATCH --output=slurm_logs/profile_palu_ranks_%j.out
#SBATCH --error=slurm_logs/profile_palu_ranks_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs results

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

echo "============================================="
echo "Profile GEMM at PaLU rank dimensions (K=8..512)"
echo "============================================="

# Profile GEMM with K dimension matching PaLU rank range
# M=1 (decode) and M=256 (prefill) scenarios, N=512 (group_dim)
# This gives actual latency data for the rank dimensions used in our GAC allocation

python3 -c "
import torch
import time
import csv

def bench_gemm(M, N, K, dtype=torch.float16, warmup=50, trials=200):
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    # Warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()
    # Measure
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    for i in range(trials):
        start_events[i].record()
        torch.mm(A, B)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # us
    return sum(times) / len(times)

# PaLU GEMM shapes:
# U matrix: [batch*seq, rank] @ [rank, group_dim] → K=rank, N=group_dim(512)
# VT matrix: [batch*seq, 4096] @ [4096, sum_ranks] → K=4096, N=sum_ranks
# We profile the U matrix GEMM since rank is the variable dimension

outfile = 'results/palu_rank_profile.csv'
with open(outfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['M', 'N', 'K', 'time_us', 'tflops', 'aligned_mod8'])

    # Sweep K (rank) from 8 to 512
    for M in [1, 256]:  # decode and prefill batch sizes
        N = 512  # group_dim
        for K in range(8, 513):
            t = bench_gemm(M, N, K)
            flops = 2 * M * N * K
            tflops = flops / (t * 1e-6) / 1e12
            aligned = (K % 8 == 0)
            writer.writerow([M, N, K, f'{t:.3f}', f'{tflops:.4f}', aligned])
            if K % 64 == 0:
                print(f'M={M} N={N} K={K}: {t:.3f} us, {tflops:.4f} TFLOPS, aligned={aligned}')

print(f'Results saved to {outfile}')
print('Done!')
"
