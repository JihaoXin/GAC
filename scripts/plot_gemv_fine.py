#!/usr/bin/env python3
"""Plot GEMV fine-grained sweep (step=1)."""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('results/gemv_fine_sweep.json') as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K sweep
ax1 = axes[0]
k_data = data['K_sweep']
K_vals = [d['K'] for d in k_data]
K_times = [d['mean_ms'] * 1000 for d in k_data]  # Convert to μs

ax1.plot(K_vals, K_times, 'b-', linewidth=0.8, alpha=0.8)
ax1.set_xlabel('K Dimension', fontsize=12)
ax1.set_ylabel('Latency (μs)', fontsize=12)
ax1.set_title('GEMV K Sweep (N=4096 fixed, step=1)', fontsize=13)
ax1.grid(True, alpha=0.3)

# Mark mod-8 and mod-16 points
for i, K in enumerate(K_vals):
    if K % 128 == 0:
        ax1.axvline(x=K, color='r', alpha=0.3, linewidth=0.5)

# N sweep
ax2 = axes[1]
n_data = data['N_sweep']
N_vals = [d['N'] for d in n_data]
N_times = [d['mean_ms'] * 1000 for d in n_data]

ax2.plot(N_vals, N_times, 'b-', linewidth=0.8, alpha=0.8)
ax2.set_xlabel('N Dimension', fontsize=12)
ax2.set_ylabel('Latency (μs)', fontsize=12)
ax2.set_title('GEMV N Sweep (K=4096 fixed, step=1)', fontsize=13)
ax2.grid(True, alpha=0.3)

for i, N in enumerate(N_vals):
    if N % 128 == 0:
        ax2.axvline(x=N, color='r', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('Latex/figures/fig_gemv_fine_sweep.png', dpi=150, bbox_inches='tight')
plt.savefig('Latex/figures/fig_gemv_fine_sweep.pdf', bbox_inches='tight')
print("Saved to Latex/figures/fig_gemv_fine_sweep.png")

# Statistics
print(f"\nK sweep: min={min(K_times):.2f}μs, max={max(K_times):.2f}μs, range={max(K_times)-min(K_times):.2f}μs")
print(f"N sweep: min={min(N_times):.2f}μs, max={max(N_times):.2f}μs, range={max(N_times)-min(N_times):.2f}μs")
