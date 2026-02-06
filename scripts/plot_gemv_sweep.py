#!/usr/bin/env python3
"""Plot GEMV dimension sweep results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
with open('results/gemv_dim_sweep.json') as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# K sweep
ax1 = axes[0]
k_data = data['K_sweep']
K_vals = [d['K'] for d in k_data]
K_times = [d['mean_ms'] * 1000 for d in k_data]  # Convert to microseconds
K_bw = [d['bandwidth_gbs'] for d in k_data]

ax1.plot(K_vals, K_times, 'b.-', linewidth=1.5, markersize=4)
ax1.set_xlabel('K Dimension', fontsize=12)
ax1.set_ylabel('Latency (μs)', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('GEMV K Sweep (N=4096 fixed)', fontsize=13)
ax1.grid(True, alpha=0.3)

# Add bandwidth on secondary axis
ax1b = ax1.twinx()
ax1b.plot(K_vals, K_bw, 'r--', linewidth=1, alpha=0.7)
ax1b.set_ylabel('Bandwidth (GB/s)', fontsize=12, color='r')
ax1b.tick_params(axis='y', labelcolor='r')

# Linear fit
coeffs = np.polyfit(K_vals, K_times, 1)
K_fit = np.polyval(coeffs, K_vals)
ax1.plot(K_vals, K_fit, 'g:', linewidth=2, label=f'Linear fit (R²={data["analysis"]["K_sweep_r2"]:.2f})')
ax1.legend(loc='upper left')

# N sweep
ax2 = axes[1]
n_data = data['N_sweep']
N_vals = [d['N'] for d in n_data]
N_times = [d['mean_ms'] * 1000 for d in n_data]
N_bw = [d['bandwidth_gbs'] for d in n_data]

ax2.plot(N_vals, N_times, 'b.-', linewidth=1.5, markersize=4)
ax2.set_xlabel('N Dimension', fontsize=12)
ax2.set_ylabel('Latency (μs)', fontsize=12, color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.set_title('GEMV N Sweep (K=4096 fixed)', fontsize=13)
ax2.grid(True, alpha=0.3)

# Add bandwidth on secondary axis
ax2b = ax2.twinx()
ax2b.plot(N_vals, N_bw, 'r--', linewidth=1, alpha=0.7)
ax2b.set_ylabel('Bandwidth (GB/s)', fontsize=12, color='r')
ax2b.tick_params(axis='y', labelcolor='r')

# Linear fit
coeffs_n = np.polyfit(N_vals, N_times, 1)
N_fit = np.polyval(coeffs_n, N_vals)
ax2.plot(N_vals, N_fit, 'g:', linewidth=2, label=f'Linear fit (R²={data["analysis"]["N_sweep_r2"]:.2f})')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('Latex/figures/fig_gemv_sweep.png', dpi=150, bbox_inches='tight')
plt.savefig('Latex/figures/fig_gemv_sweep.pdf', bbox_inches='tight')
print("Saved to Latex/figures/fig_gemv_sweep.png and .pdf")

# Also create a combined latency-only plot
fig2, ax = plt.subplots(figsize=(8, 5))

ax.plot(K_vals, K_times, 'b.-', linewidth=1.5, markersize=4, label='K sweep (N=4096)')
ax.plot(N_vals, N_times, 'r.-', linewidth=1.5, markersize=4, label='N sweep (K=4096)')

ax.set_xlabel('Dimension Size', fontsize=12)
ax.set_ylabel('Latency (μs)', fontsize=12)
ax.set_title('GEMV Latency vs Dimension (A100)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('Latex/figures/fig_gemv_latency.png', dpi=150, bbox_inches='tight')
plt.savefig('Latex/figures/fig_gemv_latency.pdf', bbox_inches='tight')
print("Saved to Latex/figures/fig_gemv_latency.png and .pdf")
