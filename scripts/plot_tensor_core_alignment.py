#!/usr/bin/env python3
"""
Plot Tensor Core MMA alignment results.
Style matching the L2 alignment plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Load data
with open('results/tensor_core_alignment_sweep.json', 'r') as f:
    raw_data = json.load(f)

# K sweep data
k_data = [(d['K'], d['tflops']) for d in raw_data if d['sweep'] == 'K_sweep']
k_data = sorted(k_data, key=lambda x: x[0])

K_vals = np.array([d[0] for d in k_data])
tflops_vals = np.array([d[1] for d in k_data])
aligned = np.array([k % 16 == 0 for k in K_vals])

# Create aligned baseline
aligned_baseline = np.interp(K_vals, K_vals[aligned], tflops_vals[aligned])

# Create figure
fig, ax = plt.subplots(figsize=(7, 3.5))

# Fill area between aligned baseline and actual (penalty region)
ax.fill_between(K_vals, tflops_vals, aligned_baseline,
                alpha=0.3, color='#D33F49', label='Alignment penalty')

# Plot aligned baseline (green line)
ax.plot(K_vals, aligned_baseline, '-', color='#389E5C', linewidth=1.5,
        label=r'Aligned ($K$ mod $16 = 0$)')

# Plot actual TFLOPS (red line)
ax.plot(K_vals, tflops_vals, '-', color='#E07070', linewidth=1.5,
        label='Misaligned')

# Labels
ax.set_xlabel(r'$K$ Dimension', fontsize=11)
ax.set_ylabel('TFLOPS', fontsize=11)

# Y limits
y_min = min(tflops_vals) * 0.9
y_max = max(aligned_baseline) * 1.1
ax.set_ylim(y_min, y_max)
ax.set_xlim(K_vals.min(), K_vals.max())

# X ticks
xticks = [4000, 4016, 4032, 4048, 4064, 4080, 4096]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks], fontsize=9)

# Legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Grid
ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('Latex/figures/fig_tc_k_alignment.pdf', bbox_inches='tight', dpi=150)
plt.savefig('Latex/figures/fig_tc_k_alignment.png', bbox_inches='tight', dpi=150)
print("Saved K sweep to Latex/figures/fig_tc_k_alignment.pdf")

# N sweep plot
n_data = [(d['N'], d['tflops']) for d in raw_data if d['sweep'] == 'N_sweep']
n_data = sorted(n_data, key=lambda x: x[0])

N_vals = np.array([d[0] for d in n_data])
tflops_n = np.array([d[1] for d in n_data])
aligned_n = np.array([n % 8 == 0 for n in N_vals])

# Create aligned baseline for N
aligned_baseline_n = np.interp(N_vals, N_vals[aligned_n], tflops_n[aligned_n])

# Create figure for N sweep
fig2, ax2 = plt.subplots(figsize=(7, 3.5))

ax2.fill_between(N_vals, tflops_n, aligned_baseline_n,
                 alpha=0.3, color='#D33F49', label='Alignment penalty')
ax2.plot(N_vals, aligned_baseline_n, '-', color='#389E5C', linewidth=1.5,
         label=r'Aligned ($N$ mod $8 = 0$)')
ax2.plot(N_vals, tflops_n, '-', color='#E07070', linewidth=1.5,
         label='Misaligned')

ax2.set_xlabel(r'$N$ Dimension', fontsize=11)
ax2.set_ylabel('TFLOPS', fontsize=11)
ax2.set_ylim(min(tflops_n) * 0.9, max(aligned_baseline_n) * 1.1)
ax2.set_xlim(N_vals.min(), N_vals.max())

xticks_n = [4000, 4016, 4032, 4048, 4064, 4080, 4096]
ax2.set_xticks(xticks_n)
ax2.set_xticklabels([str(x) for x in xticks_n], fontsize=9)

ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig('Latex/figures/fig_tc_n_alignment.pdf', bbox_inches='tight', dpi=150)
plt.savefig('Latex/figures/fig_tc_n_alignment.png', bbox_inches='tight', dpi=150)
print("Saved N sweep to Latex/figures/fig_tc_n_alignment.pdf")
