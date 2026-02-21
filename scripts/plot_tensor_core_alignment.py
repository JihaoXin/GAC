#!/usr/bin/env python3
"""
Plot Tensor Core MMA alignment results.
Style matching the L2 alignment plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path


def parse_args():
        parser = argparse.ArgumentParser(description="Plot Tensor Core alignment figures")
        parser.add_argument("--out-dir", type=str, default="Latex/figures",
                                                help="Output directory for figures")
        parser.add_argument("--suffix", type=str, default="",
                                                help="Filename suffix, e.g., _h100")
        parser.add_argument("--make-hw-combined", action="store_true",
                                                help="Also generate fig_hw_alignment (K-TC, N-TC, L2)")
        return parser.parse_args()


args = parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
suffix = args.suffix

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
out_k_pdf = out_dir / f'fig_tc_k_alignment{suffix}.pdf'
out_k_png = out_dir / f'fig_tc_k_alignment{suffix}.png'
plt.savefig(out_k_pdf, bbox_inches='tight', dpi=150)
plt.savefig(out_k_png, bbox_inches='tight', dpi=150)
print(f"Saved K sweep to {out_k_pdf}")

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
out_n_pdf = out_dir / f'fig_tc_n_alignment{suffix}.pdf'
out_n_png = out_dir / f'fig_tc_n_alignment{suffix}.png'
plt.savefig(out_n_pdf, bbox_inches='tight', dpi=150)
plt.savefig(out_n_png, bbox_inches='tight', dpi=150)
print(f"Saved N sweep to {out_n_pdf}")


if args.make_hw_combined:
        with open('results/l2_dense_sweep.json', 'r') as f:
                l2_raw = json.load(f)

        l2_data = sorted([(d['K'], d['bandwidth']) for d in l2_raw], key=lambda x: x[0])
        l2_k = np.array([d[0] for d in l2_data])
        l2_bw = np.array([d[1] for d in l2_data])
        l2_aligned = np.array([k % 16 == 0 for k in l2_k])
        l2_baseline = np.interp(l2_k, l2_k[l2_aligned], l2_bw[l2_aligned])

        fig3, axes = plt.subplots(1, 3, figsize=(10.5, 2.3))

        # (a) Tensor Core K sweep
        ax = axes[0]
        ax.fill_between(K_vals, tflops_vals, aligned_baseline, alpha=0.3, color='#D33F49', label='Alignment penalty')
        ax.plot(K_vals, aligned_baseline, '-', color='#389E5C', linewidth=1.5, label=r'Aligned ($K$ mod $16 = 0$)')
        ax.plot(K_vals, tflops_vals, '-', color='#E07070', linewidth=1.5, label='Misaligned')
        ax.set_xlabel(r'$K$ Dimension', fontsize=9)
        ax.set_ylabel('TFLOPS', fontsize=9)
        ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
        ax.set_xticks([4000, 4032, 4064, 4096])
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, framealpha=0.9, loc='lower right')
        ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # (b) Tensor Core N sweep
        ax = axes[1]
        ax.fill_between(N_vals, tflops_n, aligned_baseline_n, alpha=0.3, color='#D33F49', label='Alignment penalty')
        ax.plot(N_vals, aligned_baseline_n, '-', color='#389E5C', linewidth=1.5, label=r'Aligned ($N$ mod $8 = 0$)')
        ax.plot(N_vals, tflops_n, '-', color='#E07070', linewidth=1.5, label='Misaligned')
        ax.set_xlabel(r'$N$ Dimension', fontsize=9)
        ax.set_ylabel('TFLOPS', fontsize=9)
        ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
        ax.set_xticks([4000, 4032, 4064, 4096])
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, framealpha=0.9, loc='lower right')
        ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # (c) L2 sector sweep
        ax = axes[2]
        ax.fill_between(l2_k, l2_bw, l2_baseline, alpha=0.3, color='#D33F49', label='Alignment penalty')
        ax.plot(l2_k, l2_baseline, '-', color='#389E5C', linewidth=1.5, label=r'Aligned ($K$ mod $16 = 0$)')
        ax.plot(l2_k, l2_bw, '-', color='#E07070', linewidth=1.5, label='Misaligned')
        ax.set_xlabel(r'$K$ Dimension', fontsize=9)
        ax.set_ylabel('GB/s', fontsize=9)
        ax.text(0.02, 0.95, '(c)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
        ax.set_xticks([4000, 4032, 4064, 4096])
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, framealpha=0.9, loc='lower right')
        ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        fig3.tight_layout()
        out_hw_pdf = out_dir / f'fig_hw_alignment{suffix}.pdf'
        out_hw_png = out_dir / f'fig_hw_alignment{suffix}.png'
        fig3.savefig(out_hw_pdf, bbox_inches='tight', dpi=150)
        fig3.savefig(out_hw_png, bbox_inches='tight', dpi=150)
        plt.close(fig3)
        print(f"Saved combined hardware figure to {out_hw_pdf}")
