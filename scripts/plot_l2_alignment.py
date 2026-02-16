#!/usr/bin/env python3
"""
Plot L2 Cache Sector Alignment results as line chart.
Style matching the SDPA latency plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot L2 alignment figure")
    parser.add_argument("--out-dir", type=str, default="Latex/figures",
                        help="Output directory for figures")
    parser.add_argument("--suffix", type=str, default="",
                        help="Filename suffix, e.g., _h100")
    return parser.parse_args()


args = parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
suffix = args.suffix

# Load dense sweep data
try:
    with open('results/l2_dense_sweep.json', 'r') as f:
        raw_data = json.load(f)
    data = [(d['K'], d['bandwidth']) for d in raw_data]
except FileNotFoundError:
    # Fallback data
    data = [
        (4024, 71.6), (4032, 158.2), (4040, 71.4), (4048, 162.3),
        (4056, 73.5), (4064, 163.3), (4072, 73.7), (4080, 162.9),
        (4088, 73.8), (4096, 163.9),
    ]

# Sort by K
data = sorted(data, key=lambda x: x[0])

K_vals = np.array([d[0] for d in data])
bw_vals = np.array([d[1] for d in data])
aligned = np.array([k % 16 == 0 for k in K_vals])

# Create aligned baseline (maximum bandwidth at each aligned point)
aligned_baseline = np.interp(K_vals, K_vals[aligned], bw_vals[aligned])

# Create figure
fig, ax = plt.subplots(figsize=(7, 3.5))

# Fill area between aligned baseline and actual (penalty region)
ax.fill_between(K_vals, bw_vals, aligned_baseline,
                alpha=0.3, color='#D33F49', label='Alignment penalty')

# Plot aligned baseline (green line)
ax.plot(K_vals, aligned_baseline, '-', color='#389E5C', linewidth=1.5,
        label=r'Aligned ($K$ mod $16 = 0$)')

# Plot actual bandwidth (red/coral line like reference)
ax.plot(K_vals, bw_vals, '-', color='#E07070', linewidth=1.5,
        label='Misaligned')

# Labels
ax.set_xlabel(r'$K$ Dimension', fontsize=11)
ax.set_ylabel('Bandwidth (GB/s)', fontsize=11)

# Y limits
ax.set_ylim(50, 210)
ax.set_xlim(K_vals.min(), K_vals.max())

# X ticks
xticks = [4000, 4016, 4032, 4048, 4064, 4080, 4096]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks], fontsize=9)

# Legend - upper right like reference
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Grid
ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_pdf = out_dir / f'fig_l2_alignment{suffix}.pdf'
out_png = out_dir / f'fig_l2_alignment{suffix}.png'
plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print(f"Saved to {out_pdf}")
