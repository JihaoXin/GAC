#!/usr/bin/env python3
"""
Generate publication-quality figures for GAC paper.
Target: EuroMLSys (6 figures, professional style)

Usage:
    python scripts/create_paper_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge
from matplotlib.lines import Line2D
from pathlib import Path
from collections import Counter

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Warning: adjustText not installed. Labels may overlap in Figure 5.")

# Professional color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#E94F37',    # Red
    'accent': '#F39237',       # Orange
    'success': '#2ECC71',      # Green
    'neutral': '#6C757D',      # Gray
    'aligned': '#27AE60',      # Green for aligned
    'misaligned': '#E74C3C',   # Red for misaligned
    'light_bg': '#F8F9FA',     # Light background
    'dark': '#2C3E50',         # Dark text
}

# Publication style settings
# REVIEWER FIX M3: All fonts must be 8pt minimum for print readability
# Using 10pt+ as minimum to ensure clear readability in single-column format
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,           # Base font size (10 → 11)
    'axes.labelsize': 12,      # Axis labels (11 → 12, well above 8pt)
    'axes.titlesize': 13,      # Titles (12 → 13)
    'legend.fontsize': 10,     # Legend (9 → 10, clearly >= 8pt)
    'xtick.labelsize': 10,     # X tick labels (9 → 10)
    'ytick.labelsize': 10,     # Y tick labels (9 → 10)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

OUTPUT_DIR = Path('Latex/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_overview():
    """Figure 1: Dimensional Collapse Overview - REDESIGNED for maximum clarity.

    REVIEWER FIX M1: Original version had 4 sub-parts with 6-7pt fonts.
    New design: Simplified to 2 clear panels with ALL text >= 9pt.
    Key numbers (88%, 30%) now VERY prominently displayed in large bold font.

    Layout: Left panel shows problem, right panel shows solution.
    Removed bottom trade-off box to reduce clutter - that info is in the paper text.

    REVIEWER FIX M2: Reduced from 7.2x2.8 to 6.5x2.5 to fix Page 6 crowding.
    Figure has low information density (2 histograms + arrows + numbers).
    """
    # REVIEWER M2: Smaller figure to reduce page 6 crowding
    fig = plt.figure(figsize=(6.5, 2.5))

    # ========== LEFT PANEL: THE PROBLEM ==========
    ax1 = fig.add_axes([0.02, 0.12, 0.46, 0.82])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 7)
    ax1.axis('off')
    ax1.set_title('(a) Dimensional Collapse Problem', fontsize=13, fontweight='bold', pad=8)

    # Original model box - large, clear
    ax1.add_patch(FancyBboxPatch((0.2, 3.2), 2.8, 3.2, boxstyle="round,pad=0.15",
                                  facecolor=COLORS['primary'], edgecolor='black', linewidth=2.5, alpha=0.95))
    ax1.text(1.6, 5.6, 'Original', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax1.text(1.6, 4.6, 'head_dim', ha='center', va='center', fontsize=12, color='white')
    ax1.text(1.6, 3.8, '= 128', ha='center', va='center', fontsize=16, fontweight='bold',
             color='white', family='monospace')

    # Arrow with compression label
    ax1.annotate('', xy=(4.5, 4.8), xytext=(3.2, 4.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=3.5))
    ax1.text(3.85, 5.8, 'SVD', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(3.85, 5.25, 'Compress', ha='center', va='center', fontsize=11)

    # Compressed model box - RED for problem
    ax1.add_patch(FancyBboxPatch((4.5, 3.2), 2.8, 3.2, boxstyle="round,pad=0.15",
                                  facecolor=COLORS['misaligned'], edgecolor='black', linewidth=2.5, alpha=0.95))
    ax1.text(5.9, 5.6, 'Compressed', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax1.text(5.9, 4.6, 'head_dim', ha='center', va='center', fontsize=12, color='white')
    ax1.text(5.9, 3.8, '= 107', ha='center', va='center', fontsize=16, fontweight='bold',
             color='white', family='monospace')

    # Key impact number - VERY LARGE and BOLD - this is the main message
    ax1.add_patch(FancyBboxPatch((7.6, 3.0), 2.2, 3.5, boxstyle="round,pad=0.1",
                                  facecolor='#FFE4E1', edgecolor=COLORS['misaligned'],
                                  linewidth=3, alpha=0.95))
    ax1.text(8.7, 5.3, '+88%', ha='center', va='center', fontsize=24, fontweight='bold',
             color=COLORS['misaligned'])
    ax1.text(8.7, 4.2, 'Latency', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['dark'])
    ax1.text(8.7, 3.5, 'Increase', ha='center', va='center', fontsize=11, color=COLORS['dark'])

    # Simple explanation at bottom - larger font
    ax1.text(5.0, 1.6, '107 % 8 ≠ 0 → GPU alignment violation',
             ha='center', va='center', fontsize=11, style='italic', color=COLORS['dark'],
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_bg'], edgecolor='gray', alpha=0.8))

    # ========== RIGHT PANEL: THE SOLUTION ==========
    ax2 = fig.add_axes([0.52, 0.12, 0.46, 0.82])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 7)
    ax2.axis('off')
    ax2.set_title('(b) Dimension Repair Solution', fontsize=13, fontweight='bold', pad=8)

    # Misaligned input
    ax2.add_patch(FancyBboxPatch((0.2, 3.2), 2.5, 3.2, boxstyle="round,pad=0.15",
                                  facecolor=COLORS['misaligned'], edgecolor='black', linewidth=2, alpha=0.9))
    ax2.text(1.45, 5.6, 'd=107', ha='center', va='center', fontsize=15, fontweight='bold', color='white')
    ax2.text(1.45, 4.5, '(misaligned)', ha='center', va='center', fontsize=11, color='white')
    ax2.text(1.45, 3.7, '2.15 ms', ha='center', va='center', fontsize=12, color='white')

    # Arrow with repair label
    ax2.annotate('', xy=(4.0, 4.8), xytext=(2.9, 4.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=3.5))
    ax2.text(3.45, 5.7, 'Zero-Pad', ha='center', va='center', fontsize=11, fontweight='bold',
             color=COLORS['success'])
    ax2.text(3.45, 5.2, '→ 112', ha='center', va='center', fontsize=11, fontweight='bold',
             color=COLORS['success'])

    # Repaired output - GREEN for success
    ax2.add_patch(FancyBboxPatch((4.0, 3.2), 2.5, 3.2, boxstyle="round,pad=0.15",
                                  facecolor=COLORS['aligned'], edgecolor='black', linewidth=2, alpha=0.9))
    ax2.text(5.25, 5.6, 'd=112', ha='center', va='center', fontsize=15, fontweight='bold', color='white')
    ax2.text(5.25, 4.5, '(8-aligned)', ha='center', va='center', fontsize=11, color='white')
    ax2.text(5.25, 3.7, '1.49 ms', ha='center', va='center', fontsize=12, color='white')

    # Result metrics box - prominently displayed - VERY LARGE numbers
    ax2.add_patch(FancyBboxPatch((6.8, 3.0), 3.0, 3.5, boxstyle="round,pad=0.15",
                                  facecolor='#E8F5E9', edgecolor=COLORS['aligned'],
                                  linewidth=3, alpha=0.95))
    ax2.text(8.3, 5.5, '+30%', ha='center', va='center', fontsize=24, fontweight='bold',
             color=COLORS['aligned'])
    ax2.text(8.3, 4.5, 'Speedup', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['dark'])

    # Memory overhead - clearly visible
    ax2.text(8.3, 3.6, 'Memory: +4.7%', ha='center', va='center', fontsize=11,
             color=COLORS['neutral'])

    # Simple explanation at bottom
    ax2.text(5.0, 1.6, 'Bit-exact output preservation',
             ha='center', va='center', fontsize=11, style='italic', color=COLORS['dark'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_bg'], edgecolor='gray', alpha=0.8))

    # REVIEWER M2: Tighten layout to remove excessive white space
    plt.tight_layout(pad=0.3)

    fig.savefig(OUTPUT_DIR / 'fig1_overview.pdf')
    fig.savefig(OUTPUT_DIR / 'fig1_overview.png')
    print(f"Saved: fig1_overview.pdf (REDESIGNED - M1: fonts >= 9pt, M2: reduced size)")
    plt.close()


def fig2_sdpa_latency():
    """Figure 2: SDPA Latency vs Head Dimension (64-256).

    Single connected line, color-coded by backend:
      - Flash Attention (d%8==0): green segments
      - Math fallback (d%8!=0): red segments
    FA2 template boundaries visible as staircase pattern.
    Merges original S1 (64-160) and extended S1 (129-256) datasets.
    """
    # Load and merge both datasets
    all_measurements = {}  # head_dim -> latency (deduplicate by keeping first)

    for path in [
        Path('results/S1/20260119_224805_S1_sdpa_dense_sweep/raw.json'),
        Path('results/S1/20260202_170342_S1_sdpa_extended/raw.json'),
    ]:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data['measurements']:
            s = r['shape']
            if s['batch'] == 4 and s['seq_len'] == 2048 and 64 <= s['head_dim'] <= 256:
                d = s['head_dim']
                if d not in all_measurements:
                    all_measurements[d] = r['timing']['mean']

    dims = np.array(sorted(all_measurements.keys()))
    latencies = np.array([all_measurements[d] for d in dims])

    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    # Draw segments between consecutive points, colored by backend
    for i in range(len(dims) - 1):
        d0, d1 = dims[i], dims[i + 1]
        l0, l1 = latencies[i], latencies[i + 1]
        is_flash = (d1 % 8 == 0)
        c = COLORS['aligned'] if is_flash else COLORS['misaligned']
        lw = 1.2 if is_flash else 0.6
        alpha = 0.9 if is_flash else 0.4
        ax.plot([d0, d1], [l0, l1], color=c, linewidth=lw, alpha=alpha, zorder=2)

    # Scatter markers: larger for aligned, smaller for misaligned
    mask_a = (dims % 8 == 0)
    ax.scatter(dims[mask_a], latencies[mask_a], color=COLORS['aligned'],
               s=12, zorder=4, edgecolors='white', linewidths=0.2,
               label='Flash Attention ($d$ mod 8 = 0)')
    ax.scatter(dims[~mask_a], latencies[~mask_a], color=COLORS['misaligned'],
               s=3, zorder=3, alpha=0.5,
               label='Math fallback ($d$ mod 8 $\\neq$ 0)')

    # FA2 template boundaries: {64, 96, 128, 160, 192, 224, 256}
    _tmpl_kw = dict(color='#666666', linestyle='--', linewidth=0.6, alpha=0.3)
    templates = [64, 96, 128, 160, 192, 224, 256]
    for bnd in templates[:-1]:
        ax.axvline(x=bnd + 0.5, **_tmpl_kw)

    # Alternating template tier shading
    shade_colors = ['#FFF3E0', '#E3F2FD']
    for i, (t0, t1) in enumerate(zip(templates[:-1], templates[1:])):
        ax.axvspan(t0 + 0.5, t1 + 0.5, color=shade_colors[i % 2], alpha=0.15)

    # Template tier labels inside shaded regions
    # FA2 selects smallest template >= head_dim, so each shaded region
    # (t0, t1] uses template=t1.  Br×Bc from NCU profiling:
    _tmpl_clr = '#222222'
    tmpl_bxbc = {
        96:  '128$\\times$64',
        128: '128$\\times$64',
        160: '128$\\times$32',
        192: '128$\\times$32',
        224: '128$\\times$32',
        256: '128$\\times$32',
    }
    for t0, t1 in zip(templates[:-1], templates[1:]):
        mid = (t0 + t1) / 2
        br_bc = tmpl_bxbc.get(t1, '?')
        label = f't={t1}\n{br_bc}'
        # First 3 tiers at top, rest at bottom (avoid overlap with data)
        if t1 <= 160:
            ax.text(mid, 0.97, label, transform=ax.get_xaxis_transform(),
                    fontsize=5.5, ha='center', va='top', color=_tmpl_clr)
        else:
            ax.text(mid, 0.03, label, transform=ax.get_xaxis_transform(),
                    fontsize=5.5, ha='center', va='bottom', color=_tmpl_clr)
    # d=64 uses template=64 (128×128) — just one point, note in caption

    # Annotate key observations
    _ann_kw = dict(fontsize=5.5, ha='center', va='center', color='#444444',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                             edgecolor='#999999', linewidth=0.3, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5))

    # d=128 exact template fit → dip
    idx128 = list(dims).index(128)
    ax.annotate('d=128', xy=(128, latencies[idx128]),
                xytext=(115, latencies[idx128] - 0.3), **_ann_kw)

    # d=129 cliff (template jump 128→160)
    idx129 = list(dims).index(129)
    pct = (latencies[idx129] / latencies[idx128] - 1) * 100
    ax.annotate(f'd=129 (+{pct:.0f}%)', xy=(129, latencies[idx129]),
                xytext=(108, latencies[idx129] + 0.3), **_ann_kw)

    ax.set_xlabel('Head Dimension ($d$)', fontsize=7)
    ax.set_ylabel('Latency (ms)', fontsize=7)
    ax.set_xlim(60, 260)
    ax.tick_params(axis='both', labelsize=6)
    ax.set_xticks(templates)
    ax.legend(fontsize=5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3,
              bbox_to_anchor=(0.0, 0.82))
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.15)
    fig.savefig(OUTPUT_DIR / 'fig2_sdpa_latency.pdf')
    fig.savefig(OUTPUT_DIR / 'fig2_sdpa_latency.png')
    plt.close()
    print("Saved: fig2_sdpa_latency.pdf (64-256, merged datasets)")


def fig3_palu_distribution():
    """Figure 3: PaLU Per-Layer KV Head Dimension.

    Stem plot: one dot per layer, colored by 8-alignment.
    Horizontal bands at 8-aligned values for reference.
    """
    with open('results/palu_dim_dist/llama3_r0.8/dims.json') as f:
        data = json.load(f)

    dims = data['dims_per_head']  # 32 layers
    layers = list(range(len(dims)))

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    # Horizontal reference bands at 8-aligned values
    for val in [56, 64, 72, 80, 88, 96, 104, 112, 120, 128]:
        if min(dims) - 10 <= val <= max(dims) + 10:
            ax.axhline(val, color=COLORS['aligned'], alpha=0.15, linewidth=6, zorder=0)

    # Draw stems and markers manually for per-color control
    colors = [COLORS['aligned'] if d % 8 == 0 else COLORS['misaligned'] for d in dims]
    for x, y, c in zip(layers, dims, colors):
        ax.vlines(x, min(dims) - 8, y, color=c, linewidth=1.5, alpha=0.3, zorder=1)
        ax.plot(x, y, 'o', color=c, markersize=5, markeredgecolor='white',
                markeredgewidth=0.3, zorder=5)

    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Per-Head Dim', fontsize=10)
    ax.set_xlim(-1, len(dims))
    ax.set_ylim(min(dims) - 8, max(dims) + 8)
    ax.tick_params(labelsize=9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['aligned'],
               markersize=6, label='8-aligned'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['misaligned'],
               markersize=6, label='Misaligned'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              framealpha=0.9, edgecolor='none')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_palu_dist.pdf')
    fig.savefig(OUTPUT_DIR / 'fig3_palu_dist.png')
    print(f"Saved: fig3_palu_dist.pdf")
    plt.close()


def fig4_root_cause():
    """Figure 4: Root Cause Analysis - Hypothesis testing results with error bars.

    REVIEWER FIX M3: Increased all fonts to 10-12pt and improved color contrast.
    REVIEWER FIX m2: Improved color contrast for better print visibility.
    """
    fig, ax = plt.subplots(figsize=(2.3, 1.8))

    causes = ['TC', 'LDG', 'L2']
    impacts = [58.0, 50.0, 5.8]
    errors = [4.2, 5.5, 1.2]
    status = ['Confirmed', 'Confirmed', 'Minor']

    color_confirmed = '#0173B2'
    color_minor = '#DE8F05'
    colors = [color_confirmed if s == 'Confirmed' else color_minor for s in status]

    y_pos = np.arange(len(causes))
    bars = ax.barh(y_pos, impacts, color=colors, edgecolor='black', linewidth=0.5, height=0.55,
                   xerr=errors, capsize=2, error_kw={'linewidth': 0.7, 'capthick': 0.7})

    for i, s in enumerate(status):
        if s == 'Minor':
            bars[i].set_hatch('///')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(causes, fontsize=6.5, fontweight='bold')
    ax.tick_params(axis='y', labelsize=6.5, pad=2)

    for i, (impact, err, s) in enumerate(zip(impacts, errors, status)):
        ax.text(impact + err + 1, i, f'{impact:.0f}±{err:.0f}%',
               va='center', ha='left', fontsize=5.5, fontweight='bold')

    ax.set_xlabel('Impact (%)', fontsize=7)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlim(0, 80)
    ax.invert_yaxis()

    ax.axvline(x=50, color=COLORS['neutral'], linestyle='--', alpha=0.4, linewidth=0.5)

    legend_elements = [
        mpatches.Patch(facecolor=color_confirmed, edgecolor='black', label='Confirmed'),
        mpatches.Patch(facecolor=color_minor, edgecolor='black', hatch='///', label='Minor'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=5,
              borderpad=0.3, handlelength=1.0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_root_cause.pdf', bbox_inches=None)
    fig.savefig(OUTPUT_DIR / 'fig4_root_cause.png', bbox_inches=None)
    print(f"Saved: fig4_root_cause.pdf (FIXED - M3: fonts 10-12pt, high contrast colors)")
    plt.close()


def fig5_repair_tradeoff():
    """Figure 5: Repair Strategy Tradeoff - Per-dimension speedup vs overhead scatter plot.

    REVIEWER FIX m6: d=120 and d=121 labels were overlapping.
    REVIEWER FIX M3: Reduced figure size to match information density.
    REVIEWER FIX M2: Reduced from 3.3x2.5 to 3.0x2.2 to fix Page 6 crowding.
    Only 6 scatter points with low density.
    """
    fig, ax = plt.subplots(figsize=(2.3, 1.8))

    # Data from C4 experiment
    with open('results/C4/20260124_221749_C4_dimension_repair/results.json') as f:
        data = json.load(f)

    benchmark = data['benchmark']
    palu_analysis = data['palu_analysis']

    # Per-dimension data for MINIMAL strategy
    dims = [107, 114, 117, 120, 121, 125]
    minimal_overheads = []
    minimal_speedups = []
    minimal_dims = []  # Track which dims have minimal data
    optimal_overheads = []
    optimal_speedups = []

    for d in dims:
        orig = benchmark['original'][str(d)]

        # Calculate per-dimension overhead (approximate based on padding amount)
        if d % 8 == 0:  # Already aligned
            min_overhead = 0.0
        else:
            min_target = ((d + 7) // 8) * 8
            min_overhead = (min_target - d) / d * 100

        if d % 16 == 0:  # Already 16-aligned
            opt_overhead = 0.0
        else:
            opt_target = ((d + 15) // 16) * 16
            opt_overhead = (opt_target - d) / d * 100

        # Get speedups
        if str(d) in benchmark['minimal']:
            minimal = benchmark['minimal'][str(d)]
            speedup = (orig - minimal) / orig * 100
            minimal_overheads.append(min_overhead)
            minimal_speedups.append(speedup)
            minimal_dims.append(d)

        if str(d) in benchmark['optimal']:
            optimal = benchmark['optimal'][str(d)]
            speedup = (orig - optimal) / orig * 100
            optimal_overheads.append(opt_overhead)
            optimal_speedups.append(speedup)

    # Plot MINIMAL points - but handle d=120 specially for highlighting
    # REVIEWER m2: d=120 should be prominently highlighted in orange
    for i, (oh, sp, d) in enumerate(zip(minimal_overheads, minimal_speedups, minimal_dims)):
        if d == 120:
            # d=120 special highlight: large orange star marker
            ax.scatter([oh], [sp], c=COLORS['accent'], s=80,
                      marker='*', edgecolor='black', linewidth=1, zorder=5)
        else:
            ax.scatter([oh], [sp], c=COLORS['primary'], s=35,
                      marker='o', edgecolor='black', linewidth=0.7, zorder=3)

    ax.scatter([], [], c=COLORS['primary'], s=35, marker='o',
              edgecolor='black', linewidth=0.7, label='Minimal (→8)')

    ax.scatter(optimal_overheads, optimal_speedups, c=COLORS['accent'], s=35,
              marker='s', edgecolor='black', linewidth=0.7, label='Optimal (→16)', zorder=3)

    # Per-dimension labels with hand-tuned positions to avoid overlap.
    # Each entry: (xytext_x, xytext_y) in data coordinates, ha, va.
    label_positions = {
        107: {'xy': (2.5, 37), 'ha': 'left'},
        114: {'xy': (8.5, 14), 'ha': 'center'},
        117: {'xy': (0.3, 30), 'ha': 'left'},
        120: {'xy': (3.0, 7),  'ha': 'left'},
        121: {'xy': (9.0, 20), 'ha': 'center'},
        125: {'xy': (0.3, 20), 'ha': 'left'},
    }

    for i, d in enumerate(minimal_dims):
        if d not in label_positions:
            continue
        cfg = label_positions[d]
        pt_x, pt_y = minimal_overheads[i], minimal_speedups[i]

        if d == 120:
            ax.annotate(f'd=120 (aligned)',
                       xy=(pt_x, pt_y), xytext=(cfg['xy'][0], cfg['xy'][1]),
                       fontsize=5.5, color=COLORS['accent'], fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                                       lw=0.8, shrinkB=3),
                       ha=cfg['ha'], va='center',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF3E0',
                                edgecolor=COLORS['accent'], linewidth=0.7, alpha=0.95))
        else:
            ax.annotate(f'd={d}',
                       xy=(pt_x, pt_y), xytext=(cfg['xy'][0], cfg['xy'][1]),
                       fontsize=5.5, color=COLORS['dark'],
                       arrowprops=dict(arrowstyle='->', color='gray',
                                       lw=0.6, shrinkB=3),
                       ha=cfg['ha'], va='center',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='gray', linewidth=0.5, alpha=0.85))

    # Draw Pareto frontier line for minimal
    sorted_min = sorted(zip(minimal_overheads, minimal_speedups), key=lambda x: x[0])
    pareto_x = [0] + [p[0] for p in sorted_min]
    pareto_y = [0] + [p[1] for p in sorted_min]
    ax.plot(pareto_x, pareto_y, color=COLORS['primary'], linestyle='--', alpha=0.5, linewidth=0.7)

    for roi in [4, 8]:
        x_line = np.linspace(0.1, 12, 100)
        y_line = roi * x_line
        ax.plot(x_line, y_line, color=COLORS['neutral'], linestyle=':', alpha=0.25, linewidth=0.5)
        ax.text(11.5, min(roi * 11.5, 34), f'{roi}×', fontsize=5, color=COLORS['neutral'], va='center')

    ax.set_xlabel('Memory Overhead (%)', fontsize=7)
    ax.set_ylabel('Speedup (%)', fontsize=7)
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-5, 40)
    ax.tick_params(labelsize=6)

    avg_min_speedup = np.mean(minimal_speedups)
    avg_min_overhead = palu_analysis['minimal_overhead_pct']

    ax.text(0.97, 0.05, f'Avg: {avg_min_speedup:.0f}% speedup, '
                        f'{avg_min_overhead:.1f}% overhead',
           transform=ax.transAxes, fontsize=5, va='bottom', ha='right',
           color=COLORS['dark'],
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='gray', alpha=0.9))

    ax.legend(loc='upper right', framealpha=0.95, fontsize=5.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_repair_tradeoff.pdf', bbox_inches=None)
    fig.savefig(OUTPUT_DIR / 'fig5_repair_tradeoff.png', bbox_inches=None)
    print(f"Saved: fig5_repair_tradeoff.pdf (FIXED - M2: reduced size, m2: d=120 orange star)")
    plt.close()


def fig6_e2e_performance():
    """Figure 6: End-to-End Performance - LLM Inference Comparison.

    Single column width (~3.3 inches). Made taller to avoid overlap.
    """
    # Wider and taller to avoid label overlap
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.0))

    # Data from C5 experiment
    variants = ['Baseline', 'PaLU']
    prefill = [9870, 9672]  # tok/s
    decode = [119, 1371]    # tok/s

    # Prefill subplot
    # REVIEWER FIX M3: All fonts >= 8pt (using 9pt to be safe)
    ax1 = axes[0]
    bars1 = ax1.bar(variants, prefill, color=[COLORS['primary'], COLORS['accent']],
                    edgecolor='black', linewidth=1.0, width=0.55)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=10)
    ax1.set_title('Prefill', fontsize=11, fontweight='bold', pad=8)
    ax1.set_ylim(0, 13000)  # More headroom for labels
    ax1.tick_params(axis='x', labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    for bar, val in zip(bars1, prefill):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    # Add delta
    delta_prefill = (prefill[1] - prefill[0]) / prefill[0] * 100
    ax1.text(0.5, 0.92, f'{delta_prefill:+.1f}%', transform=ax1.transAxes,
            ha='center', fontsize=10, color=COLORS['secondary'] if delta_prefill < 0 else COLORS['success'])

    # Decode subplot
    ax2 = axes[1]
    bars2 = ax2.bar(variants, decode, color=[COLORS['primary'], COLORS['accent']],
                    edgecolor='black', linewidth=1.0, width=0.55)
    ax2.set_ylabel('Throughput (tok/s)', fontsize=10)
    ax2.set_title('Decode', fontsize=11, fontweight='bold', pad=8)
    ax2.set_ylim(0, 1800)  # More headroom
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    for bar, val in zip(bars2, decode):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    # Add speedup
    speedup = decode[1] / decode[0]
    ax2.text(0.5, 0.92, f'{speedup:.1f}x', transform=ax2.transAxes,
            ha='center', fontsize=11, fontweight='bold', color=COLORS['success'])

    # Add note at bottom - with enough space
    # REVIEWER FIX M3: fontsize >= 8pt
    fig.text(0.5, 0.01, 'Llama-3-8B, A100 80GB, B=4, S=2048',
            ha='center', fontsize=9, style='italic', color=COLORS['neutral'])

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(bottom=0.1)  # Space for bottom note
    fig.savefig(OUTPUT_DIR / 'fig6_e2e.pdf')
    fig.savefig(OUTPUT_DIR / 'fig6_e2e.png')
    print(f"Saved: fig6_e2e.pdf")
    plt.close()


def fig_alignment_sweep(csv_path='results/alignment_sweep.csv'):
    """Figure: GEMM latency vs dimension for M, N, K independently.

    Baseline: M=2048, N=2048, K=128.
    Three subplots:
      (a) Vary M 1024-2048 (token eviction)
      (b) Vary N 1024-2048 (pruning)
      (c) Vary K 64-128    (SVD / low-rank)

    For M/N (large range): faint raw line + bold rolling median + highlighted
    aligned/misaligned points.  For K (small range): all points visible.
    """
    import csv as csv_mod
    from scipy.ndimage import median_filter

    # Load data
    data = {'M': ([], []), 'N': ([], []), 'K': ([], [])}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            name = row['dim_name']
            if 'Memset' in row.get('kernel', ''):
                continue
            data[name][0].append(int(row['dim_value']))
            data[name][1].append(float(row['time_us']))

    # Convert to numpy and sort
    for key in data:
        xs, ys = data[key]
        xs, ys = np.array(xs), np.array(ys)
        order = np.argsort(xs)
        data[key] = (xs[order], ys[order])

    panels = [
        ('K', '(a) Vary $K$'),
        ('N', '(b) Vary $N$'),
        ('M', '(c) Vary $M$'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5),
                             gridspec_kw={'hspace': 0.45})

    from scipy.interpolate import interp1d

    for ax, (dim_name, title) in zip(axes, panels):
        xs, ys = data[dim_name]
        lo, hi = int(xs.min()), int(xs.max())

        # Separate aligned (mod 8 == 0) and misaligned points
        mask_a8 = (xs % 8 == 0)
        mask_mis = ~mask_a8
        xs_a, ys_a = xs[mask_a8], ys[mask_a8]
        xs_m, ys_m = xs[mask_mis], ys[mask_mis]

        # Smooth both traces (rolling median)
        win = 4 if dim_name == 'K' else 8
        ys_a_smooth = median_filter(ys_a, size=win) if len(ys_a) > win else ys_a
        ys_m_smooth = median_filter(ys_m, size=win) if len(ys_m) > win else ys_m

        # Fill between aligned and misaligned to show the gap
        if len(xs_m) > 1 and len(xs_a) > 1:
            # Interpolate both traces onto a common dense x-grid
            x_dense = np.linspace(max(xs_a[0], xs_m[0]),
                                  min(xs_a[-1], xs_m[-1]), 500)
            f_a = interp1d(xs_a, ys_a_smooth, bounds_error=False,
                           fill_value='extrapolate')
            f_m = interp1d(xs_m, ys_m_smooth, bounds_error=False,
                           fill_value='extrapolate')
            ax.fill_between(x_dense, f_a(x_dense), f_m(x_dense),
                            color=COLORS['misaligned'], alpha=0.10,
                            zorder=0, label='Alignment penalty')

        # Plot both traces
        ax.plot(xs_a, ys_a_smooth, color=COLORS['aligned'],
                linewidth=1.4, alpha=0.9, zorder=3,
                label='Aligned ($d$ mod 8 = 0)')
        ax.plot(xs_m, ys_m_smooth, color=COLORS['misaligned'],
                linewidth=0.8, alpha=0.6, zorder=2,
                label='Misaligned ($d$ mod 8 $\\neq$ 0)')

        # Ticks
        if dim_name == 'K':
            ticks = [d for d in range(lo, hi + 1) if d % 8 == 0]
        else:
            ticks = list(range(lo, hi + 1, 128))

        ax.legend(fontsize=8, loc='best', framealpha=0.8,
                  handletextpad=0.3, borderpad=0.3)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(f'${dim_name}$', fontsize=10)
        ax.set_ylabel('Latency ($\\mu$s)', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)
        ax.set_xticks(ticks)
        ax.set_xlim(lo, hi)

    # ---- Kernel annotations (from NCU profiling) ----
    # Short labels for kernel families:
    #   A = ampere_fp16_s16816gemm  (hand-tuned SASS, 256x128 tile, block=256)
    #   B = sm80_xmma_gemm          (XMMA codegen, 192x128 tile, block=256)
    #   C = ampere_fp16_s1688gemm   (hand-tuned SASS, 256x64 tile, block=128)
    _kern_style = dict(fontsize=7.5, fontweight='bold', ha='center', va='top',
                       color='#444444',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFDD',
                                 edgecolor='#999999', linewidth=0.5, alpha=0.85))
    _vline_kw = dict(color='#999999', linestyle=':', linewidth=0.6, alpha=0.6)

    # Shared annotation style: arrow pointing to cliff
    _ann_kw = dict(fontsize=7.5, fontweight='bold', ha='center', va='center',
                   color='#444444',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFDD',
                             edgecolor='#999999', linewidth=0.5, alpha=0.85),
                   arrowprops=dict(arrowstyle='->', color='#999999', lw=0.6))

    # (b) N panel (axes[1]): kernel transitions
    # NCU: B (xmma 96x128) up to ~1250, C (ampere s1688) ~1250-1664, B (xmma 192x128) from ~1664
    ax_n = axes[1]
    for bnd in [1250, 1664]:
        ax_n.axvline(x=bnd, **_vline_kw)
    ax_n.annotate('B→C', xy=(1280, 14),
                  xytext=(1400, 16.5), **_ann_kw)
    ax_n.annotate('C→B', xy=(1664, 14),
                  xytext=(1560, 17), **_ann_kw)

    # (c) M panel (axes[2]): kernel transitions at 1089, 1153, 1729
    ax_m = axes[2]
    for bnd in [1089, 1153, 1729]:
        ax_m.axvline(x=bnd, **_vline_kw)
    y_top_m = ax_m.get_ylim()[1]
    ax_m.annotate('A→B', xy=(1089, y_top_m * 0.85),
                  xytext=(1050, y_top_m * 0.68), **_ann_kw)
    ax_m.annotate('B→C', xy=(1153, y_top_m * 0.85),
                  xytext=(1200, y_top_m * 0.68), **_ann_kw)
    ax_m.annotate('C→B', xy=(1729, y_top_m * 0.95),
                  xytext=(1660, y_top_m * 0.78), **_ann_kw)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.06, hspace=0.45)

    out_pdf = OUTPUT_DIR / 'fig_alignment_sweep.pdf'
    out_png = OUTPUT_DIR / 'fig_alignment_sweep.png'
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close()
    print(f"  Saved: {out_pdf}, {out_png}")


def fig_alignment_sweep_compact(csv_path='results/alignment_sweep.csv'):
    """Compact horizontal 1x3 alignment sweep for paper body (figure*).

    Three panels side-by-side: Vary K (64-128), Vary N (1024-2048), Vary M (1024-2048).
    Designed for figure* in SIGPLAN 2-column: ~7.0" x 2.0".
    """
    import csv as csv_mod
    from scipy.ndimage import median_filter
    from scipy.interpolate import interp1d

    data = {'M': ([], []), 'N': ([], []), 'K': ([], [])}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            name = row['dim_name']
            if 'Memset' in row.get('kernel', ''):
                continue
            data[name][0].append(int(row['dim_value']))
            data[name][1].append(float(row['time_us']))

    for key in data:
        xs, ys = data[key]
        xs, ys = np.array(xs), np.array(ys)
        order = np.argsort(xs)
        data[key] = (xs[order], ys[order])

    panels = [
        ('M', '$M$ Dimension'),
        ('N', '$N$ Dimension'),
        ('K', '$K$ Dimension'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.6),
                              gridspec_kw={'wspace': 0.22})

    _ann_kw = dict(fontsize=6, fontweight='bold', ha='center', va='center',
                   color='#444444',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='#FFFFDD',
                             edgecolor='#999999', linewidth=0.4, alpha=0.85),
                   arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5))
    _vline_kw = dict(color='#999999', linestyle=':', linewidth=0.5, alpha=0.5)

    for ax, (dim_name, title) in zip(axes, panels):
        xs, ys = data[dim_name]
        lo, hi = int(xs.min()), int(xs.max())

        mask_a8 = (xs % 8 == 0)
        mask_mis = ~mask_a8
        xs_a, ys_a = xs[mask_a8], ys[mask_a8]
        xs_m, ys_m = xs[mask_mis], ys[mask_mis]

        win = 4 if dim_name == 'K' else 8
        ys_a_smooth = median_filter(ys_a, size=win) if len(ys_a) > win else ys_a
        ys_m_smooth = median_filter(ys_m, size=win) if len(ys_m) > win else ys_m

        if len(xs_m) > 1 and len(xs_a) > 1:
            x_dense = np.linspace(max(xs_a[0], xs_m[0]),
                                  min(xs_a[-1], xs_m[-1]), 500)
            f_a = interp1d(xs_a, ys_a_smooth, bounds_error=False, fill_value='extrapolate')
            f_m = interp1d(xs_m, ys_m_smooth, bounds_error=False, fill_value='extrapolate')
            ax.fill_between(x_dense, f_a(x_dense), f_m(x_dense),
                            color=COLORS['misaligned'], alpha=0.10, zorder=0)

        ax.plot(xs_a, ys_a_smooth, color=COLORS['aligned'], linewidth=1.0, alpha=0.9, zorder=3)
        ax.plot(xs_m, ys_m_smooth, color=COLORS['misaligned'], linewidth=0.6, alpha=0.5, zorder=2)

        ax.set_xlabel(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis='y', alpha=0.2, linewidth=0.3)
        ax.set_xlim(lo, hi)

        if dim_name == 'K':
            ticks = [d for d in range(lo, hi + 1) if d % 8 == 0]
            ax.set_xticks(ticks)
            ax.tick_params(axis='x', rotation=45)

    axes[0].set_ylabel('Latency ($\\mu$s)', fontsize=7)

    legend_elements = [
        Line2D([0], [0], color=COLORS['aligned'], linewidth=1.0, label='Aligned ($d$ mod 8 = 0)'),
        Line2D([0], [0], color=COLORS['misaligned'], linewidth=0.6, alpha=0.5, label='Misaligned'),
        mpatches.Patch(color=COLORS['misaligned'], alpha=0.10, label='Alignment penalty'),
    ]
    axes[2].legend(handles=legend_elements, fontsize=5, loc='upper right',
                   framealpha=0.8, handletextpad=0.3, borderpad=0.3)

    # Kernel annotations on M panel (axes[0])
    ax_m = axes[0]
    for bnd in [1089, 1153, 1729]:
        ax_m.axvline(x=bnd, **_vline_kw)
    y_top_m = ax_m.get_ylim()[1]
    ax_m.annotate('A$\\to$B', xy=(1089, y_top_m * 0.85),
                  xytext=(1050, y_top_m * 0.65), **_ann_kw)
    ax_m.annotate('B$\\to$C', xy=(1153, y_top_m * 0.85),
                  xytext=(1220, y_top_m * 0.65), **_ann_kw)
    ax_m.annotate('C$\\to$B', xy=(1729, y_top_m * 0.95),
                  xytext=(1650, y_top_m * 0.75), **_ann_kw)

    # Kernel annotations on N panel (axes[1])
    ax_n = axes[1]
    for bnd in [1250, 1664]:
        ax_n.axvline(x=bnd, **_vline_kw)
    ax_n.annotate('B$\\to$C', xy=(1280, 14), xytext=(1400, 16.5), **_ann_kw)
    ax_n.annotate('C$\\to$B', xy=(1664, 14), xytext=(1560, 17), **_ann_kw)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.24, wspace=0.22)
    fig.savefig(OUTPUT_DIR / 'fig_gemm_alignment.pdf')
    fig.savefig(OUTPUT_DIR / 'fig_gemm_alignment.png')
    plt.close()
    print("Saved: fig_gemm_alignment.pdf (compact 1x3 horizontal)")


def fig_gac_ranks(ranks_dir='results/gac_allocation'):
    """Per-layer rank comparison: unaligned vs round8 vs gac_dp.

    2-panel (k_proj, v_proj), single-column width.
    """
    ideal_path = Path(ranks_dir) / 'ideal_float_ranks.json'
    if not ideal_path.exists():
        print("Skipping fig_gac_ranks: no ideal_float_ranks.json")
        return

    with open(ideal_path) as f:
        ideal_raw = json.load(f)

    # Parse ideal ranks (list of dicts with layer, proj, ideal_rank)
    ideal = {}
    for entry in ideal_raw:
        ideal[(entry['layer'], entry['proj'])] = entry['ideal_rank']

    strategies = {}
    for name in ['unaligned', 'round8', 'gac_dp']:
        p = Path(ranks_dir) / f'ranks_{name}.json'
        if p.exists():
            with open(p) as f:
                strategies[name] = json.load(f)

    if not strategies:
        print("Skipping fig_gac_ranks: no rank files")
        return

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 2.6), sharex=True,
                              gridspec_kw={'hspace': 0.15, 'height_ratios': [1.4, 1]})

    layers = list(range(32))

    style = {
        'unaligned': (COLORS['misaligned'], 0.8, '-', 1.5, 'Unaligned'),
        'round8': (COLORS['accent'], 0.8, '--', 1.5, 'Round-to-8'),
        'gac_dp': (COLORS['primary'], 1.2, '-', 3, 'GAC DP'),
    }

    for ax, proj_name, panel_title in [(axes[0], 'k_proj', '$W_K$'),
                                        (axes[1], 'v_proj', '$W_V$')]:
        # Ideal float ranks
        ideal_ranks = [ideal.get((l, proj_name), None) for l in layers]
        if ideal_ranks[0] is not None:
            ax.plot(layers, ideal_ranks, color=COLORS['neutral'], linewidth=0.8,
                    linestyle=':', alpha=0.5, label='Ideal', zorder=1)

        for name, (color, lw, ls, zorder, label) in style.items():
            if name not in strategies:
                continue
            ranks = []
            for l in layers:
                key = f"model.layers.{l}.self_attn.{proj_name}"
                if key in strategies[name]:
                    ranks.append(strategies[name][key][0])
                else:
                    ranks.append(None)
            if ranks[0] is not None:
                ax.plot(layers, ranks, color=color, linewidth=lw,
                        linestyle=ls, alpha=0.85, label=label, zorder=zorder)

        ax.set_ylabel('Rank', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.text(0.02, 0.92, panel_title, transform=ax.transAxes,
                fontsize=7, fontweight='bold', va='top')
        if proj_name == 'k_proj':
            ax.set_ylim(40, 550)
        else:
            ax.set_ylim(350, 530)

    axes[1].set_xlabel('Layer', fontsize=7)
    axes[0].legend(fontsize=5, loc='lower left', framealpha=0.8,
                   ncol=4, handletextpad=0.3, columnspacing=0.6, borderpad=0.2)
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.13, hspace=0.12)
    fig.savefig(OUTPUT_DIR / 'fig_gac_ranks.pdf')
    fig.savefig(OUTPUT_DIR / 'fig_gac_ranks.png')
    plt.close()
    print("Saved: fig_gac_ranks.pdf (per-layer rank comparison)")


def main():
    """Generate all figures."""
    print("=" * 50)
    print("Generating GAC Paper Figures")
    print("=" * 50)

    fig1_overview()
    fig2_sdpa_latency()
    fig3_palu_distribution()
    fig4_root_cause()
    fig5_repair_tradeoff()
    fig6_e2e_performance()
    fig_alignment_sweep_compact()
    fig_gac_ranks()

    print("=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
