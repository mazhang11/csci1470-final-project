"""
visualize_results.py

Reads all saved results.txt files and produces figures:
  1. Bar chart — ADHD-200 vs paper targets
  2. Per-repeat line plot — ADHD-200
  3. Per-repeat line plot — ABIDE
  4. Side-by-side bar — ADHD-200 vs ABIDE
  5. Box plots — accuracy distribution

Usage:
  python scripts/visualize_results.py
"""

import re
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # no display needed — saves to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent
ADHD_DIR  = ROOT / 'outputs_oscar'
ABIDE_DIR = ROOT / 'outputs_abide'
FIG_DIR   = ROOT / 'outputs_oscar' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPER_TARGETS = {
    'falff':          62.06,
    'reho':           60.27,
    'gm':             65.43,
    'falff_gm_multi': 69.15,
}
COLORS = {
    'falff':          'C0',
    'reho':           'C3',
    'gm':             'C1',
    'falff_gm_multi': 'C2',
}
LABELS_PRETTY = {
    'falff':          'fALFF',
    'reho':           'ReHo',
    'gm':             'GM',
    'falff_gm_multi': 'fALFF+GM',
}


def parse_float_list(s):
    """Extract all floats from a string like [np.float64(0.8), np.float64(0.7), ...]."""
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]


def load_results(base_dir, experiments):
    results = {}
    for label in experiments:
        path = Path(base_dir) / label / 'results.txt'
        if not path.exists():
            continue
        d = {'label': label}
        with open(path) as f:
            for line in f:
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                key, val = key.strip(), val.strip()
                if key == 'mean_val_acc':
                    d['mean_val_acc'] = float(val.replace('%', ''))
                elif key == 'variance':
                    d['variance'] = float(val)
                elif key == 'best_run':
                    d['best_run'] = float(val.replace('%', ''))
                elif key == 'repeats_done':
                    done, total = val.split('/')
                    d['repeats_done'] = int(done)
                    d['n_repeats']    = int(total)
                elif key == 'all_repeat_means':
                    d['all_repeat_means'] = parse_float_list(val)
        results[label] = d
    return results


def print_summary(name, results, paper_targets=None):
    print(f'\n{"="*55}')
    print(f'{name}')
    print(f'{"="*55}')
    for label, d in results.items():
        acc    = d.get('mean_val_acc', float('nan'))
        status = f"{d.get('repeats_done','?')}/{d.get('n_repeats','?')} repeats"
        paper  = paper_targets.get(label, None) if paper_targets else None
        delta  = f'  Δ={acc-paper:+.2f}%' if paper else ''
        print(f'  {LABELS_PRETTY.get(label,label):12s}: {acc:.2f}%  ({status}){delta}')


def fig1_bar_adhd_vs_paper(adhd):
    """Bar chart comparing our ADHD-200 results to paper targets."""
    labels = [l for l in ['falff', 'reho', 'gm', 'falff_gm_multi'] if l in adhd]
    if not labels:
        print('Fig 1: no ADHD data yet')
        return

    ours  = [adhd[l]['mean_val_acc'] for l in labels]
    paper = [PAPER_TARGETS[l] for l in labels]
    x     = np.arange(len(labels))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, ours,  w, label='Ours',
                color=[COLORS[l] for l in labels])
    b2 = ax.bar(x + w/2, paper, w, label='Zou et al. 2017',
                color=[COLORS[l] for l in labels], alpha=0.4, hatch='//')

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, color='gray')

    ax.axhline(50, color='red', linestyle='--', linewidth=1, label='Chance (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS_PRETTY[l] for l in labels])
    ax.set_ylabel('Mean val accuracy (%)')
    ax.set_title('ADHD-200: Our Replication vs Zou et al. 2017')
    ax.set_ylim(40, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig1_bar_adhd_vs_paper.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def fig2_per_repeat_adhd(adhd):
    """Line plot of per-repeat accuracy for ADHD-200."""
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    for label, d in adhd.items():
        vals = d.get('all_repeat_means', [])
        if not vals:
            continue
        c = COLORS[label]
        ax.plot(range(1, len(vals)+1), vals, marker='.', color=c,
                linewidth=1, label=LABELS_PRETTY[label])
        if label in PAPER_TARGETS:
            ax.axhline(PAPER_TARGETS[label], linestyle='--', color=c, alpha=0.4)
        plotted = True

    if not plotted:
        print('Fig 2: no ADHD repeat data yet')
        return

    ax.axhline(50, color='red', linestyle=':', linewidth=1.2, label='Chance')
    ax.set_xlabel('Repeat index')
    ax.set_ylabel('Mean val accuracy (%)')
    ax.set_title('ADHD-200 — per-repeat accuracy (dashed = paper target)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig2_per_repeat_adhd.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def fig3_per_repeat_abide(abide):
    """Line plot of per-repeat accuracy for ABIDE."""
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    for label, d in abide.items():
        vals = d.get('all_repeat_means', [])
        if not vals:
            continue
        ax.plot(range(1, len(vals)+1), vals, marker='.', color=COLORS[label],
                linewidth=1, label=LABELS_PRETTY[label])
        plotted = True

    if not plotted:
        print('Fig 3: no ABIDE repeat data yet')
        return

    ax.axhline(50, color='red', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax.set_xlabel('Repeat index')
    ax.set_ylabel('Mean val accuracy (%)')
    ax.set_title('ABIDE — per-repeat accuracy (autism classification)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig3_per_repeat_abide.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def fig4_adhd_vs_abide(adhd, abide):
    """Side-by-side bar: same model on ADHD-200 vs ABIDE."""
    shared = [l for l in ['falff', 'reho', 'gm', 'falff_gm_multi']
              if l in adhd and l in abide]
    if not shared:
        print('Fig 4: no overlapping results yet')
        return

    adhd_accs  = [adhd[l]['mean_val_acc']  for l in shared]
    abide_accs = [abide[l]['mean_val_acc'] for l in shared]
    x, w = np.arange(len(shared)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, adhd_accs,  w, label='ADHD-200', color='steelblue')
    b2 = ax.bar(x + w/2, abide_accs, w, label='ABIDE (autism)', color='salmon')

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    ax.axhline(50, color='red', linestyle='--', linewidth=1, label='Chance (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS_PRETTY[l] for l in shared])
    ax.set_ylabel('Mean val accuracy (%)')
    ax.set_title('Same 3D CNN: ADHD-200 (ADHD) vs ABIDE (Autism)')
    ax.set_ylim(40, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig4_adhd_vs_abide.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def fig5_boxplots(adhd, abide):
    """Box plots of accuracy distribution across repeats."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (name, dataset) in zip(axes, [('ADHD-200', adhd), ('ABIDE', abide)]):
        data   = [dataset[l]['all_repeat_means'] for l in dataset
                  if dataset[l].get('all_repeat_means')]
        labels = [l for l in dataset if dataset[l].get('all_repeat_means')]
        if not data:
            ax.set_title(f'{name} — no data yet')
            continue
        bp = ax.boxplot(data, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.6)
        ax.axhline(50, color='red', linestyle='--', linewidth=1, label='Chance')
        ax.set_xticklabels([LABELS_PRETTY[l] for l in labels], rotation=15)
        ax.set_ylabel('Mean val accuracy (%)')
        ax.set_title(f'{name} — distribution across repeats')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / 'fig5_boxplots.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    adhd  = load_results(ADHD_DIR,  ['falff', 'reho', 'gm', 'falff_gm_multi'])
    abide = load_results(ABIDE_DIR, ['falff', 'reho', 'gm', 'falff_gm_multi'])

    print_summary('ADHD-200', adhd, PAPER_TARGETS)
    print_summary('ABIDE (autism)', abide)

    fig1_bar_adhd_vs_paper(adhd)
    fig2_per_repeat_adhd(adhd)
    fig3_per_repeat_abide(abide)
    fig4_adhd_vs_abide(adhd, abide)
    fig5_boxplots(adhd, abide)

    print(f'\nAll figures saved to {FIG_DIR}')
