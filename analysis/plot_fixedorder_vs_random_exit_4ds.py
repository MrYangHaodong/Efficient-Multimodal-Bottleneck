#!/usr/bin/env python
"""GFLOPs vs Accuracy / Macro-F1 adaptive-early-exit curves comparing seqA trained with
RANDOM modality order vs seqA trained with a FIXED val-LOO order. Both evaluated with the
SAME val-LOO + entropy early-exit method (exit when softmax entropy <= tau; sweep tau).
x = mean GFLOPs (mean #modalities mapped through per-K analytic GFLOPs). 4 datasets x {acc,f1}."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
RAND = json.load(open(os.path.join(R, 'seqA_gap_dual_exit_4ds.json')))     # random-order-trained
FIX = json.load(open(os.path.join(R, 'seqA_fixedorder_exit_4ds.json')))    # fixed-val-LOO-order-trained
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
TAG = {'iemocap': 'M=6, T=128', 'daliahar': 'M=5, T=256', 'pamap2': 'M=4, T=512', 'dsads': 'M=5, T=125'}


def curve(src, ds, which):
    d = src[ds]; M = d['M']; gpk = d['gflops_per_k']
    ec = np.array(d['exit_curve'])            # [meanK, acc, f1]
    mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    x = np.interp(mk, np.arange(1, M + 1), gpk)
    o = np.argsort(x)
    return x[o], y[o]


fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for col, ds in enumerate(DS):
    for row, (which, ylab) in enumerate([('acc', 'Accuracy'), ('f1', 'Macro-F1')]):
        ax = axes[row, col]
        xr, yr = curve(RAND, ds, which)
        xf, yf = curve(FIX, ds, which)
        ax.plot(xr, yr, '-o', color='tab:blue', lw=2.4, ms=5, zorder=6, label='Random-order training (ours)')
        ax.plot(xf, yf, '-s', color='tab:red', lw=2.2, ms=5, zorder=5, label='Fixed val-LOO-order training')
        ax.set_xscale('log')
        ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ttl = f'{ds} ({TAG[ds]})' + ('  [cross-subject]' if row == 0 else '')
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=9, loc='lower right')
fig.suptitle('Adaptive modality early-exit under cross-subject shift: seqA trained with RANDOM order (blue) vs FIXED val-LOO order (red).\n'
             'Both use the same val-LOO + softmax-entropy exit (sweep $\\tau$); x = mean GFLOPs/sample (analytic, per-K). Per-panel log x-axis.', fontsize=10.5)
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.08, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_fixedorder_vs_random_exit_4ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)

# numeric summary: random-order advantage at matched compute (interp fixed acc onto random's GFLOPs grid)
print(f"\n{'dataset':9s} {'metric':4s} {'rand range':>18s} {'fixed range':>18s} {'mean Δ(rand-fix) over shared GFLOPs':>34s}")
for ds in DS:
    for which in ['acc', 'f1']:
        xr, yr = curve(RAND, ds, which); xf, yf = curve(FIX, ds, which)
        lo, hi = max(xr.min(), xf.min()), min(xr.max(), xf.max())
        grid = np.linspace(lo, hi, 50)
        dr = np.interp(grid, xr, yr) - np.interp(grid, xf, yf)
        print(f"{ds:9s} {which:4s} {yr.min():.3f}-{yr.max():.3f} @{xr.min():.2f}-{xr.max():.2f} "
              f"{yf.min():.3f}-{yf.max():.3f} @{xf.min():.2f}-{xf.max():.2f}   {100*dr.mean():+.2f}pp")
