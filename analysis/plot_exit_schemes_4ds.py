#!/usr/bin/env python
"""GFLOPs vs Accuracy / Macro-F1 adaptive-early-exit curves for seqA, comparing exit
CONFIDENCE SCHEMES: entropy (current) vs max-softmax-prob, margin (top1-2), energy, and
patience (PABEE-style, threshold-free points). Same deployed random-order seqA, 4 datasets."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R, 'seqA_exit_schemes_4ds.json')))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
TAG = {'iemocap': 'M=6, T=128', 'daliahar': 'M=5, T=256', 'pamap2': 'M=4, T=512', 'dsads': 'M=5, T=125'}
STYLE = {  # name: (label, color, marker, is_curve)
    'entropy':  ('Entropy (current)',         'tab:blue',  'o', True),
    'maxprob':  ('Max-softmax-prob',          'tab:red',   's', True),
    'margin':   ('Margin (top1-top2)',        'tab:green', '^', True),
    'energy':   ('Energy (logsumexp)',        'tab:orange','D', True),
    'patience': ('Patience (continuous)',     'tab:purple','o', True),
    'bottleneck':('Bottleneck saturation',    'tab:cyan',  'o', True),
    'patience2':('Patience=2 (anchor)',       'k',         '*', False),
    'patience3':('Patience=3 (anchor)',       'dimgray',   'X', False),
}


def gflops(ds, meanK):
    M = D[ds]['M']; gpk = D[ds]['gflops_per_k']
    return np.interp(meanK, np.arange(1, M + 1), gpk)


fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for col, ds in enumerate(DS):
    for row, (yi, ylab) in enumerate([(1, 'Accuracy'), (2, 'Macro-F1')]):
        ax = axes[row, col]
        for nm, (lab, color, mk, is_curve) in STYLE.items():
            cur = np.array(D[ds]['schemes'][nm])           # [npts,3] = meanK,acc,f1
            x = gflops(ds, cur[:, 0]); y = cur[:, yi]
            o = np.argsort(x)
            if is_curve:
                lw = 2.6 if nm in ('patience', 'bottleneck') else 1.8
                ms = 0 if nm in ('patience', 'bottleneck') else 4
                z = 7 if nm in ('patience', 'bottleneck') else 5
                dash = '--' if nm == 'bottleneck' else '-'
                ax.plot(x[o], y[o], dash, marker=mk, color=color, lw=lw, ms=ms, zorder=z, label=lab)
            else:
                ax.scatter(x, y, color=color, marker=mk, s=150, zorder=9, edgecolor='w', linewidth=0.7, label=lab)
        ax.set_xscale('log')
        ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} ({TAG[ds]})' + ('  [cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Adaptive modality early-exit: confidence-SCHEME comparison for seqA (random-order, cross-subject). '
             'Curves sweep each scheme\'s threshold; patience points are threshold-free. x = mean GFLOPs/sample (analytic). Per-panel log x.', fontsize=10.5)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.90, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_exit_schemes_4ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)

# numeric: mean acc advantage of each scheme vs entropy over shared GFLOPs range
print(f"\n{'dataset':9s} {'scheme':12s} {'mean Δacc vs entropy':>22s} {'mean Δf1 vs entropy':>22s}  (over shared GFLOPs)")
for ds in DS:
    ent = np.array(D[ds]['schemes']['entropy']); xe = gflops(ds, ent[:, 0])
    for nm in ['maxprob', 'margin', 'energy', 'patience', 'bottleneck']:
        cur = np.array(D[ds]['schemes'][nm]); xc = gflops(ds, cur[:, 0])
        lo, hi = max(xe.min(), xc.min()), min(xe.max(), xc.max())
        g = np.linspace(lo, hi, 50)
        da = np.interp(g, np.sort(xc), cur[np.argsort(xc), 1]) - np.interp(g, np.sort(xe), ent[np.argsort(xe), 1])
        df = np.interp(g, np.sort(xc), cur[np.argsort(xc), 2]) - np.interp(g, np.sort(xe), ent[np.argsort(xe), 2])
        print(f"{ds:9s} {nm:12s} {100*da.mean():+19.2f}pp {100*df.mean():+19.2f}pp")
