#!/usr/bin/env python
"""seqA adaptive early-exit: val-LOO modality order vs RANDOM order (each sample
averaged over 5 random orders). GFLOPs vs Acc / Macro-F1 on 3 datasets.
At matched GFLOPs (=meanK), val-LOO front-loads the strong modalities -> higher
acc at low compute; the two converge at full (all modalities, order irrelevant)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R, 'seqA_vlo_vs_random_exit_3ds.json')))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
TAG = {'iemocap': 'T=128 float', 'daliahar': 'T=256 sax', 'pamap2': 'T=512 sax_noisy', 'dsads': 'T=125 sax'}


def curve(o, key, which):  # which: 'vlo'/'random'; key col 1=acc 2=f1
    M = o['M']; gpk = o['gflops_per_k']
    a = np.array(o[which])                       # (meanK, acc, f1) over taus
    gf = np.interp(a[:, 0], np.arange(1, M + 1), gpk)
    y = a[:, key]; idx = np.argsort(gf)
    return gf[idx], y[idx]


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, ds in enumerate(DS):
    o = D[ds]
    for row, (key, ylab) in enumerate([(1, 'Accuracy'), (2, 'Macro-F1')]):
        ax = axes[row, col]
        gx, vy = curve(o, key, 'vlo'); _, ry = curve(o, key, 'random')
        ax.plot(gx, vy, '-o', color='tab:blue', lw=2.4, ms=4, zorder=6,
                label='val-LOO order (strong-first)')
        ax.plot(gx, ry, '--s', color='tab:red', lw=2.0, ms=4, zorder=5,
                label='random order (avg of 5/sample)')
        ax.set_xscale('log')
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} (M={o["M"]}, {TAG[ds]})' + ('  [seqA early-exit, cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=9, loc='lower right')
fig.suptitle('seqA adaptive early-exit: val-LOO modality order vs RANDOM order (5 random orders/sample) — GFLOPs vs Acc/F1, cross-subject.\n'
             'val-LOO front-loads the strongest modalities -> higher accuracy at low compute; curves converge at full (all modalities, order irrelevant).', fontsize=11)
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.08, top=0.89, wspace=0.22, hspace=0.30)
o = os.path.join(R, 'seqA_vlo_vs_random_exit_4ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    o2 = D[ds]; gx, vy = curve(o2, 1, 'vlo'); _, ry = curve(o2, 1, 'random')
    # acc at the cheapest point (lowest GFLOPs) + max gap
    gaps = vy - ry
    print(f"{ds:9s}: cheapest GFLOPs={gx[0]:.3f} vlo_acc={vy[0]:.3f} random_acc={ry[0]:.3f} (gap {gaps[0]:+.3f}) | max gap {gaps.max():+.3f}")
