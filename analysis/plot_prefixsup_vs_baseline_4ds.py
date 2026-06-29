#!/usr/bin/env python
"""Does per-prefix deep-supervision + self-distillation training help PATIENCE early-exit?
Overlay the patience (and entropy reference) exit curves for the BASELINE seqA (gap_dual)
vs the PREFIX-SUPERVISION seqA, on GFLOPs-acc/f1, 4 datasets."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(R, 'seqA_exit_schemes_4ds.json')))            # gap_dual baseline
PS = json.load(open(os.path.join(R, 'seqA_prefixsup_exit_schemes_4ds.json')))    # prefix-supervision
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
TAG = {'iemocap': 'M=6', 'daliahar': 'M=5', 'pamap2': 'M=4', 'dsads': 'M=5'}


def xy(src, ds, scheme, yi):
    d = src[ds]; M = d['M']; gpk = d['gflops_per_k']
    c = np.array(d['schemes'][scheme]); x = np.interp(c[:, 0], np.arange(1, M + 1), gpk)
    o = np.argsort(x); return x[o], c[o, yi]


fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for col, ds in enumerate(DS):
    for row, (yi, ylab) in enumerate([(1, 'Accuracy'), (2, 'Macro-F1')]):
        ax = axes[row, col]
        xb, yb = xy(BASE, ds, 'patience', yi); xp, yp = xy(PS, ds, 'patience', yi)
        xeb, yeb = xy(BASE, ds, 'entropy', yi); xep, yep = xy(PS, ds, 'entropy', yi)
        ax.plot(xeb, yeb, '-', color='tab:gray', lw=1.3, alpha=0.7, label='entropy (baseline)')
        ax.plot(xep, yep, '--', color='tab:gray', lw=1.3, alpha=0.7, label='entropy (prefix-sup)')
        ax.plot(xb, yb, '-o', color='tab:blue', lw=2.4, ms=3, label='patience (baseline)')
        ax.plot(xp, yp, '-s', color='tab:red', lw=2.6, ms=3, label='patience (prefix-sup)')
        ax.set_xscale('log'); ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} ({TAG[ds]})' + ('  [cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Per-prefix deep-supervision + self-distillation training vs baseline, for PATIENCE early-exit (seqA, cross-subject). '
             'Red (prefix-sup) vs blue (baseline) patience curves; gray = entropy reference.', fontsize=11)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.90, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_prefixsup_vs_baseline_patience_4ds.png'); fig.savefig(o, dpi=140); print('saved ->', o)

print(f"\n{'dataset':9s} | {'full-mod acc base->PS (Δ)':>26s} | {'patience meanΔacc(PS-base)':>26s} | {'meanΔf1':>9s}  (shared GFLOPs)")
for ds in DS:
    # full-modality (k=M) acc from patience curve's max-meanK point
    cb = np.array(BASE[ds]['schemes']['patience']); cp = np.array(PS[ds]['schemes']['patience'])
    fb = cb[np.argmax(cb[:, 0]), 1]; fp = cp[np.argmax(cp[:, 0]), 1]
    for yi, lab in [(1, 'acc')]:
        xb, yb = xy(BASE, ds, 'patience', 1); xp, yp = xy(PS, ds, 'patience', 1)
        xbf, ybf = xy(BASE, ds, 'patience', 2); xpf, ypf = xy(PS, ds, 'patience', 2)
        lo, hi = max(xb.min(), xp.min()), min(xb.max(), xp.max()); g = np.linspace(lo, hi, 50)
        da = np.interp(g, xp, yp) - np.interp(g, xb, yb)
        df = np.interp(g, xpf, ypf) - np.interp(g, xbf, ybf)
        print(f"{ds:9s} | {fb*100:6.2f} -> {fp*100:6.2f} ({(fp-fb)*100:+5.2f}pp) | {100*da.mean():+23.2f}pp | {100*df.mean():+7.2f}pp")
