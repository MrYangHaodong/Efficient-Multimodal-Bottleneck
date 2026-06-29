#!/usr/bin/env python
"""Combine the EXACT entropy/patience/bottleneck curves from gflops_exit_schemes_4ds.png
(seqA_exit_schemes_4ds.json, gap_dual model) with an R-as-exit-signal curve computed on the
SAME gap_dual model (arrays_gapdual_<ds>.npz). 4 lines, acc & macro-F1, 4 datasets.
R-exit = per-sample 2-bucket on ORACLE LSMI redundancy (high-R -> 1 modality, low-R -> all M)."""
import json, os
import numpy as np
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R_ = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R_, 'seqA_exit_schemes_4ds.json')))   # gap_dual entropy/patience/bottleneck
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
TAG = {'iemocap': 'M=6, T=128', 'daliahar': 'M=5, T=256', 'pamap2': 'M=4, T=512', 'dsads': 'M=5, T=125'}
# name -> (label, color, marker, dash, is_R)
STYLE = {
    'entropy':   ('Entropy',               'tab:blue',   'o', '-',  False),
    'patience':  ('Patience (continuous)', 'tab:purple', 'o', '-',  False),
    'bottleneck':('Bottleneck saturation', 'tab:cyan',   'o', '--', False),
    'R':         ('R (oracle redundancy)', 'tab:red',    'o', '-',  True),
}


def gflops(ds, meanK):
    M = D[ds]['M']; gpk = D[ds]['gflops_per_k']; return np.interp(meanK, np.arange(1, M + 1), gpk)


def Rcurve(ds):
    """per-sample 2-bucket R-exit on gap_dual: sweep R-threshold -> (meanK, acc, f1)."""
    z = np.load(os.path.join(R_, 'lsmi_exit', f'arrays_gapdual_{ds}.npz'))
    Rv = z['R']; cor = z['correct']; pk = z['preds_km']; y = z['y_all']; M = int(z['M']); N = len(Rv)
    c1, cM = cor[:, 0], cor[:, -1]; p1, pM = pk[:, 0], pk[:, -1]
    order = np.argsort(-Rv)                         # most redundant first -> exit on 1 modality
    out = []
    for p in np.linspace(0, 1, 21):
        e = int(round(p * N)); early = order[:e]
        mask = np.zeros(N, bool); mask[early] = True
        pe = np.where(mask, p1, pM)
        acc = (c1[early].sum() + cM[order[e:]].sum()) / N
        out.append((1 * p + M * (1 - p), acc, f1_score(y, pe, average='macro')))
    return np.array(sorted(out)), z['gflops_per_k']


fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
print(f"{'dataset':9s} {'full-acc(canon)':>15s} {'full-acc(R arr)':>15s}  match?")
for col, ds in enumerate(DS):
    rc, gpk_r = Rcurve(ds)
    # sanity: full-modality acc must match between the canonical curves and the R-array
    canon_full = np.array(D[ds]['schemes']['entropy']); cf = canon_full[np.argmax(canon_full[:, 0]), 1]
    rf = rc[np.argmax(rc[:, 0]), 1]
    print(f"{ds:9s} {cf:15.3f} {rf:15.3f}  {'OK' if abs(cf-rf)<0.02 else 'DIFFER'}")
    for row, (yi, ylab) in enumerate([(1, 'Accuracy'), (2, 'Macro-F1')]):
        ax = axes[row, col]
        for nm, (lab, color, mk, dash, isR) in STYLE.items():
            if isR:
                x = gflops(ds, rc[:, 0]); y = rc[:, yi]
            else:
                cur = np.array(D[ds]['schemes'][nm]); x = gflops(ds, cur[:, 0]); y = cur[:, yi]
            o = np.argsort(x)
            lw = 2.8 if isR else (2.4 if nm in ('patience', 'bottleneck') else 1.8)
            ms = 5 if isR else 0
            ax.plot(x[o], y[o], dash, marker=mk, color=color, lw=lw, ms=ms, zorder=8 if isR else 5, label=lab)
        ax.set_xscale('log'); ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} ({TAG[ds]})' + ('  [gap_dual, cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0: ax.legend(fontsize=9, loc='lower right')
fig.suptitle('Adaptive modality early-exit (gap_dual seqA, cross-subject): per-prefix confidence schemes '
             '(entropy / patience / bottleneck) vs R-as-exit-signal. R = oracle LSMI redundancy, per-sample 2-bucket. '
             'All 4 lines on the SAME model.', fontsize=10.5)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.90, wspace=0.27, hspace=0.30)
o = os.path.join(R_, 'gflops_exit_schemes_withR_combined_4ds.png'); fig.savefig(o, dpi=140); print('\nsaved ->', o)

# numeric: mean acc/f1 of R vs the three schemes over shared GFLOPs range
print(f"\n{'dataset':9s} {'metric':4s} | {'entropy':>8s} {'patience':>8s} {'bottlenk':>8s} {'R':>8s}   R rank")
for ds in DS:
    rc, _ = Rcurve(ds); M = D[ds]['M']; gpk = D[ds]['gflops_per_k']; grid = np.linspace(gpk[0], gpk[-1], 50)
    for yi, mlab in [(1, 'acc'), (2, 'f1')]:
        vals = {}
        for nm in ['entropy', 'patience', 'bottleneck']:
            cur = np.array(D[ds]['schemes'][nm]); x = gflops(ds, cur[:, 0]); o = np.argsort(x)
            vals[nm] = np.interp(grid, x[o], cur[o, yi]).mean()
        xr = gflops(ds, rc[:, 0]); o = np.argsort(xr); vals['R'] = np.interp(grid, xr[o], rc[o, yi]).mean()
        rank = 1 + sum(v > vals['R'] for k, v in vals.items() if k != 'R')
        print(f"{ds:9s} {mlab:4s} | {vals['entropy']:8.3f} {vals['patience']:8.3f} {vals['bottleneck']:8.3f} {vals['R']:8.3f}   {rank}/4")
