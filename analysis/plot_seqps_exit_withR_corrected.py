#!/usr/bin/env python
"""CORRECTED prefix-sup (seqA_ps) adaptive-exit figure with R. Key fixes vs the earlier version:
(1) PATIENCE (prefix-sup) is the best DEPLOYABLE scheme — it covers the whole frontier incl. the
1-modality end; (2) BOTTLENECK saturation can only operate at meanK>=2 (needs >=2 boards), so it
is drawn ONLY over its valid range and the <2 region is shaded as 'bottleneck unavailable';
(3) R (oracle redundancy) is the worst. AUC reported over the FAIR shared range. Reads exit_withR.json."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R_ = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R_, 'lsmi_exit', 'exit_withR.json')))   # seqA_ps; all 4 schemes converge
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']
TAG = {'iemocap': 'M=6', 'daliahar': 'M=5', 'pamap2': 'M=4', 'dsads': 'M=5', 'mosei': 'M=3'}
# name -> (label, color, marker, dash, lw, ms, z)
STYLE = {
    'patience':    ('Patience (prefix-sup) — BEST', 'tab:purple', 'o', '-',  3.0, 4, 9),
    'bottleneck':  ('Bottleneck saturation (≥2 mod)', 'tab:cyan', 's', '--', 2.0, 3, 6),
    'entropy':     ('Entropy',                          'tab:gray',  '^', '-',  1.6, 0, 4),
    'R (oracle)':  ('R (oracle redundancy) — worst','tab:red',   'o', '-',  2.4, 5, 8),
}


def gx(ds, meanK):
    M = D[ds]['M']; g = D[ds]['gflops_per_k'] if 'gflops_per_k' in D[ds] else D[ds]['gpk']
    return np.interp(meanK, np.arange(1, M + 1), g)


fig, axes = plt.subplots(2, len(DS), figsize=(4.7 * len(DS), 8.5))
print(f"{'ds':9s} | fair-range AUC  patience  bottleneck  entropy   R")
for col, ds in enumerate(DS):
    M = D[ds]['M']; g = np.array(D[ds]['gpk'])
    cur = {k: np.array(D[ds][k]) for k in STYLE}
    g2 = gx(ds, 2.0)   # GFLOPs at meanK=2 (below this, bottleneck cannot operate)
    # fair shared range across ALL schemes (intersection of meanK spans)
    lo_k = max(c[:, 0].min() for c in cur.values()); hi_k = min(c[:, 0].max() for c in cur.values())
    grid = np.linspace(gx(ds, lo_k), gx(ds, hi_k), 50)
    auc = {}
    for k, c in cur.items():
        x = gx(ds, c[:, 0]); o = np.argsort(x); auc[k] = np.interp(grid, x[o], c[o, 1]).mean()
    print(f"{ds:9s} |                {auc['patience']:.3f}     {auc['bottleneck']:.3f}    {auc['entropy']:.3f}  {auc['R (oracle)']:.3f}")
    for row, (yi, ylab) in enumerate([(1, 'Accuracy'), (2, 'Macro-F1')]):
        ax = axes[row, col]
        # shade the region bottleneck cannot reach
        ax.axvspan(gx(ds, 1.0), g2, color='0.92', zorder=0)
        ax.text(np.sqrt(gx(ds, 1.0) * g2), ax.get_ylim()[0], '', fontsize=7)
        for nm, (lab, color, mk, dash, lw, ms, z) in STYLE.items():
            c = cur[nm]; x = gx(ds, c[:, 0]); y = c[:, yi]; o = np.argsort(x)
            ax.plot(x[o], y[o], dash, marker=mk, color=color, lw=lw, ms=ms, zorder=z, label=lab if row == 0 and col == 0 else None)
        ax.set_xscale('log'); ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ttl = f'{ds} ({TAG[ds]})'
        if row == 0:
            ttl += '  [seqA prefix-sup, cross-subject]'
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.25, which='both')
        if row == 0 and col == 0:
            ax.legend(fontsize=8.5, loc='lower right')
        if row == 0 and col == len(DS) - 1:
            ax.annotate('grey = <2 modalities:\nbottleneck cannot exit here\n(only patience/entropy can)',
                        xy=(0.02, 0.97), xycoords='axes fraction', fontsize=7.5, va='top', color='0.35')
fig.suptitle('Prefix-supervised seqA, adaptive modality early-exit (cross-subject): PATIENCE is the best deployable scheme '
             '(covers the full frontier incl. 1 modality). Bottleneck is capped at ≥2 modalities; R (oracle redundancy) is worst.', fontsize=10.5)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.90, wspace=0.27, hspace=0.30)
o = os.path.join(R_, 'gflops_exit_schemes_withR_prefixsup_5ds.png'); fig.savefig(o, dpi=140); print('\nsaved ->', o)
