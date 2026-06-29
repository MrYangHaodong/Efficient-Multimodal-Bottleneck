#!/usr/bin/env python
"""CMU-MOSEI adaptive early-exit: seqA ORIGINAL vs PREFIX-SUPERVISION, GFLOPs-acc/f1.
Schemes entropy/patience/bottleneck (M=3, 5-seed mean)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R, 'mosei', 'exit_schemes_mosei.json')))
SCH = {'entropy': ('Entropy', 'tab:gray'), 'patience': ('Patience', 'tab:purple'), 'bottleneck': ('Bottleneck sat.', 'tab:cyan')}


def xy(tag, sch, yi):
    d = D[tag]; M = d['M']; g = d['gflops_per_k']
    c = np.array(d['schemes'][sch]); x = np.interp(c[:, 0], np.arange(1, M + 1), g)
    o = np.argsort(x); return x[o], c[o, yi]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (yi, ylab) in zip(axes, [(1, 'Accuracy (Acc-2)'), (2, 'Macro-F1')]):
    for sch, (lab, col) in SCH.items():
        xs, ys = xy('seqA', sch, yi); ax.plot(xs, ys, '-o', color=col, lw=2.2, ms=5, label=f'{lab} — seqA')
        xp, yp = xy('seqA_ps', sch, yi); ax.plot(xp, yp, '--s', color=col, lw=2.0, ms=4, alpha=0.85, label=f'{lab} — seqA+prefix-sup')
    ax.set_xscale('log'); ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, which='both')
    if yi == 1:
        ax.legend(fontsize=7.5, loc='lower right', ncol=2)
g = D['seqA']['gflops_per_k']
fig.suptitle(f'CMU-MOSEI adaptive modality early-exit (M=3 vision/audio/text, 5-seed mean). seqA (solid) vs seqA+prefix-supervision (dashed).\n'
             f'GFLOPs/sample per #modalities: text-only {g[0]:.3f} -> +1 {g[1]:.3f} -> all-3 {g[2]:.3f}. The curve is near-flat: text alone reaches ~98% of full accuracy.', fontsize=10)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.84, wspace=0.22)
o = os.path.join(R, 'mosei', 'gflops_mosei_exit_schemes.png'); fig.savefig(o, dpi=140); print('saved ->', o)

print(f"\n{'scheme':11s} {'seqA k=1->k=3 acc':>22s} {'seqA_ps k=1->k=3 acc':>24s}")
for sch in SCH:
    cs = np.array(D['seqA']['schemes'][sch]); cp = np.array(D['seqA_ps']['schemes'][sch])
    print(f"{sch:11s} {cs[:,1].min():.3f} -> {cs[:,1].max():.3f}        {cp[:,1].min():.3f} -> {cp[:,1].max():.3f}")
print(f"\nGFLOPs/sample: text-only={g[0]:.3f}  +1mod={g[1]:.3f}  all-3={g[2]:.3f}  "
      f"(text-only = {100*g[0]/g[2]:.0f}% of full compute)")
