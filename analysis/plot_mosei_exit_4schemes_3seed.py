#!/usr/bin/env python
"""CMU-MOSEI adaptive early-exit on seqA (3 seeds {21,365,1024}), comparing 4 exit-scheme
variants: entropy / max-softmax-prob / patience / bottleneck-saturation. GFLOPs-acc/f1."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R, 'mosei', 'exit_schemes_mosei_3seed.json')))
TAG = 'seqA'   # the 3-seed seqA model
SCH = {'entropy': ('Entropy', 'tab:gray', '-o'), 'maxprob': ('Max-softmax-prob', 'tab:red', '-s'),
       'patience': ('Patience', 'tab:purple', '-^'), 'bottleneck': ('Bottleneck saturation', 'tab:cyan', '--D')}


def xy(sch, yi):
    d = D[TAG]; M = d['M']; g = d['gflops_per_k']
    c = np.array(d['schemes'][sch]); x = np.interp(c[:, 0], np.arange(1, M + 1), g)
    o = np.argsort(x); return x[o], c[o, yi]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
for ax, (yi, ylab) in zip(axes, [(1, 'Accuracy (Acc-2)'), (2, 'Macro-F1')]):
    for sch, (lab, col, st) in SCH.items():
        x, y = xy(sch, yi); ax.plot(x, y, st, color=col, lw=2.3, ms=6, label=lab)
    ax.set_xscale('log'); ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, which='both')
    if yi == 1:
        ax.legend(fontsize=9, loc='lower right')
g = D[TAG]['gflops_per_k']
fig.suptitle(f'CMU-MOSEI adaptive modality early-exit on seqA (3 seeds {{21,365,1024}}, M=3 vision/audio/text) — 4 exit-scheme variants.\n'
             f'GFLOPs/sample: text-only {g[0]:.3f} -> +1 {g[1]:.3f} -> all-3 {g[2]:.3f}  (text-only = {100*g[0]/g[2]:.0f}% of full compute).', fontsize=10)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.84, wspace=0.22)
o = os.path.join(R, 'mosei', 'gflops_mosei_exit_4schemes_seqA_3seed.png'); fig.savefig(o, dpi=140); print('saved ->', o)

print(f"\nseqA (3-seed) exit schemes — acc at cheapest -> full:")
for sch, (lab, _, _) in SCH.items():
    c = np.array(D[TAG]['schemes'][sch])
    print(f"  {lab:22s}: {c[:,1].min():.4f} -> {c[:,1].max():.4f}   (meanK {c[:,0].min():.2f}->{c[:,0].max():.2f})")
