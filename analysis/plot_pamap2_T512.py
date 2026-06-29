#!/usr/bin/env python
"""PAMAP2 (T=512, clean sax, cross-subject): V6-seqA ADAPTIVE EARLY-EXIT curve vs
crossattn / multimodn full-modality points. GFLOPs vs Accuracy / Macro-F1.
Uses the partial pamap2_T512 retrain (3 folds each for seqA/crossattn/multimodn)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(R, 'seqA_pamap2_T512_exit.json')))['pamap2']
BGF = json.load(open(os.path.join(R, 'pamap2_T512_baseline_gflops.json')))
M = EX['M']; gpk = EX['gflops_per_k']
ec = np.array(EX['exit_curve'])                                   # (meanK, acc, f1)
gf_exit = np.interp(ec[:, 0], np.arange(1, M + 1), gpk)


def acc(method, key):
    by = {}
    for f in glob.glob(os.path.join(R, 'pamap2_T512', method, '*', 'results_fold*.json')):
        k = f.split('results_fold')[-1].split('.')[0]
        if k not in by or os.path.getmtime(f) > os.path.getmtime(by[k]):
            by[k] = f
    v = [json.load(open(p)).get(key) for p in by.values()]
    return float(np.mean([x for x in v if x is not None]))


STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's')}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (which, rk, ylab) in zip(axes, [(1, 'best_test_acc', 'Accuracy'), (2, 'best_test_f1', 'Macro-F1')]):
    ax.plot(gf_exit, ec[:, which], '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
            label='V6-seqA (ours, adaptive early-exit)')
    for method, (color, mk) in STYLE.items():
        ax.plot([BGF[method]], [acc(method, rk)], mk, color=color, ms=13, zorder=6, label=method)
    ax.set_xscale('log'); ax.set_xlim(0.3, 3.0)
    ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.set_title(f'PAMAP2 (M={M}, T=512, clean sax, cross-subject)', fontsize=11)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9, loc='lower right')
fig.suptitle('PAMAP2 T=512: V6-seqA adaptive early-exit (curve) vs crossattn / multimodn (full-modality points). mean over 3 folds.', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
o = os.path.join(R, 'gflops_pamap2_T512_seqA_vs_baselines.png')
fig.savefig(o, dpi=150); print('saved ->', o)
print(f"seqA-exit: acc {ec[:,1].min():.3f}->{ec[:,1].max():.3f} f1 {ec[:,2].min():.3f}->{ec[:,2].max():.3f} over GFLOPs {gf_exit.min():.3f}-{gf_exit.max():.3f}")
for m in STYLE:
    print(f"  {m}: GFLOPs={BGF[m]:.3f} acc={acc(m,'best_test_acc'):.3f} f1={acc(m,'best_test_f1'):.3f}")
