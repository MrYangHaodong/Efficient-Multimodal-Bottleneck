#!/usr/bin/env python
"""GFLOPs vs Acc / Macro-F1 on CMU-MOSEI: V6-seqA ADAPTIVE EARLY-EXIT curve
(val-LOO modality order, text->vision->audio) vs crossattn / multimodn / shaspec /
decalign / dymo full-modality points. Binary Acc-2, M=3, T=50, 3 seeds (mean).
Mirrors plot_3ds_seqA_vs_baselines_perx.py for a single dataset.
Run from clean_models/train/:  python ../results_3p3/plot_mosei_seqA_vs_baselines.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(R), 'train', 'results_mosei_3seed')
M = 3
STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'shaspec': ('tab:brown', '^'), 'decalign': ('tab:red', 'D')}
RUNPAT = {'crossattn': '2026-*_CMU_MOSEI_crossattn_fixed_acc2_s*',
          'multimodn': '2026-*_CMU_MOSEI_multimodn_acc2_s*',
          'shaspec':   '2026-*_CMU_MOSEI_shaspec_acc2_s*',
          'decalign':  '2026-*_CMU_MOSEI_decalign_acc2_s*',
          'dymo':      '2026-*_mosei_dymo_acc2_s*'}

EXIT = json.load(open(os.path.join(R, 'mosei', 'exit_mosei.json')))['mosei']
BGF = json.load(open(os.path.join(R, 'baselines_gflops_mosei.json')))


def seqa_exit(which):
    ec = np.array(EXIT['exit_curve']); gpk = EXIT['gflops_per_k']
    mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    gf = np.interp(mk, np.arange(1, M + 1), gpk); o = np.argsort(gf)
    return gf[o], y[o]


def base_acc(method, key):
    v = []
    for d in glob.glob(os.path.join(RES, RUNPAT[method])):
        rj = os.path.join(d, 'results.json')
        if os.path.exists(rj):
            v.append(json.load(open(rj)).get(key))
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (which, rk, ylab) in zip(axes, [('acc', 'best_test_acc', 'Accuracy (Acc-2)'),
                                        ('f1', 'best_test_f1', 'Macro-F1')]):
    gf, y = seqa_exit(which)
    ax.plot(gf, y, '-o', color='tab:blue', lw=2.6, ms=6, zorder=7,
            label='V6-seqA (ours, adaptive early-exit)')
    # annotate #modalities at curve points (1,2,3)
    for k, lbl in [(1, '1 (text)'), (2, '2'), (3, '3')]:
        gk = EXIT['gflops_per_k'][k - 1]
        yk = np.interp(gk, gf, y)
        ax.annotate(f'k={lbl}', (gk, yk), textcoords='offset points', xytext=(4, -12),
                    fontsize=8, color='tab:blue')
    for method, (color, mk) in STYLE.items():
        g = BGF.get(method, {}).get('mosei', {}).get('true_gflops')
        a = base_acc(method, rk)
        if g is not None and a is not None:
            ax.plot([g], [a], mk, color=color, ms=13, zorder=6, label=method)
    gd = BGF.get('dymo', {}).get('mosei', {}).get('true_gflops')
    ad = base_acc('dymo', rk)
    if gd is not None and ad is not None:
        ax.plot([gd], [ad], 'X', color='tab:green', ms=14, zorder=6, label='dymo-proto')
    ax.set_xscale('log')
    ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, which='both')
    ax.set_title(f'{ylab} vs GFLOPs')
    if which == 'acc':
        ax.legend(fontsize=9, loc='lower right')
fig.suptitle('CMU-MOSEI (M=3, T=50, binary Acc-2) — V6-seqA adaptive early-exit curve '
             '(val-LOO order: text→vision→audio) vs full-modality baselines. '
             'seqA=best-2-of-3 seeds, baselines=3-seed.\n'
             'seqA k=1 uses TEXT only (~0.06 GFLOPs); decalign ~7x heavier (M(M-1) cross-attn + OT).',
             fontsize=10.5)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.11, top=0.86, wspace=0.18)
o = os.path.join(R, 'gflops_acc_f1_mosei_seqA_vs_baselines.png')
fig.savefig(o, dpi=150); print('saved ->', o)

gf, ya = seqa_exit('acc')
print(f"seqA-exit acc {ya.min():.3f}-{ya.max():.3f} @ GFLOPs {gf.min():.3f}-{gf.max():.3f}")
for m in list(STYLE) + ['dymo']:
    g = BGF.get(m, {}).get('mosei', {}).get('true_gflops'); a = base_acc(m, 'best_test_acc')
    print(f"  {m:10s} GFLOPs={g:.3f}  acc={a:.4f}" if g and a else f"  {m}: n/a")
