#!/usr/bin/env python
"""GFLOPs vs Acc / Macro-F1 on EAV: seqA-prefixsup ADAPTIVE EARLY-EXIT curve
(val-LOO modality order, audio->video->eeg) vs crossattn / multimodn / shaspec /
decalign / dymo full-modality points. 5-class emotion, M=3, T=128, 3 cross-subject
folds (mean). Mirrors plot_mosei_seqA_vs_baselines.py.
Run from clean_models/train/:  python ../results_3p3/plot_eav_seqA_vs_baselines.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(R), 'train', 'results_eav')
M = 3
STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'shaspec': ('tab:brown', '^'), 'decalign': ('tab:red', 'D')}

EXIT = json.load(open(os.path.join(R, 'eav', 'exit_eav.json')))['eav']
BGF = json.load(open(os.path.join(R, 'eav', 'baselines_gflops_eav.json')))
ORDER = EXIT['consensus_order']                       # ['audio','video','eeg']


def seqa_exit(which):
    ec = np.array(EXIT['exit_curve']); gpk = EXIT['gflops_per_k']
    mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    gf = np.interp(mk, np.arange(1, M + 1), gpk); o = np.argsort(gf)
    return gf[o], y[o]


def base_acc(method, key):
    v = []
    for d in glob.glob(os.path.join(RES, method, '*')):
        for rj in sorted(glob.glob(os.path.join(d, 'results_fold*.json'))):
            x = json.load(open(rj)).get(key)
            if x is not None:
                v.append(x)
    return float(np.mean(v)) if v else None


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (which, rk, ylab) in zip(axes, [('acc', 'best_test_acc', 'Accuracy (5-class)'),
                                        ('f1', 'best_test_f1', 'Macro-F1')]):
    gf, y = seqa_exit(which)
    ax.plot(gf, y, '-o', color='tab:blue', lw=2.6, ms=6, zorder=7,
            label='seqA-prefixsup (ours, adaptive early-exit)')
    for k, lbl in [(1, f'1 ({ORDER[0]})'), (2, '2'), (3, '3')]:
        gk = EXIT['gflops_per_k'][k - 1]
        yk = np.interp(gk, gf, y)
        ax.annotate(f'k={lbl}', (gk, yk), textcoords='offset points', xytext=(4, -12),
                    fontsize=8, color='tab:blue')
    for method, (color, mk) in STYLE.items():
        g = BGF.get(method, {}).get('eav', {}).get('true_gflops')
        a = base_acc(method, rk)
        if g is not None and a is not None:
            ax.plot([g], [a], mk, color=color, ms=13, zorder=6, label=method)
    gd = BGF.get('dymo', {}).get('eav', {}).get('true_gflops')
    ad = base_acc('dymo', rk)
    if gd is not None and ad is not None:
        ax.plot([gd], [ad], 'X', color='tab:green', ms=14, zorder=6, label='dymo-proto')
    ax.set_xscale('log')
    ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, which='both')
    ax.set_title(f'{ylab} vs GFLOPs')
    if which == 'acc':
        ax.legend(fontsize=9, loc='lower right')
fig.suptitle('EAV (M=3, T=128, 5-class emotion) — seqA-prefixsup adaptive early-exit curve '
             f'(val-LOO order: {"→".join(ORDER)}) vs full-modality baselines. Cross-subject 3-fold mean.\n'
             f'seqA k=1 uses {ORDER[0].upper()} only (~{EXIT["gflops_per_k"][0]:.2f} GFLOPs); '
             'decalign ~6x heavier (M(M-1) cross-attn + OT).',
             fontsize=10.5)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.11, top=0.86, wspace=0.18)
o = os.path.join(R, 'gflops_acc_f1_eav_seqA_vs_baselines.png')
fig.savefig(o, dpi=150); print('saved ->', o)

gf, ya = seqa_exit('acc')
print(f"seqA-exit acc {ya.min():.3f}-{ya.max():.3f} @ GFLOPs {gf.min():.3f}-{gf.max():.3f}")
for m in list(STYLE) + ['dymo']:
    g = BGF.get(m, {}).get('eav', {}).get('true_gflops'); a = base_acc(m, 'best_test_acc')
    print(f"  {m:10s} GFLOPs={g:.3f}  acc={a:.4f}" if g and a else f"  {m}: n/a")
