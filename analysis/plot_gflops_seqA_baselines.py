#!/usr/bin/env python
"""GFLOPs vs Accuracy / Macro-F1 (cross-subject alpha=0):
  V6-seqA = per-K static LINE; crossattn / multimodn / decalign / shaspec = single
  full-modality POINTS (full-modality GFLOPs + mean acc/F1 over 3 folds)."""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['daliahar', 'pamap2', 'wesad', 'iemocap']
SEQA_ACC = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))
SEQA_GF = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))
BGF = json.load(open(os.path.join(R, 'baselines_gflops_corrected.json')))  # true = library + attention
STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'decalign': ('tab:red', 'D'), 'shaspec': ('tab:brown', '^')}


def folds_mean(method, ds, key):
    by = {}
    for f in glob.glob(os.path.join(R, f'{method}_{ds}/*/results_fold*.json')):
        mo = re.search(r'results_fold(\d+)\.json$', f)
        if mo and (mo.group(1) not in by or os.path.getmtime(f) > os.path.getmtime(by[mo.group(1)])):
            by[mo.group(1)] = f
    v = [json.load(open(p)).get(key) for p in by.values()]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, ds in enumerate(DS):
    for row, (exit_key, res_key, ylab) in enumerate([
            ('exit_acc', 'best_test_acc', 'Accuracy'),
            ('exit_f1', 'best_test_f1', 'Macro-F1')]):
        ax = axes[row, col]
        # V6-seqA ADAPTIVE-EXIT curve: per-sample entropy early-exit over 21 thresholds;
        # map mean #modalities-used-at-exit -> GFLOPs via the per-K cost curve.
        gpk = SEQA_GF[ds]['gflops_per_k']; Mg = len(gpk)
        mk = SEQA_ACC[ds]['exit_meanK']; ey = SEQA_ACC[ds][exit_key]
        gf_exit = np.interp(mk, np.arange(1, Mg + 1), gpk)
        ax.plot(gf_exit, ey, '-o', color='tab:blue', lw=2.4, ms=4,
                label='V6-seqA (ours, adaptive-exit)', zorder=7)
        for method, (color, mk) in STYLE.items():
            gg = BGF[method].get(ds); g = gg['true_gflops'] if gg else None
            y = folds_mean(method, ds, res_key)
            if g is not None and y is not None:
                ax.plot([g], [y], mk, color=color, ms=12, label=method, zorder=6)
        ax.set_xscale('log'); ax.set_xlim(0.04, 50)   # fixed scale across all panels
        M = len(json.load(open(glob.glob(os.path.join(R, f'shaspec_{ds}/*/config.json'))[0])).get('modalities', []))
        ax.set_title(f'{ds} (M={M}' + (', d=128' if ds == 'iemocap' else '') + ')'
                     + ('  [3enc+3fus, α=0]' if row == 0 else ''), fontsize=10)
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('GFLOPs vs Accuracy / Macro-F1 under cross-subject shift (α=0)\n'
             'V6-seqA adaptive-exit curve (per-sample entropy early-exit)  vs  crossattn / multimodn / decalign / shaspec (full-modality points). '
             'DecAlign is 10-100x heavier (M(M-1) cross-attn + OT); the rest sit in seqA\'s GFLOPs range. mean/3 folds.',
             fontsize=11)
# identical subplot geometry across BOTH figures -> same physical length == same
# GFLOPs interval (same xlim [0.04,50] log + same axes box on both figures).
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.085, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_acc_f1_seqA_vs_baselines.png')
fig.savefig(o, dpi=140); print('saved ->', o)   # no bbox=tight -> exact figsize, identical box to the other figure
