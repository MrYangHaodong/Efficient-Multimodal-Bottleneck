#!/usr/bin/env python
"""GFLOPs vs Accuracy / Macro-F1, cross-subject (alpha=0):
  V6-seqA = ADAPTIVE EARLY-EXIT curve (entropy modality early-exit) ;
  crossattn / multimodn / shaspec / decalign = full-modality POINTS.
daliahar + wesad at T=256 (retrained); pamap2 + iemocap at T=128 (existing).
Columns ordered: pamap2, iemocap, wesad, daliahar."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['pamap2', 'iemocap', 'wesad', 'daliahar']
T256 = {'wesad', 'daliahar'}
M_OF = {'daliahar': 5, 'wesad': 10, 'pamap2': 4, 'iemocap': 6}
STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'decalign': ('tab:red', 'D'), 'shaspec': ('tab:brown', '^')}

EXIT128 = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))      # pamap2/iemocap exit
GF128 = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))          # pamap2/iemocap per-K gflops
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json')))  # daliahar/wesad exit + per-K gflops
BGF128 = json.load(open(os.path.join(R, 'baselines_gflops_corrected.json')))
BGF256 = json.load(open(os.path.join(R, 'baselines_gflops_T256.json')))


def seqa_exit(ds, which):  # which: 'acc' or 'f1' -> (gflops_array, y_array)
    M = M_OF[ds]
    if ds in T256:
        ec = np.array(CMP256[ds]['exit_curve'])          # (meanK, acc, f1)
        gpk = CMP256[ds]['gflops_per_k']
        mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    else:
        gpk = GF128[ds]['gflops_per_k']
        mk = np.array(EXIT128[ds]['exit_meanK'])
        y = np.array(EXIT128[ds]['exit_' + which])
    gf = np.interp(mk, np.arange(1, M + 1), gpk)
    o = np.argsort(gf)
    return gf[o], y[o]


def base_gflops(method, ds):
    src = BGF256 if ds in T256 else BGF128
    return src.get(method, {}).get(ds, {}).get('true_gflops')


def base_acc(method, ds, key):
    if ds in T256:
        root = 'T256' if ds == 'daliahar' else 'T256_wesad'
        pat = os.path.join(R, root, method, '*', 'results_fold*.json')
    else:
        pat = os.path.join(R, f'{method}_{ds}', '*', 'results_fold*.json')
    by = {}
    for f in glob.glob(pat):
        mo = f.split('results_fold')[-1].split('.')[0]
        if mo not in by or os.path.getmtime(f) > os.path.getmtime(by[mo]):
            by[mo] = f
    v = [json.load(open(p)).get(key) for p in by.values()]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, ds in enumerate(DS):
    for row, (which, rk, ylab) in enumerate([('acc', 'best_test_acc', 'Accuracy'),
                                             ('f1', 'best_test_f1', 'Macro-F1')]):
        ax = axes[row, col]
        gf, y = seqa_exit(ds, which)
        ax.plot(gf, y, '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
                label='V6-seqA (ours, adaptive early-exit)')
        for method, (color, mk) in STYLE.items():
            g = base_gflops(method, ds); a = base_acc(method, ds, rk)
            if g is not None and a is not None:
                ax.plot([g], [a], mk, color=color, ms=12, zorder=6, label=method)
        ax.set_xscale('log'); ax.set_xlim(0.04, 100)
        tag = 'T=256' if ds in T256 else 'T=128'
        ax.set_title(f'{ds} (M={M_OF[ds]}, {tag})' + ('  [α=0, cross-subject]' if row == 0 else ''), fontsize=10)
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab); ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('GFLOPs vs Accuracy / Macro-F1 under cross-subject shift  —  V6-seqA adaptive early-exit (curve) vs crossattn / multimodn / shaspec / decalign (full-modality points)\n'
             'daliahar & wesad retrained at T=256; pamap2 & iemocap at T=128. DecAlign is 10-100x heavier (M(M-1) cross-attn + OT). mean over 3 folds.',
             fontsize=11)
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.085, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_acc_f1_seqA_vs_baselines_T256mixed.png')
fig.savefig(o, dpi=140); print('saved ->', o)
# quick console summary
for ds in DS:
    gf, ya = seqa_exit(ds, 'acc')
    print(f"{ds:9s}({'T256' if ds in T256 else 'T128'}): seqA-exit acc {ya.min():.3f}-{ya.max():.3f} over GFLOPs {gf.min():.3f}-{gf.max():.3f} | "
          + ' '.join(f"{m}={base_gflops(m,ds):.2f}/{base_acc(m,ds,'best_test_acc'):.3f}" for m in STYLE))
