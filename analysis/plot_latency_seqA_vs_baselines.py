#!/usr/bin/env python
"""Same comparison as gflops_acc_f1_seqA_vs_baselines_perx.png but the x-axis is
torch.compile per-sample LATENCY (ms, batch=1, CUDA-graph compiled) instead of GFLOPs.
seqA = adaptive early-exit curve (acc/f1 vs mean-#modalities mapped to per-K latency);
baselines = full-modality single points. Per-panel x-axis (NOT aligned)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
M_OF = {'iemocap': 6, 'daliahar': 5, 'pamap2': 4, 'dsads': 5}
TAG = {'iemocap': 'T=128, float', 'daliahar': 'T=256, sax', 'pamap2': 'T=512, sax_noisy', 'dsads': 'T=125, sax'}
STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'shaspec': ('tab:brown', '^'), 'decalign': ('tab:red', 'D')}

EXIT128 = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))            # iemocap exit (meanK,acc,f1)
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json'))) # daliahar exit
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))  # pamap2 exit
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))            # dsads exit
LAT = json.load(open(os.path.join(R, 'latency_compile_4ds.json')))            # torch.compile latency


def seqa_exit(ds, which):
    M = M_OF[ds]
    if ds == 'iemocap':
        mk = np.array(EXIT128[ds]['exit_meanK']); y = np.array(EXIT128[ds]['exit_' + which])
    else:
        src = {'daliahar': CMP256['daliahar'], 'pamap2': P2EX['pamap2'], 'dsads': DSEX['dsads']}[ds]
        ec = np.array(src['exit_curve']); mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    lpk = LAT[ds]['seqA_latency_per_k_ms']
    x = np.interp(mk, np.arange(1, M + 1), lpk); o = np.argsort(x)
    return x[o], y[o]


def base_lat(method, ds):
    v = LAT[ds].get(method, {})
    return v.get('latency_ms')


def base_acc(method, ds, key):
    pat = (os.path.join(R, 'T256', method, '*', 'results_fold*.json') if ds == 'daliahar'
           else os.path.join(R, 'dsads', method, '*', 'results_fold*.json') if ds == 'dsads'
           else os.path.join(R, f'{method}_{ds}', '*', 'results_fold*.json'))
    by = {}
    for f in glob.glob(pat):
        k = f.split('results_fold')[-1].split('.')[0]
        if k not in by or os.path.getmtime(f) > os.path.getmtime(by[k]):
            by[k] = f
    v = [json.load(open(p)).get(key) for p in by.values()]
    return float(np.mean([x for x in v if x is not None])) if v else None


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, ds in enumerate(DS):
    for row, (which, rk, ylab) in enumerate([('acc', 'best_test_acc', 'Accuracy'),
                                             ('f1', 'best_test_f1', 'Macro-F1')]):
        ax = axes[row, col]
        x, y = seqa_exit(ds, which)
        ax.plot(x, y, '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
                label='V6-seqA (ours, adaptive early-exit)')
        for method, (color, mk) in STYLE.items():
            g = base_lat(method, ds); a = base_acc(method, ds, rk)
            if g is not None and a is not None:
                ax.plot([g], [a], mk, color=color, ms=12, zorder=6, label=method)
        ax.set_xscale('log')   # per-panel x autoscale
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + ('  [cross-subject]' if row == 0 else ''), fontsize=10)
        ax.set_xlabel('Latency / sample (ms, torch.compile, log)'); ax.set_ylabel(ylab); ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('torch.compile per-sample LATENCY vs Accuracy / Macro-F1 (cross-subject) — V6-seqA adaptive early-exit (curve) vs crossattn/multimodn/shaspec/decalign (full-modality points). mean/folds.\n'
             'Latency = batch=1, CUDA-graph compiled (reduce-overhead), CUDA-event median. Per-panel x-axis (NOT aligned). IEMOCAP T=128 | daliahar T=256 | PAMAP2 T=512 | DSADS T=125.', fontsize=11)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.89, wspace=0.22, hspace=0.30)
o = os.path.join(R, 'latency_acc_f1_seqA_vs_baselines_perx.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    x, ya = seqa_exit(ds, 'acc')
    print(f"{ds:9s}: seqA-exit lat {x.min():.2f}-{x.max():.2f}ms acc {ya.min():.3f}-{ya.max():.3f} | "
          + ' '.join(f"{m}={base_lat(m,ds):.2f}ms/{base_acc(m,ds,'best_test_acc'):.3f}" for m in STYLE if base_lat(m, ds)))
