#!/usr/bin/env python
"""Latency version of gflops_seqA_vs_dynmm_adamml_4ds.png: x-axis = torch.compile
per-sample latency (ms, batch=1, CUDA-graph compiled, empty GPU) instead of GFLOPs.
seqA = adaptive early-exit curve (acc/f1 vs mean-#modalities mapped to per-K latency);
DynMM/AdaMML = single routed operating points at their usage-weighted compiled latency.
Per-panel x-axis (NOT aligned)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
M_OF = {'iemocap': 6, 'daliahar': 5, 'pamap2': 4, 'dsads': 5}
TAG = {'iemocap': 'T=128, float', 'daliahar': 'T=256, sax', 'pamap2': 'T=512, sax_noisy', 'dsads': 'T=125, sax'}

EXIT128 = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))            # iemocap exit (meanK,acc,f1)
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json'))) # daliahar exit
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))  # pamap2 exit
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))            # dsads exit
LAT = json.load(open(os.path.join(R, 'latency_compile_4ds.json')))            # seqA per-K latency
LATDA = json.load(open(os.path.join(R, 'latency_dynmm_adamml_4ds.json')))      # dynmm/adamml routed latency

DYNMM = {'iemocap': 'dynmm_iemocap/dynmm_seqA/dynmm_seqA_T128/results_fold*.json',
         'daliahar': 'T256/dynmm_seqA/dynmm_seqA_T256/results_fold*.json',
         'pamap2': 'dynmm_pamap2/dynmm_seqA/dynmm_seqA_T512sn/results_fold*.json',
         'dsads': 'dynmm_dsads/dynmm_seqA/*/results_fold*.json'}
ADAMML = {ds: f'adamml_{ds}/adamml_ours/results_fold*.json' for ds in DS}


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


def routed_acc(pat, rk):
    fs = sorted(glob.glob(os.path.join(R, pat)))
    v = [json.load(open(f)).get(rk) for f in fs]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for col, ds in enumerate(DS):
    for row, (which, rk, ylab) in enumerate([('acc', 'best_test_acc', 'Accuracy'), ('f1', 'best_test_f1', 'Macro-F1')]):
        ax = axes[row, col]
        x, y = seqa_exit(ds, which)
        ax.plot(x, y, '-o', color='tab:blue', lw=2.4, ms=4, zorder=7, label='V6-seqA (ours, adaptive early-exit)')
        gd = LATDA[ds]['dynmm']['lat_ms']; ad = routed_acc(DYNMM[ds], rk)
        if ad is not None:
            ax.plot([gd], [ad], '*', color='tab:green', ms=18, zorder=6, label='DynMM')
        ga = LATDA[ds]['adamml']['lat_ms']; aa = routed_acc(ADAMML[ds], rk)
        if aa is not None:
            ax.plot([ga], [aa], 'P', color='tab:cyan', ms=13, zorder=6, label='AdaMML')
        ax.set_xscale('log')   # per-panel x autoscale
        ax.set_xlabel('Latency / sample (ms, torch.compile, log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + ('  [cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=9, loc='lower right')
fig.suptitle('torch.compile per-sample LATENCY vs Accuracy / Macro-F1 (cross-subject) --- V6-seqA adaptive early-exit (curve) vs DynMM / AdaMML (routed operating point).\n'
             'Latency: batch=1, CUDA-graph compiled (reduce-overhead), CUDA-event median, empty GPU. DynMM=gate+usage-weighted experts; AdaMML=policy+usage-weighted encoders. Per-panel x-axis.', fontsize=10.5)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'latency_seqA_vs_dynmm_adamml_4ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    x, ya = seqa_exit(ds, 'acc')
    gd = LATDA[ds]['dynmm']['lat_ms']; ad = routed_acc(DYNMM[ds], 'best_test_acc')
    ga = LATDA[ds]['adamml']['lat_ms']; aa = routed_acc(ADAMML[ds], 'best_test_acc')
    se_d = np.interp(gd, x, ya); se_a = np.interp(ga, x, ya)
    print(f"{ds:9s}: seqA-exit {ya.min():.3f}-{ya.max():.3f} @ {x.min():.2f}-{x.max():.2f}ms | "
          f"DynMM {ad:.3f}@{gd:.2f}ms (seqA@same {se_d:.3f}) | AdaMML {aa:.3f}@{ga:.2f}ms (seqA@same {se_a:.3f})")
