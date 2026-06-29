#!/usr/bin/env python
"""Latency version: torch.compile per-sample latency vs Acc / Macro-F1 — V6-seqA
adaptive early-exit curve vs DynMM / AdaMML routed points, on 5 datasets (4 +
CMU-MOSEI). Extends plot_latency_seqA_vs_dynmm_adamml.py with a MOSEI 5th column."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
MRES = os.path.join(os.path.dirname(R), 'train', 'results_mosei_3seed')
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']
M_OF = {'iemocap': 6, 'daliahar': 5, 'pamap2': 4, 'dsads': 5, 'mosei': 3}
TAG = {'iemocap': 'T=128, float', 'daliahar': 'T=256, sax', 'pamap2': 'T=512, sax_noisy',
       'dsads': 'T=125, sax', 'mosei': 'T=50, float, Acc-2'}

EXIT128 = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json')))
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))
MOEX = json.load(open(os.path.join(R, 'mosei', 'exit_mosei.json')))
LAT = json.load(open(os.path.join(R, 'latency_compile_4ds.json')))
LATMO = json.load(open(os.path.join(R, 'latency_compile_mosei.json')))
LATDA = json.load(open(os.path.join(R, 'latency_dynmm_adamml_4ds.json')))
LATDA_MO = json.load(open(os.path.join(R, 'latency_dynmm_adamml_mosei.json')))

DYNMM = {'iemocap': 'dynmm_iemocap/dynmm_seqA/dynmm_seqA_T128/results_fold*.json',
         'daliahar': 'T256/dynmm_seqA/dynmm_seqA_T256/results_fold*.json',
         'pamap2': 'dynmm_pamap2/dynmm_seqA/dynmm_seqA_T512sn/results_fold*.json',
         'dsads': 'dynmm_dsads/dynmm_seqA/*/results_fold*.json',
         'mosei': 'dynmm_mosei/dynmm_seqA/*/results_fold*.json'}
ADAMML = {ds: f'adamml_{ds}/adamml_ours/results_fold*.json' for ds in DS[:4]}
ADAMML['mosei'] = os.path.join(MRES, 'adamml_s*', 'results_fold0.json')


def seqa_exit(ds, which):
    M = M_OF[ds]
    if ds == 'iemocap':
        mk = np.array(EXIT128[ds]['exit_meanK']); y = np.array(EXIT128[ds]['exit_' + which])
        lpk = LAT[ds]['seqA_latency_per_k_ms']
    elif ds == 'mosei':
        ec = np.array(MOEX['mosei']['exit_curve']); mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
        lpk = LATMO['mosei']['seqA_latency_per_k_ms']
    else:
        src = {'daliahar': CMP256['daliahar'], 'pamap2': P2EX['pamap2'], 'dsads': DSEX['dsads']}[ds]
        ec = np.array(src['exit_curve']); mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
        lpk = LAT[ds]['seqA_latency_per_k_ms']
    x = np.interp(mk, np.arange(1, M + 1), lpk); o = np.argsort(x)
    return x[o], y[o]


def routed_acc(pat, rk):
    fs = sorted(glob.glob(pat if os.path.isabs(pat) else os.path.join(R, pat)))
    return float(np.mean([json.load(open(f))[rk] for f in fs])) if fs else None


def comp_lat(ds, which):       # which: 'dynmm'/'adamml'
    return (LATDA_MO['mosei'] if ds == 'mosei' else LATDA[ds])[which]['lat_ms']


fig, axes = plt.subplots(2, 5, figsize=(23.5, 8.5))
for col, ds in enumerate(DS):
    for row, (which, rk, ylab) in enumerate([('acc', 'best_test_acc', 'Accuracy'),
                                             ('f1', 'best_test_f1', 'Macro-F1')]):
        ax = axes[row, col]
        x, y = seqa_exit(ds, which)
        ax.plot(x, y, '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
                label='V6-seqA (ours, adaptive early-exit)')
        ad = routed_acc(DYNMM[ds], rk)
        if ad is not None:
            ax.plot([comp_lat(ds, 'dynmm')], [ad], '*', color='tab:green', ms=18, zorder=6, label='DynMM')
        aa = routed_acc(ADAMML[ds], rk)
        if aa is not None:
            ax.plot([comp_lat(ds, 'adamml')], [aa], 'P', color='tab:cyan', ms=13, zorder=6, label='AdaMML')
        ax.set_xscale('log')
        ax.set_xlabel('Latency / sample (ms, torch.compile, log)'); ax.set_ylabel(ylab)
        sub = '  [cross-subject]' if (row == 0 and ds != 'mosei') else ('  [random split]' if row == 0 else '')
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + sub, fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=9, loc='lower right')
fig.suptitle('torch.compile LATENCY head-to-head: V6-seqA adaptive early-exit (curve) vs DynMM / AdaMML (routed point). '
             'CUDA-graph compiled, batch=1, CUDA-event median.\n'
             'IEMOCAP/daliahar/pamap2/dsads (cross-subject) | CMU-MOSEI T=50 Acc-2 (random split, GloVe; '
             'DynMM->text expert, AdaMML->text-only).', fontsize=10.5)
fig.subplots_adjust(left=0.037, right=0.997, bottom=0.08, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'latency_seqA_vs_dynmm_adamml_5ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    x, ya = seqa_exit(ds, 'acc')
    print(f"{ds:9s}: seqA-exit lat {x.min():.2f}-{x.max():.2f}ms | "
          f"DynMM {routed_acc(DYNMM[ds],'best_test_acc'):.3f}@{comp_lat(ds,'dynmm'):.2f}ms | "
          f"AdaMML {routed_acc(ADAMML[ds],'best_test_acc'):.3f}@{comp_lat(ds,'adamml'):.2f}ms")
