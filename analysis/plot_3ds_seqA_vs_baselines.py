#!/usr/bin/env python
"""GFLOPs vs Acc / Macro-F1 (cross-subject): V6-seqA ADAPTIVE EARLY-EXIT curve vs
crossattn / multimodn / shaspec / decalign full-modality points, on 4 datasets:
  IEMOCAP (T=128, float) | daliahar (T=256, sax) | PAMAP2 (T=512, sax_noisy) | DSADS (T=125, sax)."""
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

EXIT128 = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))            # iemocap exit
GF128 = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))                # iemocap per-K gflops
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json'))) # daliahar exit+gflops
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))  # pamap2 T512 exit+gflops
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))            # dsads T125 exit+gflops (val-LOO order)
BGF128 = json.load(open(os.path.join(R, 'baselines_gflops_corrected.json')))   # iemocap+pamap2 (T512)
BGF256 = json.load(open(os.path.join(R, 'baselines_gflops_T256.json')))        # daliahar T256
BGFDS = json.load(open(os.path.join(R, 'baselines_gflops_dsads.json')))        # dsads T125
DYMOG = json.load(open(os.path.join(R, 'dymo_proto_gflops_3ds.json')))         # dymo-proto greedy gflops @ plot-T

# DyMo-proto routed operating point: select_test_* (greedy selection) @ greedy_gflops.
DYMO_RUN = {'iemocap': 'dymo_proto_iemocap/*/results_fold*.json',
            'daliahar': 'T256/dymo_proto/*/results_fold*.json',
            'pamap2': 'dymo_proto_pamap2/*/results_fold*.json',
            'dsads': 'dymo_proto_dsads/*/results_fold*.json'}


def dymo_point(ds, rk):  # rk: 'best_test_acc'/'best_test_f1' -> map to select_test_*
    sk = rk.replace('best_test', 'select_test')
    fs = sorted(glob.glob(os.path.join(R, DYMO_RUN[ds])))
    v = [json.load(open(f)).get(sk) for f in fs]
    v = [x for x in v if x is not None]
    return DYMOG[ds]['greedy_gflops'], (float(np.mean(v)) if v else None)


def seqa_exit(ds, which):
    M = M_OF[ds]
    if ds == 'iemocap':
        gpk = GF128[ds]['gflops_per_k']; mk = np.array(EXIT128[ds]['exit_meanK'])
        y = np.array(EXIT128[ds]['exit_' + which])
    elif ds == 'daliahar':
        ec = np.array(CMP256[ds]['exit_curve']); gpk = CMP256[ds]['gflops_per_k']
        mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    elif ds == 'pamap2':
        ec = np.array(P2EX['pamap2']['exit_curve']); gpk = P2EX['pamap2']['gflops_per_k']
        mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    else:  # dsads (val-LOO modality order)
        ec = np.array(DSEX['dsads']['exit_curve']); gpk = DSEX['dsads']['gflops_per_k']
        mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    gf = np.interp(mk, np.arange(1, M + 1), gpk); o = np.argsort(gf)
    return gf[o], y[o]


def base_gflops(method, ds):
    if ds == 'daliahar':
        return BGF256[method]['daliahar']['true_gflops']
    if ds == 'dsads':
        return BGFDS[method]['dsads']['true_gflops']
    return BGF128[method][ds]['true_gflops']


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
        gf, y = seqa_exit(ds, which)
        ax.plot(gf, y, '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
                label='V6-seqA (ours, adaptive early-exit)')
        for method, (color, mk) in STYLE.items():
            g = base_gflops(method, ds); a = base_acc(method, ds, rk)
            if g is not None and a is not None:
                ax.plot([g], [a], mk, color=color, ms=12, zorder=6, label=method)
        gd, ad = dymo_point(ds, rk)            # DyMo-proto (per-sample greedy modality selection)
        if ad is not None:
            ax.plot([gd], [ad], 'X', color='tab:green', ms=13, zorder=6, label='DyMo-proto')
        ax.set_xscale('log'); ax.set_xlim(0.1, 40)
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + ('  [cross-subject]' if row == 0 else ''), fontsize=10)
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab); ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('GFLOPs vs Accuracy / Macro-F1 (cross-subject) — V6-seqA adaptive early-exit (curve, val-LOO modality order) vs crossattn/multimodn/shaspec/decalign (full-modality points) + DyMo-proto (greedy select). mean/folds.\n'
             'IEMOCAP T=128 float | daliahar T=256 sax | PAMAP2 T=512 sax_noisy | DSADS T=125 sax (4 scenarios). DecAlign is 10-30x heavier (M(M-1) cross-attn + OT).', fontsize=11)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.89, wspace=0.22, hspace=0.30)
o = os.path.join(R, 'gflops_acc_f1_seqA_vs_baselines_3ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    gf, ya = seqa_exit(ds, 'acc')
    print(f"{ds:9s}({TAG[ds]}): seqA-exit acc {ya.min():.3f}-{ya.max():.3f} @ GFLOPs {gf.min():.3f}-{gf.max():.3f} | "
          + ' '.join(f"{m}={base_gflops(m,ds):.2f}/{base_acc(m,ds,'best_test_acc'):.3f}" for m in STYLE))
