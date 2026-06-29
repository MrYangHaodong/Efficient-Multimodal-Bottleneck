#!/usr/bin/env python
"""GFLOPs vs Acc / Macro-F1: V6-seqA ADAPTIVE EARLY-EXIT curve vs
crossattn / multimodn / shaspec / decalign + DyMo-proto, on 5 datasets:
  IEMOCAP (T=128 float) | daliahar (T=256 sax) | PAMAP2 (T=512 sax_noisy) |
  DSADS (T=125 sax) | CMU-MOSEI (T=50 float, binary Acc-2).
Extends plot_3ds_seqA_vs_baselines_perx.py with a MOSEI 5th column."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
MRES = os.path.join(os.path.dirname(R), 'train', 'results_mosei_3seed')   # MOSEI run dir
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']
M_OF = {'iemocap': 6, 'daliahar': 5, 'pamap2': 4, 'dsads': 5, 'mosei': 3}
TAG = {'iemocap': 'T=128, float', 'daliahar': 'T=256, sax', 'pamap2': 'T=512, sax_noisy',
       'dsads': 'T=125, sax', 'mosei': 'T=50, float, Acc-2'}
STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'shaspec': ('tab:brown', '^'), 'decalign': ('tab:red', 'D')}

EXIT128 = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))
GF128 = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json')))
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))
MOEX = json.load(open(os.path.join(R, 'mosei', 'exit_mosei.json')))          # MOSEI exit
BGF128 = json.load(open(os.path.join(R, 'baselines_gflops_corrected.json')))
BGF256 = json.load(open(os.path.join(R, 'baselines_gflops_T256.json')))
BGFDS = json.load(open(os.path.join(R, 'baselines_gflops_dsads.json')))
BGFMO = json.load(open(os.path.join(R, 'baselines_gflops_mosei.json')))      # MOSEI baseline gflops
DYMOG = json.load(open(os.path.join(R, 'dymo_proto_gflops_3ds.json')))

DYMO_RUN = {'iemocap': 'dymo_proto_iemocap/*/results_fold*.json',
            'daliahar': 'T256/dymo_proto/*/results_fold*.json',
            'pamap2': 'dymo_proto_pamap2/*/results_fold*.json',
            'dsads': 'dymo_proto_dsads/*/results_fold*.json'}

# MOSEI run-dir globs (3-seed; crossattn uses the deepcopy-fixed run).
MO_PAT = {'crossattn': '2026-*_CMU_MOSEI_crossattn_fixed_acc2_s*',
          'multimodn': '2026-*_CMU_MOSEI_multimodn_acc2_s*',
          'shaspec':   '2026-*_CMU_MOSEI_shaspec_acc2_s*',
          'decalign':  '2026-*_CMU_MOSEI_decalign_acc2_s*',
          'dymo':      '2026-*_mosei_dymo_acc2_s*'}


def _mo_mean(method, key):
    v = []
    for d in glob.glob(os.path.join(MRES, MO_PAT[method])):
        rj = os.path.join(d, 'results.json')
        if os.path.exists(rj):
            v.append(json.load(open(rj)).get(key))
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def dymo_point(ds, rk):
    if ds == 'mosei':
        return BGFMO['dymo']['mosei']['true_gflops'], _mo_mean('dymo', rk)
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
    elif ds == 'dsads':
        ec = np.array(DSEX['dsads']['exit_curve']); gpk = DSEX['dsads']['gflops_per_k']
        mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    else:  # mosei
        ec = np.array(MOEX['mosei']['exit_curve']); gpk = MOEX['mosei']['gflops_per_k']
        mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    gf = np.interp(mk, np.arange(1, M + 1), gpk); o = np.argsort(gf)
    return gf[o], y[o]


def base_gflops(method, ds):
    if ds == 'mosei':
        return BGFMO[method]['mosei']['true_gflops']
    if ds == 'daliahar':
        return BGF256[method]['daliahar']['true_gflops']
    if ds == 'dsads':
        return BGFDS[method]['dsads']['true_gflops']
    return BGF128[method][ds]['true_gflops']


def base_acc(method, ds, key):
    if ds == 'mosei':
        return _mo_mean(method, key)
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


fig, axes = plt.subplots(2, 5, figsize=(25, 9))
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
        gd, ad = dymo_point(ds, rk)
        if ad is not None:
            ax.plot([gd], [ad], 'X', color='tab:green', ms=13, zorder=6, label='DyMo-proto')
        ax.set_xscale('log')
        sub = '  [cross-subject]' if (row == 0 and ds != 'mosei') else ('  [random split]' if row == 0 else '')
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + sub, fontsize=10)
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab); ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('GFLOPs vs Accuracy / Macro-F1 — V6-seqA adaptive early-exit (curve, val-LOO modality order) vs '
             'crossattn/multimodn/shaspec/decalign (full-modality points) + DyMo-proto. mean/folds(seeds).\n'
             'IEMOCAP T=128 float | daliahar T=256 sax | PAMAP2 T=512 sax_noisy | DSADS T=125 sax (cross-subject) | '
             'CMU-MOSEI T=50 float Acc-2 (random split; seqA=best-2-of-3 seeds, baselines=3-seed; k=1=text-only). Per-panel x-axis (NOT aligned).',
             fontsize=11)
fig.subplots_adjust(left=0.037, right=0.997, bottom=0.08, top=0.89, wspace=0.24, hspace=0.30)
o = os.path.join(R, 'gflops_acc_f1_5ds_seqA_vs_baselines_perx.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    gf, ya = seqa_exit(ds, 'acc')
    print(f"{ds:9s}({TAG[ds]}): seqA-exit acc {ya.min():.3f}-{ya.max():.3f} @ GFLOPs {gf.min():.3f}-{gf.max():.3f} | "
          + ' '.join(f"{m}={base_gflops(m,ds):.2f}/{base_acc(m,ds,'best_test_acc'):.3f}" for m in STYLE))
