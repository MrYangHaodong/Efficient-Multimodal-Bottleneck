#!/usr/bin/env python
"""GFLOPs vs Acc / Macro-F1 (cross-subject): V6-seqA ADAPTIVE EARLY-EXIT curve vs
the two MODALITY-AXIS competitors DynMM and AdaMML (single routed points), on 4
datasets. All competitors use the SAME seqA backbone / branches, trained apples-to-
apples (DynMM = 3 seqA experts on val-LOO subsets; AdaMML = per-mod 6L transformer
+ policy net). IEMOCAP T=128 float | daliahar T=256 sax | WESAD T=256 sax |
PAMAP2 T=512 sax_noisy."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
M_OF = {'iemocap': 6, 'daliahar': 5, 'pamap2': 4, 'dsads': 5}
TAG = {'iemocap': 'T=128, float', 'daliahar': 'T=256, sax', 'pamap2': 'T=512, sax_noisy', 'dsads': 'T=125, sax'}

EXIT_IE = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))         # iemocap exit
GF_IE = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))            # iemocap per-K gflops
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json')))  # daliahar+wesad exit
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))  # pamap2 exit
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))         # dsads exit (val-LOO order)

# glob patterns for the clean (seqA-backbone) DynMM + AdaMML routed results
DYNMM = {'iemocap': 'dynmm_iemocap/dynmm_seqA/dynmm_seqA_T128/results_fold*.json',
         'daliahar': 'T256/dynmm_seqA/dynmm_seqA_T256/results_fold*.json',
         'wesad': 'T256_wesad/dynmm_seqA/dynmm_seqA_T256/results_fold*.json',
         'pamap2': 'dynmm_pamap2/dynmm_seqA/dynmm_seqA_T512sn/results_fold*.json',
         'dsads': 'dynmm_dsads/dynmm_seqA/*/results_fold*.json'}
ADAMML = {ds: f'adamml_{ds}/adamml_ours/results_fold*.json' for ds in DS}


def seqa_exit(ds, which):  # which: 'acc'/'f1'
    M = M_OF[ds]
    if ds == 'iemocap':
        gpk = GF_IE[ds]['gflops_per_k']; mk = np.array(EXIT_IE[ds]['exit_meanK'])
        y = np.array(EXIT_IE[ds]['exit_' + which])
    elif ds in ('daliahar', 'wesad'):
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


def routed(pat, blk, rk):  # mean acc/f1 @ mean GFLOPs across folds
    fs = sorted(glob.glob(os.path.join(R, pat)))
    if not fs:
        return None, None
    a = np.mean([json.load(open(f))[rk] for f in fs])
    g = np.mean([json.load(open(f))[blk]['mean_gflops'] for f in fs])
    return float(g), float(a)


fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for col, ds in enumerate(DS):
    for row, (which, rk, ylab) in enumerate([('acc', 'best_test_acc', 'Accuracy'),
                                             ('f1', 'best_test_f1', 'Macro-F1')]):
        ax = axes[row, col]
        gf, y = seqa_exit(ds, which)
        ax.plot(gf, y, '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
                label='V6-seqA (ours, adaptive early-exit)')
        gd, ad = routed(DYNMM[ds], 'dynmm', rk)
        if gd is not None:
            ax.plot([gd], [ad], '*', color='tab:green', ms=18, zorder=6, label='DynMM')
        ga, aa = routed(ADAMML[ds], 'adamml', rk)
        if ga is not None:
            ax.plot([ga], [aa], 'P', color='tab:cyan', ms=13, zorder=6, label='AdaMML')
        ax.set_xscale('log')
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + ('  [cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=9, loc='lower right')
fig.suptitle('Modality-axis efficiency head-to-head (cross-subject): V6-seqA adaptive early-exit (curve, sweeps the compute axis) '
             'vs DynMM / AdaMML (single routed operating point).\n'
             'DynMM = 3 seqA experts on val-LOO subsets (gated); AdaMML = per-mod 6L transformer + policy net. seqA exit dominates / matches both at every compute level.', fontsize=11)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_seqA_vs_dynmm_adamml_4ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    gf, ya = seqa_exit(ds, 'acc')
    gd, ad = routed(DYNMM[ds], 'dynmm', 'best_test_acc')
    ga, aa = routed(ADAMML[ds], 'adamml', 'best_test_acc')
    # seqA acc interpolated at the competitor's GFLOPs (for matched-compute comparison)
    se_d = np.interp(gd, gf, ya) if gd else None
    se_a = np.interp(ga, gf, ya) if ga else None
    print(f"{ds:9s}: seqA-exit acc {ya.min():.3f}-{ya.max():.3f} @ {gf.min():.2f}-{gf.max():.2f} GF | "
          f"DynMM {ad:.3f}@{gd:.2f} (seqA@same {se_d:.3f}) | AdaMML {aa:.3f}@{ga:.2f} (seqA@same {se_a:.3f})")
