#!/usr/bin/env python
"""GFLOPs vs Accuracy / Macro-F1 (cross-subject alpha=0):
  V6-seqA (per-K static curve)  |  faithful DyMo (dymo_proto: FULL -> GREEDY)  |  DynMM (gate@budget).
seqA + DynMM GFLOPs from the 3enc+3fus measurements; dymo_proto GFLOPs from
dymo_proto_gflops.json (FULL forward + the real greedy-search cost)."""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['daliahar', 'pamap2', 'wesad', 'iemocap']
SEQA_ACC = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))
SEQA_GF = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))
BGF = json.load(open(os.path.join(R, 'gflops_3p3_backbone.json')))
DPG = json.load(open(os.path.join(R, 'dymo_proto_gflops.json')))


def folds_mean(method, ds, key):
    by = {}
    for f in glob.glob(os.path.join(R, f'{method}_{ds}/*/results_fold*.json')):
        mo = re.search(r'results_fold(\d+)\.json$', f)
        if mo and (mo.group(1) not in by or os.path.getmtime(f) > os.path.getmtime(by[mo.group(1)])):
            by[mo.group(1)] = f
    vals = [json.load(open(p)).get(key) for p in by.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def dynmm_fold_mean(ds, key):
    vs = []
    for fold in (0, 1, 2):
        p = os.path.join(R, 'shift_sweep_3p3', f'{ds}_a0.0_f{fold}.json')
        if os.path.exists(p):
            vs.append(json.load(open(p))['gate'][key])
    return float(np.mean(vs)) if vs else None


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, ds in enumerate(DS):
    M = BGF[ds]['M']
    for row, (seqa_key, dp_full_k, dp_sel_k, dyn_k, ylab) in enumerate([
            ('static_k_acc', 'best_test_acc', 'select_test_acc', 'acc', 'Accuracy'),
            ('static_k_f1', 'best_test_f1', 'select_test_f1', 'f1', 'Macro-F1')]):
        ax = axes[row, col]
        # --- V6-seqA per-K static curve ---
        sgf = SEQA_GF[ds]['gflops_per_k']; sy = SEQA_ACC[ds][seqa_key]
        ax.plot(sgf[:len(sy)], sy, '-o', color='tab:blue', lw=2.6, ms=7,
                label='V6-seqA (ours, per-K)', zorder=7)
        # --- DynMM gate@budget (shared backbone) ---
        gk = dynmm_fold_mean(ds, 'meanK'); gy = dynmm_fold_mean(ds, dyn_k)
        if gk is not None:
            gx = float(np.interp(gk, np.arange(1, M + 1), BGF[ds]['dynmm_gflops_per_k']))
            ax.plot([gx], [gy], 'D', color='tab:green', ms=12, label='DynMM (gate@budget)', zorder=6)
        # --- faithful DyMo: FULL -> GREEDY (per-sample selection) ---
        gf_full, gf_greedy = DPG[ds]['full_gflops'], DPG[ds]['greedy_gflops']
        y_full = folds_mean('dymo_proto', ds, dp_full_k)
        y_sel = folds_mean('dymo_proto', ds, dp_sel_k)
        ax.plot([gf_full], [y_full], '*', color='tab:purple', ms=17,
                label='DyMo full (all M)', zorder=8)
        ax.plot([gf_greedy], [y_sel], 'X', color='tab:red', ms=12,
                label='DyMo greedy-select', zorder=8)
        ax.annotate('', xy=(gf_greedy, y_sel), xytext=(gf_full, y_full),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.4, ls='--'))
        ax.set_xscale('log'); ax.set_xlim(0.04, 50)   # fixed scale across all panels
        ax.set_title(f'{ds} (M={M}' + (', d=128' if ds == 'iemocap' else '') + ')'
                     + ('  [3enc+3fus, α=0]' if row == 0 else ''), fontsize=10)
        ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=8, loc='lower right')
fig.suptitle('GFLOPs vs Accuracy / Macro-F1 under cross-subject shift (α=0)\n'
             'V6-seqA (per-K static)  vs  faithful DyMo (full → per-sample greedy-select)  vs  DynMM (gate). '
             'DyMo greedy costs MORE GFLOPs than its full forward (encodes all M to search) yet scores lower → Pareto-dominated. mean/3 folds.',
             fontsize=11)
# identical subplot geometry across BOTH figures -> same physical length == same
# GFLOPs interval (same xlim [0.04,50] log + same axes box on both figures).
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.085, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'gflops_acc_f1_seqA_dymoproto_dynmm.png')
fig.savefig(o, dpi=140); print('saved ->', o)   # no bbox=tight -> exact figsize, identical box to the other figure
