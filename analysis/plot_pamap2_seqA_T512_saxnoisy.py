#!/usr/bin/env python
"""PAMAP2 (T=512, sax_noisy, cross-subject): V6-seqA ADAPTIVE EARLY-EXIT curve
(from the 2026-06-20 T=512 seqA, val-LOO order) vs crossattn/multimodn/shaspec/
decalign full-modality points (all T=512 sax_noisy). Saved in results_3p3/seqA_pamap2/."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))['pamap2']
BGF = json.load(open(os.path.join(R, 'baselines_gflops_corrected.json')))   # pamap2 = T=512
M = EX['M']; gpk = EX['gflops_per_k']
ec = np.array(EX['exit_curve'])                       # (meanK, acc, f1)
gf_exit = np.interp(ec[:, 0], np.arange(1, M + 1), gpk)


def base_acc(method, key):
    by = {}
    for f in glob.glob(os.path.join(R, f'{method}_pamap2', '*', 'results_fold*.json')):
        k = f.split('results_fold')[-1].split('.')[0]
        if k not in by or os.path.getmtime(f) > os.path.getmtime(by[k]):
            by[k] = f
    v = [json.load(open(p)).get(key) for p in by.values()]
    return float(np.mean([x for x in v if x is not None]))


STYLE = {'crossattn': ('tab:orange', 'o'), 'multimodn': ('tab:purple', 's'),
         'shaspec': ('tab:brown', '^'), 'decalign': ('tab:red', 'D')}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (wi, rk, ylab) in zip(axes, [(1, 'best_test_acc', 'Accuracy'), (2, 'best_test_f1', 'Macro-F1')]):
    ax.plot(gf_exit, ec[:, wi], '-o', color='tab:blue', lw=2.4, ms=4, zorder=7,
            label='V6-seqA (ours, adaptive early-exit)')
    for method, (color, mk) in STYLE.items():
        g = BGF[method]['pamap2']['true_gflops']; a = base_acc(method, rk)
        ax.plot([g], [a], mk, color=color, ms=12, zorder=6, label=method)
    # adaptive modality-axis competitors (DynMM, AdaMML): mean routed acc/f1 @ mean GFLOPs
    for lab, pat, blk, color, mk in [
            ('DynMM', 'dynmm_pamap2/dynmm_seqA/*/results_fold*.json', 'dynmm', 'tab:green', '*'),
            ('AdaMML', 'adamml_pamap2/adamml_ours/results_fold*.json', 'adamml', 'tab:cyan', 'P')]:
        fs = sorted(glob.glob(os.path.join(R, pat)))
        if fs:
            a = np.mean([json.load(open(f))[rk] for f in fs])
            g = np.mean([json.load(open(f))[blk]['mean_gflops'] for f in fs])
            ax.plot([g], [a], mk, color=color, ms=14, zorder=6, label=lab)
    ax.set_xscale('log'); ax.set_xlim(0.3, 20)
    ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.set_title(f'PAMAP2 (M={M}, T=512, sax_noisy, cross-subject)', fontsize=11)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('PAMAP2 T=512 (sax_noisy): V6-seqA adaptive early-exit (curve) vs crossattn/multimodn/shaspec/decalign (full-modality points). mean/3 folds. DecAlign ~12 GFLOPs (M(M-1) cross-attn + OT).', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
o = os.path.join(R, 'seqA_pamap2', 'gflops_T512_saxnoisy_seqA_vs_baselines.png')
fig.savefig(o, dpi=150); print('saved ->', o)
print(f"seqA-exit: acc {ec[:,1].min():.3f}->{ec[:,1].max():.3f} f1 {ec[:,2].min():.3f}->{ec[:,2].max():.3f} over GFLOPs {gf_exit.min():.3f}-{gf_exit.max():.3f}")
for m in STYLE:
    print(f"  {m}: GFLOPs={BGF[m]['pamap2']['true_gflops']:.3f} acc={base_acc(m,'best_test_acc'):.3f} f1={base_acc(m,'best_test_f1'):.3f}")
