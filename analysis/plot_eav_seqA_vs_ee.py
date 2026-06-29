#!/usr/bin/env python
"""GFLOPs vs Acc / Macro-F1 on EAV: seqA-prefixsup adaptive early-exit curve
(modality-axis) vs the 4 efficiency baselines — DynMM (gate), AdaMML (policy),
DMBF (full-modality), MMEE (depth-axis exit curve). 5-class, M=3, T=128,
cross-subject 3-fold mean.
Run from clean_models/train/:  python ../results_3p3/plot_eav_seqA_vs_ee.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(R)
sys.path.insert(0, _CM)
torch.backends.mha.set_fastpath_enabled(False)
from data.EAV.get_data import FEATURE_DIMS, EAVDataset                       # noqa: E402
from models.mmee import MMEEClassifier                     # noqa: E402
from models.dmbf import DMBFClassifier                     # noqa: E402

TRAIN = os.path.join(_CM, 'train')
MODS = list(EAVDataset.ALL_MODALITIES); M = 3; T = 128; NUMC = 5
VAR = {m: FEATURE_DIMS[m] for m in MODS}; C = sum(VAR.values())

# ---- SDPA attention-FLOP patch (FlopCounterMode misses SDPA); same as baselines ----
_A = {'f': 0}; _O = F.scaled_dot_product_attention
def _p(q, k, v, *a, **kw):
    lead = 1
    for s in q.shape[:-2]:
        lead *= s
    _A['f'] += 4 * lead * q.shape[-2] * k.shape[-2] * q.shape[-1]
    return _O(q, k, v, *a, **kw)
def gf(fn):
    _A['f'] = 0; fc = FlopCounterMode(display=False)
    with torch.enable_grad(), fc:
        fn()
    return (fc.get_total_flops() + _A['f']) / 1e9


class _Cfg:
    modalities = MODS; variates = VAR; num_classes = NUMC


# ---- MMEE per-exit-layer GFLOPs (depth axis; all M modalities always processed) ----
F.scaled_dot_product_attention = _p
x = torch.randn(1, T, C)
mm = MMEEClassifier(_Cfg(), T, input_mode='float', d_model=128, nhead=8, num_layers=6).eval()
def mmee_cost(li):
    def fn():
        h = mm._embed(x)
        for layer in mm.layers[:li]:
            h = layer(h)
        return mm.exit_heads[str(li)](h[:, 0])
    return gf(fn)
mmee_gpl = [mmee_cost(e) for e in mm.exit_layers]            # [layer1..6]

# ---- DMBF full GFLOPs (full-modality point) ----
dm = DMBFClassifier(MODS, VAR, NUMC, T, d_model=128, nhead=8, num_layers=2).eval()
dmbf_gf = gf(lambda: dm([torch.randn(1, T, VAR[m]) for m in MODS]))
F.scaled_dot_product_attention = _O

# ---- seqA exit curve ----
EX = json.load(open(os.path.join(R, 'eav', 'exit_eav.json')))['eav']
def seqa_curve(which, key='exit_curve'):
    ec = np.array(EX[key]); gpk = EX['gflops_per_k']
    mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    g = np.interp(mk, np.arange(1, M + 1), gpk); o = np.argsort(g)
    return g[o], y[o]


def mean_folds(glb, ak, fk, gk=None, sub=None, gsub=None):
    a, f, g = [], [], []
    for fp in sorted(glob.glob(glb)):
        r = json.load(open(fp))
        a.append(r.get(ak)); f.append(r.get(fk))
        d = r.get(sub, r) if sub else r
        if gk:
            g.append((r.get(gsub, {}) if gsub else d).get(gk))
    a = [v for v in a if v is not None]; f = [v for v in f if v is not None]; g = [v for v in g if v is not None]
    return (np.mean(a) if a else None, np.mean(f) if f else None, np.mean(g) if g else None)


# DynMM (gate) + AdaMML (policy): self-reported policy-weighted GFLOPs
dyn_a, dyn_f, dyn_g = mean_folds(os.path.join(TRAIN, '../results_3p3/dynmm_eav/dynmm_seqA/dynmm_seqA_eav/results_fold*.json'),
                                 'best_test_acc', 'best_test_f1', 'mean_gflops', gsub='dynmm')
ada_a, ada_f, ada_g = mean_folds(os.path.join(TRAIN, 'results_eav/adamml/*/results_fold*.json'),
                                 'best_test_acc', 'best_test_f1', 'mean_gflops', gsub='adamml')
dmbf_a, dmbf_f, _ = mean_folds(os.path.join(TRAIN, 'results_eav/dmbf/*/results_fold*.json'),
                               'best_test_acc', 'best_test_f1')

# MMEE depth-exit curve (3-fold mean of meanLayer/acc/f1) -> GFLOPs via mmee_gpl
mmee_curves = [json.load(open(fp))['exit_curve_meanLayer_acc_f1']
               for fp in sorted(glob.glob(os.path.join(TRAIN, 'results_eav/mmee/mmee_eav/fold*/results_fold*.json')))]
mc = np.mean(np.array(mmee_curves), axis=0)                  # (taus, 3): meanLayer, acc, f1
mmee_g = np.interp(mc[:, 0], np.arange(1, 7), mmee_gpl)

print(f"MMEE gflops/layer={[round(g,3) for g in mmee_gpl]} | DMBF full={dmbf_gf:.3f}")
print(f"DynMM ({dyn_g:.3f} GF, {dyn_a:.3f}) | AdaMML ({ada_g:.3f} GF, {ada_a:.3f}) | DMBF ({dmbf_gf:.3f} GF, {dmbf_a:.3f})")

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (which, ai, ylab) in zip(axes, [('acc', 1, 'Accuracy (5-class)'), ('f1', 2, 'Macro-F1')]):
    gs, ys = seqa_curve(which)
    ax.plot(gs, ys, '-o', color='tab:blue', lw=2.6, ms=6, zorder=7,
            label='seqA-prefixsup (entropy exit)')
    gp, yp = seqa_curve(which, 'exit_curve_patience')
    ax.plot(gp, yp, '--^', color='tab:cyan', lw=2.0, ms=5, zorder=6,
            label='seqA-prefixsup (patience exit)')
    for k, lbl in [(1, '1 (audio)'), (2, '2'), (3, '3')]:
        gk = EX['gflops_per_k'][k - 1]
        ax.annotate(f'k={lbl}', (gk, np.interp(gk, gs, ys)), textcoords='offset points',
                    xytext=(4, -12), fontsize=8, color='tab:blue')
    # MMEE depth-exit curve
    o = np.argsort(mmee_g)
    ax.plot(mmee_g[o], mc[o, ai], '-s', color='tab:brown', lw=2.0, ms=5, zorder=5,
            label='MMEE (depth early-exit)')
    # points
    da, df = (dyn_a, dyn_f); aa, af = (ada_a, ada_f); ba, bf = (dmbf_a, dmbf_f)
    ax.plot([dyn_g], [da if which == 'acc' else df], 'X', color='tab:green', ms=15, zorder=6, label='DynMM (gate→audio)')
    ax.plot([ada_g], [aa if which == 'acc' else af], 'P', color='tab:red', ms=14, zorder=6, label='AdaMML (policy→audio)')
    ax.plot([dmbf_gf], [ba if which == 'acc' else bf], 'D', color='tab:purple', ms=12, zorder=6, label='DMBF (full-modality)')
    ax.set_xscale('log'); ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, which='both'); ax.set_title(f'{ylab} vs GFLOPs')
    if which == 'acc':
        ax.legend(fontsize=8.5, loc='lower right')
fig.suptitle('EAV (M=3, T=128, 5-class) — seqA-prefixsup modality early-exit vs efficiency baselines '
             '(DynMM, AdaMML, DMBF, MMEE). Cross-subject 3-fold mean.\n'
             'Both DynMM gate AND AdaMML policy COLLAPSE to audio-only (usage=[0,1,0]) — landing on seqA\'s '
             'k=1 point; MMEE depth-exit is flat & costly (all 3 modalities always processed).',
             fontsize=10)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.11, top=0.85, wspace=0.18)
out = os.path.join(R, 'gflops_acc_f1_eav_seqA_vs_ee.png')
fig.savefig(out, dpi=150); print('saved ->', out)
