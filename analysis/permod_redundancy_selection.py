#!/usr/bin/env python
"""PER-MODALITY redundancy selection WITHIN each sample (not the per-sample scalar).

For each test sample we compute R[m] = modality m's LSMI redundancy (mean pairwise redundancy
with the other modalities, oracle w.r.t. y_true) — the UN-collapsed [M,N] R from rus(), NOT
R.mean(0). Then each sample orders its OWN modalities by redundancy and we run per-K modality
early-exit (seqA_ps prefix readout) in that order:
   * R-keep-least  : process LEAST-redundant first  (drop redundant modalities)  -> keep top-K unique
   * R-keep-most   : process MOST-redundant first   (opposite direction)
vs the deployed STATIC val-LOO order (same for every sample).

acc/f1 per K -> GFLOPs (gflops_per_k, consensus). Fold-averaged. 5 LSMI datasets (no eav).
Reuses the LSMI machinery in lsmi_exit_seqps_4ds.py. Run from clean_models/results_3p3/.

Out: <REPO>/results/figures/permod_redundancy_selection_5ds.png
     <REPO>/results/tables/permod_redundancy_selection_5ds.json
"""
import sys, os, json
from collections import defaultdict
from itertools import combinations
import numpy as np
sys.argv = ['lsmi', '--ds', 'iemocap']                         # satisfy module-level argparse
import torch, torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, '/files1/haodong/MAESTRO_ttn_robustness_old/clean_models/results_3p3')  # LSMI R/U/S machinery
from lsmi_exit_seqps_4ds import (setup, extract_z, subsets, train_knife, H_of, acc as acc_v,
                                 gflops_per_k, UNITS, dev)

REPO = '/files1/haodong/Efficient-Multimodal-Bottleneck'
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']


def R_permod(zte, sing, labels, knife, mods, pairs):
    """UN-collapsed per-modality redundancy R[M,N] (oracle, y_true)."""
    M = len(mods); N = len(labels); C = sing[mods[0]].shape[1]; logC = float(np.log(C))
    tgt = F.one_hot(torch.tensor(labels), C).float()
    i_m = {m: logC + (sing[m].clamp_min(1e-12).log() * tgt).sum(-1).numpy() for m in mods}
    H = {m: H_of(knife[m], zte[m]) for m in mods}
    I = np.stack([i_m[m] for m in mods]); R = np.zeros((M, N))
    for (a, b) in pairs:
        r = np.minimum(H[mods[a]], H[mods[b]]) - np.minimum(H[mods[a]] - I[a], H[mods[b]] - I[b])
        R[a] += r; R[b] += r
    return R / (M - 1)                                          # [M, N]


def preload(teb, mods):
    X = {m: torch.cat([b[0][m].cpu() for b in teb], 0) for m in mods}
    y = torch.cat([b[2].cpu() for b in teb], 0).numpy()
    return X, y


@torch.no_grad()
def prefix_softmax(model, X, mods, orders, C, sub_bs=256):
    """orders [N,M] canonical modality-idx per sample -> per-K softmax [N,M,C] (prefix readout)."""
    N, M = orders.shape; PREF = np.zeros((N, M, C), np.float32); fuse = model.bottleneck_fusion
    groups = defaultdict(list)
    for i in range(N):
        groups[tuple(int(v) for v in orders[i])].append(i)
    for ordr, idxs in groups.items():
        for s in range(0, len(idxs), sub_bs):
            ch = idxs[s:s + sub_bs]
            h = {m: X[m][ch].to(dev).float() for m in mods}
            fuse.seq_modality_order = list(ordr)
            o = model(h, h, training=False, return_selection_info=False, return_prefix_logits=True)
            pl = o[1] if isinstance(o, tuple) else o
            PREF[ch] = F.softmax(pl, -1).cpu().numpy()
    fuse.seq_modality_order = None
    return PREF


def per_k(PREF, y):
    return [[float((PREF[:, k].argmax(-1) == y).mean()),
             float(f1_score(y, PREF[:, k].argmax(-1), average='macro'))] for k in range(PREF.shape[1])]


RES = {}
for ds in DS:
    print(f'=== {ds} ===')
    fold_static, fold_least, fold_most = [], [], []
    M = gpk = None
    for unit in UNITS[ds]:
        model, (trb, vb, teb), mods, NUMC = setup(ds, unit)
        M = len(mods); pairs = list(combinations(range(M), 2))
        full = acc_v(model, vb, mods)
        loo = {m: full - acc_v(model, vb, [x for x in mods if x != m]) for m in mods}
        vlo = sorted(mods, key=lambda m: -loo[m]); vlo_idx = [mods.index(m) for m in vlo]
        ztr = extract_z(model, trb, mods); zte = extract_z(model, teb, mods)
        knife = train_knife(ztr, mods)
        _, sing, _, labels = subsets(model, teb, mods, pairs)
        R = R_permod(zte, sing, labels, knife, mods, pairs)         # [M,N]
        X, y = preload(teb, mods)
        assert (y == labels).all()
        N = len(y)
        ord_static = np.tile(vlo_idx, (N, 1))
        ord_least = np.argsort(R, axis=0).T                          # ascending R -> least-redundant first
        ord_most = np.argsort(-R, axis=0).T                          # descending R
        fold_static.append(per_k(prefix_softmax(model, X, mods, ord_static, NUMC), y))
        fold_least.append(per_k(prefix_softmax(model, X, mods, ord_least, NUMC), y))
        fold_most.append(per_k(prefix_softmax(model, X, mods, ord_most, NUMC), y))
        if gpk is None:
            gpk = gflops_per_k(model, teb, mods, vlo)
        print(f"  unit {unit}: vlo={vlo}")
        del model; torch.cuda.empty_cache()
    RES[ds] = {'M': M, 'gflops_per_k': list(map(float, gpk)),
               'static_valLOO': np.mean(fold_static, 0).tolist(),
               'R_keep_least': np.mean(fold_least, 0).tolist(),
               'R_keep_most': np.mean(fold_most, 0).tolist()}
    s = np.array(RES[ds]['static_valLOO']); l = np.array(RES[ds]['R_keep_least'])
    print(f"  per-K F1  static={[round(x,3) for x in s[:,1]]}  Rleast={[round(x,3) for x in l[:,1]]}")

json.dump(RES, open(f'{REPO}/results/tables/permod_redundancy_selection_5ds.json', 'w'), indent=2)

fig, axes = plt.subplots(1, 5, figsize=(23, 4.6))
for ax, ds in zip(axes, DS):
    r = RES[ds]; g = np.array(r['gflops_per_k'])
    for key, sty, lab in [('static_valLOO', '-s', 'static val-LOO order'),
                          ('R_keep_least', '-o', 'per-modality R: keep least-redundant'),
                          ('R_keep_most', '--^', 'per-modality R: keep most-redundant')]:
        c = np.array(r[key]); ax.plot(g, c[:, 1], sty, ms=4, lw=2, label=lab)
    ax.set_title(f'{ds}  (M={r["M"]})'); ax.set_xlabel('GFLOPs'); ax.set_ylabel('macro-F1')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower right')
fig.suptitle('Per-sample modality selection by PER-MODALITY redundancy (within each sample) vs static val-LOO — seqA_ps, per-K exit', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = f'{REPO}/results/figures/permod_redundancy_selection_5ds.png'
fig.savefig(png, dpi=140, bbox_inches='tight')
print('saved ->', png)
