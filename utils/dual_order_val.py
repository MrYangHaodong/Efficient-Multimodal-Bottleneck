"""Dual-order validation for sequential-fusion seqA models.

The seqA bottleneck fusion processes modalities one-at-a-time, so a full-modality
prediction depends on the PROCESSING ORDER. This helper scores the val set two ways:

  * method A (val-LOO order): compute per-modality leave-one-out importance on val
    (canonical order), sort strong-first, then eval full-modality IN THAT ORDER.
  * method B (random-order avg): eval full-modality under K random orders, average.

It operates on the INNER DualVideoBottleneckModelV6Downsample (i.e. wrapper.model),
toggling ``inner.bottleneck_fusion.seq_modality_order`` per the chosen order. Batches
are a preloaded list of (high_dict, low_dict, y) on CPU; tensors are moved per-batch.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


@torch.no_grad()
def _eval(inner, batches, mods, order_idx, dev, acc_only=False):
    inner.bottleneck_fusion.seq_modality_order = list(order_idx)
    preds, ys = [], []
    for high, low, y in batches:
        h = {m: high[m].to(dev, non_blocking=True).float() for m in mods}
        l = {m: low[m].to(dev, non_blocking=True).float() for m in mods}
        o = inner(h, l, training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        preds.append(o.argmax(-1).cpu()); ys.append(y)
    p = torch.cat(preds).numpy(); t = torch.cat(ys).numpy()
    acc = float((p == t).mean())
    if acc_only:
        return acc
    return acc, float(f1_score(t, p, average='macro'))


def loo_order(inner, batches, MODS, dev):
    """val leave-one-out (canonical order) -> strong-first modality order (names)."""
    M = len(MODS)
    full = _eval(inner, batches, MODS, list(range(M)), dev, acc_only=True)
    loo = {}
    for j, m in enumerate(MODS):
        keep = [x for x in MODS if x != m]
        loo[m] = full - _eval(inner, batches, keep, list(range(M - 1)), dev, acc_only=True)
    return sorted(MODS, key=lambda m: -loo[m]), loo


def dual_val(inner, batches, MODS, dev, K=5, seed=1234):
    """Returns dict with method-A (val-LOO order) and method-B (random-avg) val acc/f1.
    Restores seq_modality_order=None afterwards."""
    M = len(MODS)
    orderA, loo = loo_order(inner, batches, MODS, dev)
    accA, f1A = _eval(inner, batches, MODS, [MODS.index(m) for m in orderA], dev)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(K):
        pi = list(rng.permutation(M))
        a, f = _eval(inner, batches, MODS, pi, dev)
        accs.append(a); f1s.append(f)
    inner.bottleneck_fusion.seq_modality_order = None
    return {'accA': accA, 'f1A': f1A, 'orderA': orderA,
            'accB': float(np.mean(accs)), 'f1B': float(np.mean(f1s)),
            'loo': loo}


def rand_avg_eval(inner, batches, MODS, dev, K=5, seed=4321):
    """Full-modality eval averaged over K random orders. Returns (acc, f1)."""
    M = len(MODS); rng = np.random.default_rng(seed); accs, f1s = [], []
    for _ in range(K):
        a, f = _eval(inner, batches, MODS, list(rng.permutation(M)), dev)
        accs.append(a); f1s.append(f)
    inner.bottleneck_fusion.seq_modality_order = None
    return float(np.mean(accs)), float(np.mean(f1s))


def eval_full_order(inner, batches, MODS, dev, order_names=None):
    """Full-modality test eval in a given order (names); canonical if order_names None.
    Returns (acc, f1). Restores seq_modality_order=None afterwards."""
    M = len(MODS)
    oi = list(range(M)) if order_names is None else [MODS.index(m) for m in order_names]
    acc, f1 = _eval(inner, batches, MODS, oi, dev)
    inner.bottleneck_fusion.seq_modality_order = None
    return acc, f1
