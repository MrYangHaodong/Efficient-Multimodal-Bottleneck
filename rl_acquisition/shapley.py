"""Modality Shapley values over a cached prefix-tree.

Used in two places:
  * dqn_policy_rlv2       -> the "advice" state features (expected gain of each missing modality)
  * baselines.py          -> the STATIC Shapley baseline (fixed global order, fixed top-k)

⚠ ALWAYS COMPUTE ON **val**, NEVER ON **train**.
The frozen backbones memorise: on IEMOCAP the video encoder reaches 0.9989 TRAIN accuracy
but 0.3124 TEST. Train-Shapley therefore ranks whichever modality memorises hardest --
it puts `video` FIRST on IEMOCAP and MELD, both of which are at chance on test. Measured
cost of using train instead of val: -0.097 macro-F1 (IEMOCAP), -0.101 (CMI), -0.074 (MELD).
`split='train'` is exposed only so that failure can be reported as an ablation.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score


def set_index(cache):
    """frozenset(modalities) -> [node ids], i.e. every cached ordering of that set."""
    g = defaultdict(list)
    for i, k in enumerate(cache['nodes']):
        s = frozenset() if k == '' else frozenset(int(x) for x in k.split(','))
        g[s].append(i)
    return g


def set_value(cache, groups, S, idx=None, metric='macro_f1'):
    """v(S) = value of the frozen model given modality set S, averaged over all cached
    orderings of S (the order-invariant readout). None if S isn't in the depth-capped tree.

    metric='macro_f1' (default) matches what we REPORT and model-select on. Ranking
    modalities by their accuracy contribution while scoring macro-F1 would optimise a
    different objective than the one being measured -- and the two diverge on imbalanced
    label sets (CMI: 18 classes, 4x imbalance).
    """
    ids = groups.get(frozenset(S))
    if not ids:
        return None
    y = cache['y'] if idx is None else cache['y'][idx]
    sm = cache['softmax'] if idx is None else cache['softmax'][:, idx, :]
    if metric == 'accuracy':
        return float(np.mean([(sm[i].argmax(-1) == y).mean() for i in ids]))
    return float(np.mean([f1_score(y, sm[i].argmax(-1), average='macro') for i in ids]))


def shapley_values(cache, M, max_depth, idx=None, metric='macro_f1'):
    """Per-modality Shapley value of MACRO-F1 (the reported metric) -> phi[M].

    Truncated to coalitions |S| <= max_depth-1 (the tree is depth-capped), averaging the
    marginal contribution uniformly over coalition sizes -- the standard truncated estimator.

    idx: optional sample subset (e.g. an inner-train split) so advice stays out-of-sample.
    """
    groups = set_index(cache)
    phi = np.zeros(M)
    for m in range(M):
        others = [x for x in range(M) if x != m]
        per_size = []
        for s in range(0, max_depth):
            deltas = []
            for S in combinations(others, s):
                a = set_value(cache, groups, set(S) | {m}, idx, metric)
                b = set_value(cache, groups, set(S), idx, metric)
                if a is not None and b is not None:
                    deltas.append(a - b)
            if deltas:
                per_size.append(float(np.mean(deltas)))
        phi[m] = float(np.mean(per_size)) if per_size else 0.0
    return phi


def shapley_order(cache, M, max_depth, idx=None, metric='macro_f1'):
    """Global static acquisition order: modality indices by descending Shapley value."""
    return list(np.argsort(-shapley_values(cache, M, max_depth, idx, metric)))


def advice_vector(cache, M, max_depth, idx=None, metric='macro_f1'):
    """RL_v2's `advice` features: a length-M vector of per-modality Shapley values,
    min-max normalised to [0,1] so it is on the same scale as the softmax features.
    Constant across nodes; the policy reads it alongside the acquisition mask, so it acts
    as a data-driven prior over which modality is worth taking next."""
    phi = shapley_values(cache, M, max_depth, idx, metric)
    lo, hi = float(phi.min()), float(phi.max())
    return np.full(M, 0.5, np.float32) if hi - lo < 1e-9 else ((phi - lo) / (hi - lo)).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  ADVICE — the RL_v2 mechanism, restored (see dqn_policy_rlv2).
#
#  Pipeline:
#      phi[N, M]  per-sample Shapley of correctness        <- USES LABELS
#      vv[C, M]   = mean of phi over samples of each class <- compresses the label dependence
#      advice[n, N, M] = softmax[node] @ vv, masked        <- NO label at test time
#
#  The label is used ONCE, on val, at DESIGN time, to build `vv`. At deploy, `vv` is a fixed
#  table and `softmax[node]` is observable, so `advice` asks: "given what I currently believe
#  about the class, which modality helps?" That is sample-conditional VIA THE POSTERIOR and
#  fully deployable. It is not leakage.
# ═════════════════════════════════════════════════════════════════════════════
def shapley_per_sample(cache, M, max_depth, metric='correct'):
    """phi[N, M] — each sample's own Shapley value per modality. USES LABELS: val/design only.

    v(S) for sample j = 1[the model is right about j given set S], averaged over cached
    orderings of S. Truncated to |S| <= max_depth-1, uniform over coalition sizes.
    """
    groups = set_index(cache)
    y = cache['y']; N = len(y); soft = cache['softmax']

    def v(S):
        ids = groups.get(frozenset(S))
        if not ids:
            return None
        return np.mean([(soft[i].argmax(-1) == y).astype(np.float64) for i in ids], axis=0)  # [N]

    phi = np.zeros((N, M))
    for m in range(M):
        others = [x for x in range(M) if x != m]
        per_size = []
        for s in range(0, max_depth):
            deltas = []
            for S in combinations(others, s):
                a, b = v(set(S) | {m}), v(set(S))
                if a is not None and b is not None:
                    deltas.append(a - b)
            if deltas:
                per_size.append(np.mean(deltas, axis=0))
        if per_size:
            phi[:, m] = np.mean(per_size, axis=0)
    return phi


def value_val_perclass(phi, y, M, C):
    """vv[c, m] = mean per-sample Shapley of modality m over the val samples of class c."""
    vv = np.zeros((C, M), np.float32)
    for c in range(C):
        sel = (y == c)
        if sel.any():
            vv[c] = phi[sel].mean(0)
    return vv


def node_advice(cache, M, max_depth, vv, avail=None):
    """advice[n, N, M] = softmax[node] @ vv, ZEROED for modalities already acquired (and, if
    `avail` is given, for modalities that do not exist for that sample).

    Varies per NODE (different acquired sets) and per SAMPLE (different posterior, different
    availability) — that is the whole point. A constant advice vector is a dead input.
    """
    from collections import defaultdict as _dd
    nodes = cache['nodes']; soft = cache['softmax']; n, N, C = soft.shape
    used = [frozenset() if k == '' else frozenset(int(x) for x in k.split(',')) for k in nodes]
    adv = np.zeros((n, N, M), np.float32)
    for i in range(n):
        a = soft[i] @ vv                       # [N, M] expected value under the current belief
        for m in used[i]:
            a[:, m] = 0.0                      # already acquired -> no advice
        adv[i] = a
    if avail is not None:
        adv *= np.asarray(avail, np.float32)[None]   # unavailable -> no advice
    return adv


def static_advice(cache, M, max_depth, phi_pop, avail=None):
    """advice[n, N, M] = the STATIC population Shapley value per modality, masked the same way.

    Same masking as node_advice, but the value does NOT depend on the posterior — only on which
    modalities remain. This is the "static Shapley ordering" arm: it isolates how much of
    advice's value comes from being sample-conditional vs just being a good global ranking.
    """
    nodes = cache['nodes']; soft = cache['softmax']; n, N, C = soft.shape
    used = [frozenset() if k == '' else frozenset(int(x) for x in k.split(',')) for k in nodes]
    adv = np.zeros((n, N, M), np.float32)
    base = np.asarray(phi_pop, np.float32)
    for i in range(n):
        a = np.tile(base, (N, 1))
        for m in used[i]:
            a[:, m] = 0.0
        adv[i] = a
    if avail is not None:
        adv *= np.asarray(avail, np.float32)[None]
    return adv


# ═════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL advice — the value of m GIVEN what you already hold.
#
#  The MARGINAL Shapley (above) averages over ALL coalitions, including those WITHOUT the
#  modalities you already have. So `audio` keeps its +0.179 even at the node where you already
#  hold `text` and audio is largely redundant. Measured on IEMOCAP fold 0 (val):
#
#      S = {}            audio +0.303          S = {text}         audio +0.059   (3x smaller)
#      S = {text,audio}  mocap_rot -0.013, video -0.060           (adding MORE hurts)
#
#  The conditional gain  d(S,m) = v(S+m) - v(S)  is a direct cache lookup, and it is what a
#  one-step Q prior actually needs. Same design-time trick as node_advice: compute per-sample on
#  val (labels), compress per class, apply via the posterior at test -> deployable.
# ═════════════════════════════════════════════════════════════════════════════
def conditional_tables(cache, M, max_depth):
    """{frozenset(S): [C, M] table} of per-class conditional gains d(S,m) = v(S+m) - v(S).

    USES LABELS -> compute on the fit split (val) only. Size: 2^M x C x M
    (IEMOCAP 64x4x6 = 1,536 numbers; CMI 128x18x7 = 16k). Cheap.
    """
    groups = set_index(cache)
    y = cache['y']; C = cache['softmax'].shape[-1]; soft = cache['softmax']

    def v(S):
        ids = groups.get(frozenset(S))
        if not ids:
            return None
        return np.mean([(soft[i].argmax(-1) == y).astype(np.float64) for i in ids], axis=0)  # [N]

    out = {}
    for S in groups:
        if len(S) >= max_depth:
            continue
        base = v(S)
        if base is None:
            continue
        tab = np.zeros((C, M), np.float32)
        for m in range(M):
            if m in S:
                continue                       # already held -> no gain
            a = v(set(S) | {m})
            if a is None:
                continue
            d = a - base                       # [N] per-sample conditional gain
            for c in range(C):
                sel = (y == c)
                if sel.any():
                    tab[c, m] = d[sel].mean()  # compress the label dependence per class
        out[frozenset(S)] = tab
    return out


def conditional_advice(cache, M, max_depth, tabs, avail=None):
    """advice[n, N, M] = softmax[node] @ tabs[S_node], masked to unacquired ∩ available.

    Unlike node_advice, the [C,M] table is SELECTED BY THE NODE'S SET, so the value of m is
    conditioned on what has already been acquired — which is exactly what stops the prior from
    over-valuing a redundant modality.
    """
    nodes = cache['nodes']; soft = cache['softmax']; n, N, C = soft.shape
    used = [frozenset() if k == '' else frozenset(int(x) for x in k.split(',')) for k in nodes]
    adv = np.zeros((n, N, M), np.float32)
    fallback = np.zeros((C, M), np.float32)
    for i in range(n):
        tab = tabs.get(used[i])
        if tab is None:                        # at max depth: nothing left to gain
            continue
        a = soft[i] @ tab                      # [N, M] expected conditional gain under the belief
        for m in used[i]:
            a[:, m] = 0.0
        adv[i] = a
    if avail is not None:
        adv *= np.asarray(avail, np.float32)[None]
    return adv
