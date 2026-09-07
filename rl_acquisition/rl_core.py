"""Shared RL primitives used by all three policy scripts (dqn_policy_*.py).
Pure numpy/torch; no repo dependency. See environment.py for the data/model side."""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ITERS, EPOCHS, LR = 8, 40, 1e-3          # Fitted-Q Iteration protocol


# ─────────────────────────────────────────────────────────────────────────────
# Prefix-tree bookkeeping. A `cache` node is an ordered set of acquired modalities,
# encoded as a comma-joined string of modality indices ('' = root, nothing acquired).
# ─────────────────────────────────────────────────────────────────────────────
def node_meta(nodes, M, max_depth):
    """Returns depth[n], used[n] (frozenset of acquired modality idxs), and
    ch[n] = list of (modality m, child-node-id) reachable by acquiring m."""
    idx = {k: i for i, k in enumerate(nodes)}
    depth = np.array([0 if k == '' else k.count(',') + 1 for k in nodes])
    used = [frozenset() if k == '' else frozenset(int(x) for x in k.split(',')) for k in nodes]
    ch = {}
    for i, k in enumerate(nodes):
        if depth[i] >= max_depth:
            ch[i] = []; continue
        u = [] if k == '' else [int(x) for x in k.split(',')]
        ch[i] = [(m, idx[','.join(map(str, u + [m]))]) for m in range(M)
                 if m not in u and ','.join(map(str, u + [m])) in idx]
    return depth, used, ch


def action_mask(n, A, depth, ch, M):
    """Legal actions per node: acquire m if a child exists; STOP if not the root."""
    mask = np.zeros((n, A), bool)
    for i in range(n):
        for (m, _c) in ch[i]:
            mask[i, m] = True
        if depth[i] > 0:
            mask[i, M] = True
    return mask


def child_table(ch, n, M):
    """child_of[node, m] = child node id when acquiring m (or -1 if illegal)."""
    child_of = np.full((n, M), -1, np.int64)
    for i in range(n):
        for (m, c) in ch[i]:
            child_of[i, m] = c
    return child_of


# ═════════════════════════════════════════════════════════════════════════════
#  AVAILABILITY — part of the environment, not an add-on (protocol §9).
#  Each sample arrives with a set A of physically present modalities. avail=None means
#  "everything present" (condition A0), which is the DEGENERATE case: the root state then
#  carries no per-sample information at all, so the first pick cannot adapt (§8.8).
# ═════════════════════════════════════════════════════════════════════════════
def sample_availability(N, M, p, rng):
    """Bernoulli(p) per modality per sample; guarantees |A| >= 1. p=1.0 -> condition A0."""
    av = rng.random((N, M)) < p
    empty = ~av.any(1)
    if empty.any():
        av[empty, rng.integers(0, M, int(empty.sum()))] = True
    return av


def _avail(avail, N, M):
    return np.ones((N, M), bool) if avail is None else np.asarray(avail, bool)


# ─────────────────────────────────────────────────────────────────────────────
# STATE: for every (node, sample), the vector the Q-network sees.
#   [ mask(M) | available(M) | softmax(C) | entropy | margin | depth/M | |A|/M ]
#   Fdim = 2M + C + 4
# `mask` alone conflates "not acquired yet" with "does not exist"; |A|/M because a
# 2-modality sample calls for a different strategy than a 6-modality one.
# ─────────────────────────────────────────────────────────────────────────────
def build_state(cache, M, md, avail=None):
    nodes = cache['nodes']; soft = cache['softmax']; n, N, C = soft.shape
    depth, used, _ = node_meta(nodes, M, md)
    av = _avail(avail, N, M)
    H = np.zeros((n, N)); margin = np.zeros((n, N)); kk = np.zeros((n, N))
    mask = np.zeros((n, N, M), np.float32)
    for i in range(n):
        p_ = np.clip(soft[i], 1e-8, 1.0)
        H[i] = -(p_ * np.log(p_)).sum(-1)
        srt = np.sort(soft[i], -1); margin[i] = srt[:, -1] - srt[:, -2]
        kk[i] = depth[i] / M
        for m in used[i]:
            mask[i, :, m] = 1.0
    avb = np.broadcast_to(av.astype(np.float32)[None], (n, N, M))
    nav = np.broadcast_to((av.sum(1) / M).astype(np.float32)[None, :, None], (n, N, 1))
    return np.concatenate([mask, avb, soft, H[..., None], margin[..., None],
                           kk[..., None], nav], -1).astype(np.float32)


# ── FORCED FIRST PICK (protocol §8.8 fix) ───────────────────────────────────
# At A0 the ROOT state is identical for every sample (mask=0, avail=all-ones, root
# softmax constant), so the first action cannot be per-sample adaptive — argmax Q is a
# constant, and which constant it lands on is left to the FQI residual, which sometimes
# overrides the Shapley prior with a worse modality.
#
# Setting FORCE_FIRST_ORDER to a static Shapley order (descending phi) removes that
# non-decision: at depth 0 the ONLY legal action is the highest-Shapley modality that the
# sample actually has. Every later step stays fully dynamic and is conditioned on what the
# first modality returned.
#
# Because this lives in the action mask, it binds identically in TRAINING (the FQI
# targets take argmax over legal actions) and at TEST (both rollouts mask the same way),
# so the deployed policy is the one the Bellman targets were computed for.
FORCE_FIRST_ORDER = None      # None = off; else list of modality indices, best first


def action_mask_avail(n, M, depth, ch, avail, N):
    """[n, N, A] legal actions PER SAMPLE:
         acquire m  <- available[sample,m] AND m not acquired AND child exists
         STOP       <- depth > 0
    With FORCE_FIRST_ORDER set, depth-0 nodes keep only the first AVAILABLE modality in
    that order (availability guarantees >=1 survivor, so the row is never all-False).
    """
    av = _avail(avail, N, M); A = M + 1
    mk = np.zeros((n, N, A), bool)
    for i in range(n):
        for (m, _c) in ch[i]:
            mk[i, :, m] = av[:, m]
        if depth[i] > 0:
            mk[i, :, M] = True
    if FORCE_FIRST_ORDER is not None:
        # first available modality per sample, in Shapley order
        forced = np.full(N, -1, int)
        for m in FORCE_FIRST_ORDER:
            take = (forced < 0) & av[:, m]
            forced[take] = m
        for i in range(n):
            if depth[i] != 0:
                continue
            keep = np.zeros((N, A), bool)
            have = forced >= 0
            # only force a child that actually exists at the root
            child_ok = np.zeros(N, bool)
            for (m, _c) in ch[i]:
                child_ok |= have & (forced == m)
            keep[child_ok, forced[child_ok]] = True
            # samples whose forced pick has no child keep their original legal set
            mk[i] = np.where(child_ok[:, None], keep, mk[i])
    return mk


def reachable_nodes(used, avail, n, N, M):
    """[n, N] — can this sample even BE at this node? A node whose acquired set contains a
    modality the sample does not have can never occur; exclude it from the loss."""
    av = _avail(avail, N, M)
    R = np.ones((n, N), bool)
    for i in range(n):
        for m in used[i]:
            R[i] &= av[:, m]
    return R


def step_cost(lam, avail, N, M):
    """[N] per-acquire cost = lam / |A| (protocol §5).

    NORMALISED by what was available, so the penalty means the same thing across conditions:
    a flat lam prices 2 acquires identically whether |A|=2 ("took everything") or |A|=6
    ("took a third"). With this, consuming ALL of A costs exactly lam, whatever |A| is —
    so lam is directly comparable to an F1 gain.
    """
    av = _avail(avail, N, M)
    return (lam / av.sum(1).astype(np.float32))


BALANCED = True      # class-balanced reward. Set False for plain 1[correct] (accuracy-aligned).
CONF_WEIGHTED = os.environ.get('RLB_CONF_WEIGHTED') == '1'
CW_EFF        = os.environ.get('RLB_CW_EFF')        == '1'  # CW + efficiency penalty for late stops
# option-2 reward: R = 1[correct] * p(y|s) * 1/freq(y).
# The softmax is frozen/cached so the policy cannot game its own confidence;
# the only effect is denser reward signal (partial credit for high-p correct stops).


def correctness(cache, balanced=None):
    """Terminal reward at every node, per sample -> [n, N].

    balanced=False :  R = 1[correct]                                      -- ACCURACY
    balanced=True  :  R = 1[correct] * 1/freq(y)                         -- MACRO-F1
    CONF_WEIGHTED  :  R = 1[correct] * p(y|s) * 1/freq(y)                -- CW macro-F1
    CW_EFF         :  R = 1[correct] * p(y|s) * (M+1-depth)/(M+1) / freq -- CW + efficiency

    CW_EFF addresses the failure mode of plain CONF_WEIGHTED: p(y|s) increases with more
    acquired modalities, so pure CW pushes the policy to acquire everything regardless of
    lambda. The efficiency factor (M+1-depth)/(M+1) decays linearly from 1 at depth-0 to
    1/(M+1) at depth-M, making early confident stops strictly more rewarding than late ones
    at the same confidence level. It never reaches zero so full-modality stops still carry
    a positive reward.
    """
    balanced = BALANCED if balanced is None else balanced
    soft = cache['softmax']; y = cache['y']
    corr = (soft.argmax(-1) == y[None, :]).astype(np.float32)
    if CONF_WEIGHTED or CW_EFF:
        N = len(y)
        p_y = soft[:, np.arange(N), y].astype(np.float32)   # [n, N] prob of true class
        corr = corr * p_y
        if CW_EFF:
            M = int(cache['M'])
            nodes = cache['nodes']
            d = np.array([0 if k == '' else k.count(',') + 1 for k in nodes], np.float32)
            eff = ((M + 1 - d) / (M + 1))[:, None]          # [n, 1], range 1/(M+1)..1
            corr = corr * eff
    if not balanced:
        return corr
    C = soft.shape[-1]
    freq = np.bincount(y, minlength=C).astype(np.float64) / len(y)
    w = 1.0 / np.maximum(freq[y], 1e-8)                  # per-sample inverse class frequency
    w = w / w.mean()                                     # keep E[w] = 1 -> reward scale unchanged
    return (corr * w[None, :].astype(np.float32)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Q-NETWORK: tiny MLP, state -> one Q-value per action (M acquire + 1 STOP).
# ─────────────────────────────────────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, Fdim, A, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Fdim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, A))

    def forward(self, x):
        return self.net(x)


def inner_split(N, seed=1000, frac=0.2):
    """Deterministic held-out split of the val samples for checkpoint selection."""
    perm = np.random.default_rng(seed).permutation(N)
    cut = max(5, int(frac * N))
    return perm[cut:], perm[:cut]              # (train_idx, holdout_idx)


# ═════════════════════════════════════════════════════════════════════════════
#  DEPLOY ROLLOUTS — availability-aware. Two exit rules:
#    rollout_delta    : LEARNED stop  -> stop when Q[STOP] + delta >= max legal acquire Q
#    rollout_conf_eps : CONFIDENCE    -> stop when max softmax >= tau
#  Both intersect the legal action set with `avail` per sample, and both return
#  (macro-F1, mean #acquired, mean GFLOPs, stop_nodes, first_pick).
#  eps > 0 = epsilon-greedy deploy: a greedy rollout on a deterministic cache gives std = 0
#  by construction, which is not an honest error bar.
# ═════════════════════════════════════════════════════════════════════════════
def _egreedy(Qi, legal_i, eps, rng):
    """epsilon-greedy over each sample's OWN legal set. Qi [N,A], legal_i [N,A] -> [N]."""
    N, A = Qi.shape
    a = np.where(legal_i, Qi, -1e9).argmax(-1)
    if eps <= 0:
        return a
    explore = rng.random(N) < eps
    if explore.any():
        a = a.copy()
        for r in np.flatnonzero(explore):
            legal = np.flatnonzero(legal_i[r])
            if len(legal):
                a[r] = rng.choice(legal)
    return a


def _finish(cache, sn, depth, first):
    N = len(cache['y'])
    preds = cache['softmax'][sn, np.arange(N)].argmax(-1)
    f1 = float(f1_score(cache['y'], preds, average='macro'))
    g = float(cache['gflops'][sn].mean()) if 'gflops' in cache else float('nan')
    return f1, float(depth[sn].mean()), g, sn, first


@torch.no_grad()
def _walk(net, cache, M, md, Sf, avail, eps, rng, stop_fn):
    """Shared tree walk. stop_fn(node_i, here) -> bool[N]: should these samples stop here?"""
    nodes = cache['nodes']; depth, used, ch = node_meta(nodes, M, md)
    n, N, Fdim = Sf.shape; A = M + 1
    dev = next(net.parameters()).device
    Q = net(torch.tensor(Sf).reshape(n * N, Fdim).to(dev)).reshape(n, N, A).cpu().numpy()
    mk = action_mask_avail(n, M, depth, ch, avail, N)
    cur = np.zeros(N, int); stopped = np.zeros(N, bool)
    sn = cur.copy(); first = np.full(N, -1, int)
    for step in range(md + 1):
        for i in np.unique(cur[~stopped]):
            here = (cur == i) & (~stopped)
            if not ch[i]:                                  # no children: forced stop
                sn[here] = i; stopped |= here; continue
            no_legal = ~mk[i].any(-1)                      # nothing available to acquire
            fin = here & (no_legal | stop_fn(i, here, Q, mk))
            sn[fin] = i; stopped |= fin
            go = here & (~fin)
            if not go.any():
                continue
            a = _egreedy(Q[i], mk[i], eps, rng)
            if step == 0:
                first[go] = a[go]
            for (m, c) in ch[i]:
                cur[go & (a == m)] = c
        if stopped.all():
            break
    sn[~stopped] = cur[~stopped]
    return _finish(cache, sn, depth, first)


def rollout_delta(net, cache, M, md, Sf, delta=0.0, avail=None, eps=0.0, rng=None):
    """LEARNED stop. `delta` sweeps the frontier at eval time (delta up -> stop earlier)."""
    rng = rng or np.random.default_rng(0)
    def stop_fn(i, here, Q, mk):
        Qa = np.where(mk[i][:, :M], Q[i][:, :M], -1e9).max(-1)
        return mk[i][:, M] & (Q[i][:, M] + delta >= Qa)
    return _walk(net, cache, M, md, Sf, avail, eps, rng, stop_fn)


def rollout_conf_eps(net, cache, M, md, tau, Sf, avail=None, eps=0.0, rng=None):
    """CONFIDENCE exit at tau. The policy learns ORDER only."""
    rng = rng or np.random.default_rng(0)
    conf = cache['softmax'].max(-1)
    def stop_fn(i, here, Q, mk):
        return mk[i][:, M] & (conf[i] >= tau)              # mk[:,M] is False at the root
    return _walk(net, cache, M, md, Sf, avail, eps, rng, stop_fn)


def macro_f1(y, preds):
    return float(f1_score(y, preds, average='macro'))


# ═════════════════════════════════════════════════════════════════════════════
#  COST MODEL — UNIFORM. Every modality costs the same: `lam` per acquire.
#
#  Why uniform, and not weighted by compute or feature dimension:
#
#   * MEASURED GFLOPs are already ~uniform per modality on these datasets --
#       cmi 1.01x,  czu_mhad 1.22x,  iemocap 1.28x  (max/min over modalities).
#     And gflops[S] is exactly additive (|gflops[S] - sum_m c_m| < 3e-5), so a
#     GFLOPs cost is just `lam * 0.075 * (#modalities)` -- uniform cost in disguise.
#   * FEATURE DIMENSION is actively misleading: the MBT projects every modality to
#     128-d before fusion, so IEMOCAP feature dims span 6..1280 (213x) while real
#     GFLOPs span 1.28x. The proxy prices audio at 8.7x mocap_head; truth is 1.28x.
#
#  So cost = lam * (number of modalities acquired). `lam` is the STOP lever for the
#  learned-stop variants and is swept; for confidence-exit variants tau sets the stop
#  and lam only shapes the acquisition ORDER.
#
#  GFLOPs is still REPORTED (see rollout_delta / _score callers) -- it just does not
#  steer the policy.
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Non-uniform cost: per-modality weight c_m from raw feature dimension (softened).
#   c_m = 1 + log2(feat_dim_m / min_dim)   (cheapest modality = 1)
