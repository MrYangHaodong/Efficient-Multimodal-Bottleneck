"""The baselines reported alongside EVERY design, under every availability condition (§3).

  B1'  static Shapley, restricted  — one global order, skip what's not in A, stop at conf>=tau.
                                     THE BAR. Defines B_star (§7).
  B2'  all-available               — forward with all of A present. Reference at ratio 1.0.
  B3'  random-order                — per sample, uniformly permute A. B1'-B3' is what the
                                     Shapley ORDER buys, separated from what COUNTING buys.
  B_full  all M modalities         — the performance we want to match. ORDER-SENSITIVE.

None have learned parameters, so none vary with seed — compute once per fold.

⚠ Shapley is computed on the FIT split and on MACRO-F1 (the reported metric), never accuracy
and never on a split the backbone memorised (see shapley.py, protocol §8.1).
"""
import numpy as np

import rl_core as rl
import shapley


def _score(cache, sn, depth):
    """(macro-F1, mean #acquired, mean cumulative GFLOPs/sample) at the stop nodes."""
    N = len(cache['y'])
    preds = cache['softmax'][sn, np.arange(N)].argmax(-1)
    g = float(cache['gflops'][sn].mean()) if 'gflops' in cache else float('nan')
    return rl.macro_f1(cache['y'], preds), float(depth[sn].mean()), g


def _walk_order(cache, md, orders, tau):
    """Walk a per-sample order, stopping at the first prefix with conf >= tau.
    orders: list of length N, each a list of modality indices (already restricted to A).
    Returns (stop_nodes, first_pick)."""
    nidx = {k: i for i, k in enumerate(cache['nodes'])}
    conf = cache['softmax'].max(-1)
    N = cache['softmax'].shape[1]
    sn = np.zeros(N, int); first = np.full(N, -1, int)
    for j in range(N):
        key, node = '', 0
        for step, m in enumerate(orders[j][:md]):
            key = str(m) if key == '' else key + ',' + str(m)
            if key not in nidx:                    # beyond the cached depth
                break
            node = nidx[key]
            if step == 0:
                first[j] = m
            if conf[node, j] >= tau:
                break
        sn[j] = node
    return sn, first


def _avail(avail, N, M):
    return np.ones((N, M), bool) if avail is None else np.asarray(avail, bool)


def shapley_order_from(cache_fit, M, md):
    """The one global order. Computed on the FIT split, on macro-F1."""
    return shapley.shapley_order(cache_fit, M, md)


def b1_shapley_restricted(cache_test, M, md, order, tau, avail=None):
    """B1' — global Shapley order, skipping anything not in A, stop at conf >= tau."""
    N = cache_test['softmax'].shape[1]
    av = _avail(avail, N, M)
    depth = np.array([0 if k == '' else k.count(',') + 1 for k in cache_test['nodes']])
    orders = [[m for m in order if av[j, m]] for j in range(N)]
    sn, first = _walk_order(cache_test, md, orders, tau)
    return _score(cache_test, sn, depth), first


def b3_random_order(cache_test, M, md, tau, avail=None, seed=0):
    """B3' — per sample, a uniformly random permutation of A. Isolates the value of ORDER."""
    rng = np.random.default_rng(seed)
    N = cache_test['softmax'].shape[1]
    av = _avail(avail, N, M)
    depth = np.array([0 if k == '' else k.count(',') + 1 for k in cache_test['nodes']])
    orders = [list(rng.permutation(np.flatnonzero(av[j]))) for j in range(N)]
    sn, first = _walk_order(cache_test, md, orders, tau)
    return _score(cache_test, sn, depth), first


def b2_all_available(cache_test, M, md, order, avail=None):
    """B2' — everything in A, no early stop. The ratio-1.0 reference for a restricted condition.

    Walked in the given global order so the sequential bottleneck sees a sensible sequence;
    with |A| > md the walk is truncated at md (the tree's depth)."""
    N = cache_test['softmax'].shape[1]
    av = _avail(avail, N, M)
    depth = np.array([0 if k == '' else k.count(',') + 1 for k in cache_test['nodes']])
    orders = [[m for m in order if av[j, m]] for j in range(N)]
    sn, _ = _walk_order(cache_test, md, orders, tau=2.0)      # tau>1 -> never stops early
    return _score(cache_test, sn, depth)


def b_full(dataset, fold, split='test', order=None):
    """B_full — ALL M modalities, one direct forward (NOT read off the tree).

    ⚠ ORDER-SENSITIVE. Pass the val-LOO order. The canonical order puts `video` first on
    IEMOCAP — the memorising modality, the worst seed — and costs ~0.04 F1 (protocol §3, §8.4).
    """
    import environment as env
    a = env.all_modality(dataset=dataset, fold=fold, split=split, order=order)
    return rl.macro_f1(a['y'], a['softmax'].argmax(-1)), float(a['gflops'])
