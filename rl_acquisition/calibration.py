"""Per-depth temperature scaling of the cached posteriors (softmax) — fitted on the VAL
cache, applied to fit/val/test.

Why per depth: shallow prefix-tree nodes (1 modality acquired) are systematically more
overconfident than full-depth nodes, and Q[STOP] = conf means that overconfidence directly
causes premature stopping. A single scalar T would average the two regimes away.

Only the stop policy changes: temperature scaling is monotone per row, so argmax — and
therefore F1 at any FIXED acquisition set — is untouched. Any F1 change downstream is
attributable to better stop/acquire decisions, which is exactly the isolated test.

No logits are stored in the caches, so we recalibrate from probabilities:
softmax(log p / T) = normalize(p^(1/T)) — the standard probability-space form.
"""
import numpy as np


def _node_depth(node):
    return 0 if node == '' else node.count(',') + 1


def _nll(p, y, T):
    q = p ** (1.0 / T)
    q = q / q.sum(-1, keepdims=True)
    return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0)).mean()


def fit_temperature(cache, grid=None):
    """T per depth, by NLL grid search on the val cache. Returns {depth: T}."""
    if grid is None:
        grid = np.logspace(-0.7, 0.9, 81)          # T in ~[0.2, 8]
    soft, y = cache['softmax'], np.asarray(cache['y'])
    depths = np.array([_node_depth(nd) for nd in cache['nodes']])
    Ts = {}
    for d in np.unique(depths):
        p = soft[depths == d].reshape(-1, soft.shape[-1])
        yy = np.tile(y, (depths == d).sum())
        Ts[int(d)] = float(grid[np.argmin([_nll(p, yy, t) for t in grid])])
    return Ts


def apply_temperature(cache, Ts):
    """In-place: softmax -> normalize(softmax^(1/T_depth)). Fresh-loaded caches only."""
    soft = cache['softmax']
    for i, nd in enumerate(cache['nodes']):
        T = Ts.get(_node_depth(nd), 1.0)
        if abs(T - 1.0) < 1e-9:
            continue
        q = soft[i] ** (1.0 / T)
        soft[i] = q / q.sum(-1, keepdims=True)
    return cache
