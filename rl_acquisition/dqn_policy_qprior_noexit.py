"""
dqn_policy_qprior_noexit — qprior's brain, but EXIT = STATE, not action.
=========================================================================
Same conditional-Shapley-bootstrapped ordering as `dqn_policy_qprior`; the ONLY difference is how
the episode ends:

    qprior          EXIT = ACTION.  STOP is action M; the net learns Q[STOP]; `delta` is the lever.
    qprior_noexit   EXIT = STATE.   No STOP action. The ENVIRONMENT terminates when the model's own
                                    confidence (max-softmax) crosses `tau`. `tau` is the lever.

WHY they share a checkpoint. The optimal ORDER is the same under both exits: the ordering that
climbs toward a correct prediction fastest is what a learned STOP rewards AND what conf-exit needs.
So the trained ResidualQ net is identical -- this module imports it and the whole training stack
from `dqn_policy_qprior` and changes ONLY the rollout. `python run_design.py` with either design
loads the same `checkpoints/<ds>/qprior_fold*_seed*_lam0.050_*.pt`.

WHY conf-exit at all. `delta` is a post-hoc offset on a learned Q -- it does not transfer val->test
and it floors utilisation at ~0.55 (the learned STOP wants ~4 modalities regardless of lam). `tau`
is a NATIVE, transferable exit rule (same family as baseline B1'): it reaches util 0.17, and it
makes qprior_noexit vs B1' a clean matched-exit test where the ONLY difference is the order
(adaptive per-sample vs static global Shapley).

  reward/cost/state/training : ALL inherited from dqn_policy_qprior (unchanged).
  stop  : environment forces termination at conf >= tau; `tau` sweeps the frontier at eval time.

Run:  python dqn_policy_qprior_noexit.py
"""
import numpy as np
import torch

import environment as env
import rl_core as rl

# Shared brain: identical training, Q-prior, network, and checkpointing as the exit-as-action policy.
from dqn_policy_qprior import (  # noqa: F401  (re-exported so callers can use one import)
    LAM, GAMMA, SEED, P_TRAIN, RESAMPLE,
    q_prior, ResidualQ, _fwd, train, train_or_load, _ckpt_path,
)


@torch.no_grad()
def _rollout_confexit(net, cache, M, md, Sf, Qp, tau, avail=None, eps=0.0, rng=None):
    """Adaptive order (argmax Q over available acquire actions), but STOP is NOT an action -- the
    environment terminates at max-softmax >= tau.

    Semantics match baselines._walk_order EXACTLY, so qprior_noexit vs B1' differs ONLY in the
    order: always take >=1 modality, then stop at the first prefix whose conf >= tau. The Q STOP
    column is unused; the acquire columns (conf + advice - lam/|A| + residual) supply the order.
    """
    rng = rng or np.random.default_rng(0)
    nodes = cache['nodes']; depth, used, ch = rl.node_meta(nodes, M, md)
    n, N, Fdim = Sf.shape; A = M + 1
    dev = next(net.parameters()).device
    Q = _fwd(net, torch.tensor(Sf).reshape(n * N, Fdim).to(dev)).reshape(n, N, A).cpu().numpy() + Qp
    conf = cache['softmax'].max(-1)                       # [n, N] observable, deployable
    mk = rl.action_mask_avail(n, M, depth, ch, avail, N)
    cur = np.zeros(N, int); stopped = np.zeros(N, bool)
    sn = cur.copy(); first = np.full(N, -1, int)
    for stp in range(md + 1):
        for i in np.unique(cur[~stopped]):
            here = (cur == i) & (~stopped)
            if not ch[i]:                                 # leaf: nothing left to acquire
                sn[here] = i; stopped |= here; continue
            has_acq = mk[i][:, :M].any(-1)                # any available, unacquired modality?
            root = depth[i] == 0                          # never stop at the empty set (take >=1)
            confstop = conf[i] >= tau if not root else np.zeros(N, bool)
            fin = here & (~has_acq | confstop)            # env exit: no moves left, or conf>=tau
            sn[fin] = i; stopped |= fin
            go = here & (~fin)
            if not go.any():
                continue
            a = np.where(mk[i][:, :M], Q[i][:, :M], -1e9).argmax(-1)   # adaptive greedy order
            if eps > 0:
                flip = go & (rng.random(N) < eps)
                for j in np.flatnonzero(flip):
                    legal = np.flatnonzero(mk[i][j, :M])
                    if len(legal):
                        a[j] = rng.choice(legal)
            if stp == 0:
                first[go] = a[go]
            for (m, c) in ch[i]:
                cur[go & (a == m)] = c
        if stopped.all():
            break
    sn[~stopped] = cur[~stopped]
    preds = cache['softmax'][sn, np.arange(N)].argmax(-1)
    g = float(cache['gflops'][sn].mean()) if 'gflops' in cache else float('nan')
    return rl.macro_f1(cache['y'], preds), float(depth[sn].mean()), g, sn, first


def rollout(net, cache, M, md, vv, lam=LAM, tau=0.5, avail=None, eps=0.0, rng=None):
    """Signature mirrors dqn_policy_qprior.rollout, but the lever is `tau`, not `delta`."""
    Sf = rl.build_state(cache, M, md, avail)
    Qp = q_prior(cache, M, md, vv, lam, avail)
    return _rollout_confexit(net, cache, M, md, Sf, Qp, tau, avail, eps, rng)


def main():
    ds, fd = env.DATASET, env._FOLD
    val = env.get_cache('val', ds, fd); test = env.get_cache('test', ds, fd)
    M, md = val['M'], env.depth_of(ds)
    net, vv = train_or_load(val, M, md, dataset=ds, fold=fd)   # loads the shared qprior checkpoint
    f1, macq, g, sn, first = rollout(net, test, M, md, vv, tau=0.9)
    print(f"\n[Q-PRIOR NOEXIT — conf>=tau environment exit]  {ds} fold{fd} (A0, tau=0.9)")
    print(f"  TEST macro-F1    : {f1:.4f}")
    print(f"  mean #acquired   : {macq:.2f}  (of {M})")
    print(f"  mean GFLOPs      : {g:.4f}")


if __name__ == '__main__':
    main()
