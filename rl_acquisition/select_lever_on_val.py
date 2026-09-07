"""Pick (lambda, delta) on VAL, report on TEST, and measure what the automation costs.

The delta sweep already rolled out every (lambda, delta) on BOTH splits
(val_f1/val_mods next to test_f1/test_mods/test_gflops), so every rule below is a
re-selection over existing rollouts -- no retraining, no new inference.

The only number that makes an automatic rule credible is the REGRET: how much test F1 the
val-chosen point loses against an oracle that had been allowed to choose on test. A rule
with low regret is a rule; a rule with high regret is test-set selection wearing a
disguise. Both are printed per dataset.

Rules (all selection uses val ONLY):
  global-f1    one (lambda,delta) for all 8 datasets, max mean val F1
  global-knee  one (lambda,delta) for all 8, cheapest whose mean val F1 >= (1-eps)*best
  per-ds-f1    per dataset, max val F1
  per-ds-knee  per dataset, cheapest whose val F1 >= (1-eps)*that dataset's best val F1
The knee rules are the efficiency-aware ones: among points that are statistically
indistinguishable on val, take the cheapest. eps defaults to 2%.

PROTOCOL CAVEAT, stated because it does not go away: the policy is FIT on val
(protocol section 2), so choosing the lever on val reuses that split and is mildly
optimistic. It is nonetheless the established rule here, and selecting on test is not a
rule at all. The regret column is measured against a test oracle, so it bounds how much
of the reported gain could be selection luck.

  python select_lever_on_val.py [--eps 0.02]
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ['iemocap', 'mmfi', 'cmi', 'czu_mhad', 'dsads', 'eav', 'pamap2', 'utd_mhad']


def load():
    fs = [os.path.join(HERE, 'results_k16p', f'delta_sweep_original_{r}.csv')
          for r in ('k16', 'k16p')]
    d = pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)])
    d['val_mu'] = d.val_mods / d.M
    d['test_mu'] = d.test_mods / d.M
    # average over folds and seeds -> one row per (dataset, lambda, delta)
    return d.groupby(['dataset', 'net_lam', 'delta']).agg(
        val_f1=('val_f1', 'mean'), val_mu=('val_mu', 'mean'),
        test_f1=('test_f1', 'mean'), test_mu=('test_mu', 'mean'),
        test_gf=('test_gflops', 'mean')).reset_index()


def knee(sub, eps, f1col, mucol):
    """Cheapest point (lowest MU) whose F1 is within eps of the best -- on whichever
    split's columns are passed in. Ties on MU break toward higher F1."""
    thr = (1.0 - eps) * sub[f1col].max()
    ok = sub[sub[f1col] >= thr]
    return ok.sort_values([mucol, f1col], ascending=[True, False]).iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.02)
    a = ap.parse_args()
    g = load()
    eps = a.eps

    # ── global rules: one operating point for every dataset ──────────────────────
    m = g.groupby(['net_lam', 'delta']).agg(
        val_f1=('val_f1', 'mean'), val_mu=('val_mu', 'mean'),
        test_f1=('test_f1', 'mean'), test_mu=('test_mu', 'mean'),
        test_gf=('test_gf', 'mean')).reset_index()
    gf1 = m.sort_values('val_f1', ascending=False).iloc[0]
    gkn = knee(m, eps, 'val_f1', 'val_mu')
    orc = m.sort_values('test_f1', ascending=False).iloc[0]      # test oracle, for regret

    print('=' * 92)
    print(f'GLOBAL rules -- ONE (lambda, delta) for all 8 datasets   [eps={eps:.0%}]')
    print('=' * 92)
    print(f"{'rule':14s} {'lam':>5s} {'delta':>6s} | {'val F1':>7s} | "
          f"{'test F1':>8s} {'test MU':>8s} {'test GF':>8s} | {'regret':>7s}")
    # Each rule is scored against ITS OWN test-side oracle. Charging the knee rule the gap
    # to the unconstrained test maximum would bill it for the accuracy it gave up ON
    # PURPOSE to be cheap, which is not selection error and would make a good rule look bad.
    okn = knee(m, eps, 'test_f1', 'test_mu')
    for name, r, o in [('global-f1', gf1, orc), ('global-knee', gkn, okn)]:
        print(f"{name:14s} {r.net_lam:5.2f} {r.delta:6.2f} | {r.val_f1:7.4f} | "
              f"{r.test_f1:8.4f} {r.test_mu:8.3f} {r.test_gf:8.3f} | "
              f"{o.test_f1 - r.test_f1:+7.4f}")
    print(f"{'oracle(max-F1)':14s} {orc.net_lam:5.2f} {orc.delta:6.2f} | {'':7s} | "
          f"{orc.test_f1:8.4f} {orc.test_mu:8.3f} {orc.test_gf:8.3f} |")
    print(f"{'oracle(knee)':14s} {okn.net_lam:5.2f} {okn.delta:6.2f} | {'':7s} | "
          f"{okn.test_f1:8.4f} {okn.test_mu:8.3f} {okn.test_gf:8.3f} |")

    # ── per-dataset rules ────────────────────────────────────────────────────────
    for rule in ('per-ds-f1', 'per-ds-knee'):
        rows = []
        for ds in DS:
            s = g[g.dataset == ds]
            if rule == 'per-ds-f1':
                pick = s.sort_values('val_f1', ascending=False).iloc[0]
                ref = s.test_f1.max()                       # same objective, on test
            else:
                pick = knee(s, eps, 'val_f1', 'val_mu')
                ref = knee(s, eps, 'test_f1', 'test_mu').test_f1   # knee-vs-knee
            rows.append((ds, pick.net_lam, pick.delta, pick.test_f1, pick.test_mu,
                         pick.test_gf, ref - pick.test_f1))
        print()
        print('=' * 92)
        print(f'{rule.upper()} -- lever chosen per dataset on VAL   [eps={eps:.0%}]')
        print('=' * 92)
        print(f"{'dataset':9s} {'lam':>5s} {'delta':>6s} | {'test F1':>8s} {'test MU':>8s} "
              f"{'test GF':>8s} | {'regret':>7s}")
        for r in rows:
            print(f'{r[0]:9s} {r[1]:5.2f} {r[2]:6.2f} | {r[3]:8.4f} {r[4]:8.3f} '
                  f'{r[5]:8.3f} | {r[6]:+7.4f}')
        arr = np.array([[r[3], r[4], r[5], r[6]] for r in rows])
        print(f"{'MEAN':9s} {'':5s} {'':6s} | {arr[:,0].mean():8.4f} {arr[:,1].mean():8.3f} "
              f"{arr[:,2].mean():8.3f} | {arr[:,3].mean():+7.4f}")


if __name__ == '__main__':
    main()
