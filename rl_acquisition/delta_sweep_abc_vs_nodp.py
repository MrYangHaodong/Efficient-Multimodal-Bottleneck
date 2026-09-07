"""Cheap frontier: sweep the TEST-TIME delta on the already-trained lambda=1.0 nets.

delta is a post-hoc offset on Q[STOP] applied at rollout (dqn_policy_qprior._rollout:
`Q[:, :, M] += delta`), so it needs NO retraining — one net traces a whole frontier. That
is how RL_AAAI_final swept its frontier (LAM fixed at 0.05, lever = delta); the branch
promoted lambda to the lever for Bellman consistency, at 7x the training cost.

Here we use delta purely as a cheap PROBE of the A+C vs no-DP ablation at matched
training cost (both nets trained at lambda=1.0, --force_first --res_l2 0.1):
    A+C    = --exact_dp --stop_bce 1.0   (EXACT_DP=True,  STOP_BCE=1.0)
    no-DP  = FQI bootstrap               (EXACT_DP=False, STOP_BCE=0)

Caveat kept explicit: a delta-swept point is NOT self-consistent with its net's Bellman
targets, so these curves are for locating the interesting region, not for publication.
Retrain at the chosen lambda for the reported number.

  RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity \
    PYTHONPATH=/home/group/maestro_visual/Efficient-Multimodal-Bottleneck \
    python delta_sweep_abc_vs_nodp.py [ds ...]
"""
import os
import sys

os.environ.setdefault('RLB_RUNSET', 'k16p')
os.environ.setdefault('RLB_CACHE_SUBDIR', 'botneck_16_parity')

import numpy as np
import pandas as pd
import torch

import environment as env
import rl_core as rl
import baselines
import shapley
import dqn_policy_qprior as QPR

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = {'iemocap': [0, 1, 2], 'cmi': [0, 1, 2], 'czu_mhad': [0, 1, 2], 'mmfi': [0, 1, 2],
         'dsads': [0, 1, 2, 3], 'eav': [0, 1, 2], 'pamap2': [0, 1, 2, 3], 'utd_mhad': [0, 1, 2]}
# Informative range only: measured on eav/utd_mhad, |delta| >= 0.5 saturates (all M
# modalities below, a single modality above), so the grid is concentrated where the
# stop margin actually flips.
DELTAS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
# every trained lambda is swept: lambda picks the NET, delta then traces that net's own
# frontier, so the union is a dense (lambda x delta) grid at no extra training cost.
import glob as _glob, re as _re


def available_lams(ds, is_abc):
    out = set()
    for f in _glob.glob(f'checkpoints_k16p/{ds}/qprior_*_ff*.pt'):
        b = os.path.basename(f)
        if ('_dp' in b and '_sb1' in b) != is_abc:
            continue
        m = _re.search(r'lam([0-9.]+)_', b)
        if m:
            out.add(float(m.group(1)))
    return sorted(out)
SEEDS = [0, 1, 2]
VARIANTS = {  # name -> (EXACT_DP, STOP_BCE)
    'A+C':   (True, 1.0),
    'no-DP': (False, 0.0),
}


def run(ds):
    rows = []
    md = env.depth_of(ds)
    for fold in FOLDS[ds]:
        val = env.get_cache('val', ds, fold)
        test = env.get_cache('test', ds, fold)
        M = val['M']
        # both variants were trained with --force_first: the deployed walk must match
        rl.FORCE_FIRST_ORDER = baselines.shapley_order_from(val, M, md)
        vv = shapley.conditional_tables(val, M, md)
        for name, (dp, sb) in VARIANTS.items():
            QPR.EXACT_DP, QPR.STOP_BCE, QPR.RES_L2 = dp, sb, 0.1
            QPR.CKPT_SUFFIX = '_ff'
            for net_lam in available_lams(ds, name == 'A+C'):
              for seed in SEEDS:
                ck = QPR._ckpt_path(ds, fold, seed, net_lam, QPR.P_TRAIN, QPR.RESAMPLE)
                if not os.path.exists(ck):
                    continue
                net, tabs = QPR.train_or_load(val, M, md, dataset=ds, fold=fold, seed=seed,
                                              lam=net_lam, p_train=QPR.P_TRAIN,
                                              resample=QPR.RESAMPLE, tables=vv,
                                              force_retrain=False)
                for d in DELTAS:
                    f1, macq, g, _sn, _f = QPR.rollout(net, test, M, md, tabs, delta=d,
                                                       avail=None, eps=0.0, rng=None)
                    rows.append(dict(dataset=ds, variant=name, fold=fold, seed=seed,
                                     net_lam=net_lam, delta=d, macro_f1=f1,
                                     mean_acquired=macq, gflops=g))
        rl.FORCE_FIRST_ORDER = None
    return rows


def main():
    todo = sys.argv[1:] or ['eav', 'utd_mhad', 'mmfi', 'pamap2', 'dsads',
                            'iemocap', 'czu_mhad', 'cmi']      # cheapest first
    allrows = []
    for ds in todo:
        try:
            r = run(ds)
            allrows += r
            print(f'{ds}: {len(r)} rows', flush=True)
            pd.DataFrame(allrows).to_csv(          # incremental: usable if interrupted
                os.path.join(HERE, 'results_k16p', 'delta_sweep_abc_vs_nodp.csv'), index=False)
        except Exception as e:
            print(f'SKIP {ds}: {type(e).__name__}: {str(e)[:160]}', flush=True)
    if not allrows:
        return
    df = pd.DataFrame(allrows)
    out = os.path.join(HERE, 'results_k16p', 'delta_sweep_abc_vs_nodp.csv')
    df.to_csv(out, index=False)
    print(f'\nwrote {out}  ({len(df)} rows)')


if __name__ == '__main__':
    main()
