"""Dataset-adaptive (lambda, delta) selection — and an honest measure of what it costs.

lambda picks WHICH frontier the policy sits on (a training cost, one net per value);
delta picks WHERE on that frontier it operates (a test-time offset on Q[STOP], free).
The delta sweep showed the best lambda is strongly dataset-dependent (0.1 on utd_mhad,
0.2 on dsads/czu, 1.0 on eav, 1.2 on iemocap, 2.0 on pamap2, 3.0 on mmfi), so a single
global lambda is leaving accuracy on the table.

This rolls out EVERY (lambda, delta) on BOTH val and test, then answers the only question
that matters for an automatic rule: if we select on VAL at a target budget, how much test
F1 do we lose versus an oracle that selected on TEST? That gap is the price of automation
and it has to be reported, not assumed away.

Protocol note: the policy is fit on val (protocol section 2), so val-based lever selection
reuses that split and is mildly optimistic. It is nonetheless the established rule here,
and the alternative (selecting on test) is not a rule at all.

  RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity \
    PYTHONPATH=/home/group/maestro_visual/Efficient-Multimodal-Bottleneck \
    python lambda_delta_autoselect.py [ds ...]
"""
import os
import sys

os.environ.setdefault('RLB_RUNSET', 'k16p')
os.environ.setdefault('RLB_CACHE_SUBDIR', 'botneck_16_parity')

import glob as _glob
import re as _re

import numpy as np
import pandas as pd

import environment as env
import rl_core as rl
import baselines
import shapley
import dqn_policy_qprior as QPR

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = {'iemocap': [0, 1, 2], 'cmi': [0, 1, 2], 'czu_mhad': [0, 1, 2], 'mmfi': [0, 1, 2],
         'dsads': [0, 1, 2, 3], 'eav': [0, 1, 2], 'pamap2': [0, 1, 2, 3], 'utd_mhad': [0, 1, 2]}
DELTAS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2]
OUT = os.path.join(HERE, 'results_k16p', 'lambda_delta_valtest.csv')


def available_lams(ds):
    out = set()
    for f in _glob.glob(f'checkpoints_k16p/{ds}/qprior_*_ff*.pt'):
        b = os.path.basename(f)
        if not ('_dp' in b and '_sb1' in b):       # A+C nets only
            continue
        m = _re.search(r'lam([0-9.]+)_', b)
        if m:
            out.add(float(m.group(1)))
    return sorted(out)


def run(ds):
    rows = []
    md = env.depth_of(ds)
    for fold in FOLDS[ds]:
        val = env.get_cache('val', ds, fold)
        test = env.get_cache('test', ds, fold)
        M = val['M']
        rl.FORCE_FIRST_ORDER = baselines.shapley_order_from(val, M, md)   # A+C was trained with it
        vv = shapley.conditional_tables(val, M, md)
        QPR.EXACT_DP, QPR.STOP_BCE, QPR.RES_L2 = True, 1.0, 0.1
        QPR.CKPT_SUFFIX = '_ff'
        for lam in available_lams(ds):
            for seed in SEEDS:
                if not os.path.exists(QPR._ckpt_path(ds, fold, seed, lam,
                                                     QPR.P_TRAIN, QPR.RESAMPLE)):
                    continue
                net, tabs = QPR.train_or_load(val, M, md, dataset=ds, fold=fold, seed=seed,
                                              lam=lam, p_train=QPR.P_TRAIN,
                                              resample=QPR.RESAMPLE, tables=vv,
                                              force_retrain=False)
                for d in DELTAS:
                    vf1, vm, _vg, _a, _b = QPR.rollout(net, val, M, md, tabs, delta=d,
                                                       avail=None, eps=0.0, rng=None)
                    tf1, tm, tg, _c, _e = QPR.rollout(net, test, M, md, tabs, delta=d,
                                                      avail=None, eps=0.0, rng=None)
                    rows.append(dict(dataset=ds, fold=fold, seed=seed, net_lam=lam, delta=d,
                                     val_f1=vf1, val_mods=vm,
                                     test_f1=tf1, test_mods=tm, test_gflops=tg, M=M))
        rl.FORCE_FIRST_ORDER = None
    return rows


def main():
    todo = sys.argv[1:] or ['eav', 'utd_mhad', 'mmfi', 'pamap2', 'dsads',
                            'iemocap', 'czu_mhad', 'cmi']
    allrows = []
    for ds in todo:
        try:
            allrows += run(ds)
            print(f'{ds}: {len(allrows)} rows total', flush=True)
            pd.DataFrame(allrows).to_csv(OUT, index=False)
        except Exception as e:
            print(f'SKIP {ds}: {type(e).__name__}: {str(e)[:150]}', flush=True)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
