"""delta sweep on the ORIGINAL RL policy (plain dqn_qprior) across all 8 datasets.

Original = no res_l2, no exact-DP, no stop-BCE, no forced first pick — the published
policy, the one RL_AAAI_final ships. delta is the test-time offset on Q[STOP] applied at
rollout, so one trained net per lambda traces a whole frontier for free.

Runset per dataset group, so every backbone is the enc3/fus3/distill0 reference arch:
  iemocap/cmi/czu_mhad/mmfi -> k16   (already the reference architecture)
  dsads/eav/pamap2/utd_mhad -> k16p  (the 2026-08-01 architecture-parity retrains)

FORCE_FIRST_ORDER stays None: the original policy was trained with a free first pick, so
pinning it at rollout would deploy a different policy than the one that was trained.

  RLB_RUNSET=k16  RLB_CACHE_SUBDIR=botneck_16 \
    PYTHONPATH=/home/group/maestro_visual/Efficient-Multimodal-Bottleneck \
    python delta_sweep_original.py iemocap cmi czu_mhad mmfi
  RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity ... python delta_sweep_original.py dsads eav pamap2 utd_mhad
"""
import os
import sys

import numpy as np
import pandas as pd

import environment as env
import rl_core as rl
import shapley
import dqn_policy_qprior as QPR

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = {'iemocap': [0, 1, 2], 'cmi': [0, 1, 2], 'czu_mhad': [0, 1, 2], 'mmfi': [0, 1, 2],
         'dsads': [0, 1, 2, 3], 'eav': [0, 1, 2], 'pamap2': [0, 1, 2, 3], 'utd_mhad': [0, 1, 2]}
DELTAS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
LAMS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
SEEDS = [0, 1, 2]
RUNSET = os.environ.get('RLB_RUNSET', 'k16')
OUT = os.path.join(HERE, 'results_k16p', f'delta_sweep_original_{RUNSET}.csv')


def run(ds):
    rows, md = [], env.depth_of(ds)
    for fold in FOLDS[ds]:
        val = env.get_cache('val', ds, fold)
        test = env.get_cache('test', ds, fold)
        M = val['M']
        rl.FORCE_FIRST_ORDER = None          # original policy: free first pick
        vv = shapley.conditional_tables(val, M, md)
        # plain policy: every optimisation knob off
        QPR.EXACT_DP, QPR.STOP_BCE, QPR.RES_L2, QPR.BW_K = False, 0.0, 0.0, 0.0
        QPR.CKPT_SUFFIX, QPR.ADVICE_MODE, QPR.ABLATION = '', 'conditional', 'full'
        QPR.W_ADV = 1.0
        for lam in LAMS:
            for seed in SEEDS:
                if not os.path.exists(QPR._ckpt_path(ds, fold, seed, lam,
                                                     QPR.P_TRAIN, QPR.RESAMPLE)):
                    continue
                net, tabs = QPR.train_or_load(val, M, md, dataset=ds, fold=fold, seed=seed,
                                              lam=lam, p_train=QPR.P_TRAIN,
                                              resample=QPR.RESAMPLE, tables=vv,
                                              force_retrain=False)
                for d in DELTAS:
                    vf1, vm, _g, _a, _b = QPR.rollout(net, val, M, md, tabs, delta=d,
                                                      avail=None, eps=0.0, rng=None)
                    tf1, tm, tg, _c, _e = QPR.rollout(net, test, M, md, tabs, delta=d,
                                                      avail=None, eps=0.0, rng=None)
                    rows.append(dict(dataset=ds, runset=RUNSET, fold=fold, seed=seed,
                                     net_lam=lam, delta=d, val_f1=vf1, val_mods=vm,
                                     test_f1=tf1, test_mods=tm, test_gflops=tg, M=M))
    return rows


def main():
    allrows = []
    for ds in (sys.argv[1:] or list(FOLDS)):
        try:
            allrows += run(ds)
            print(f'{ds}: {len(allrows)} rows total', flush=True)
            pd.DataFrame(allrows).to_csv(OUT, index=False)
        except Exception as e:
            print(f'SKIP {ds}: {type(e).__name__}: {str(e)[:150]}', flush=True)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
