"""Roll out ONE fixed (lambda, delta) at every missingness condition -> Main_Results 'Ours'.

The lambda-sweep CSVs cover A0/A20/A40 but only at delta=0; the delta-sweep covers every
delta but only at A0 (it rolls out with avail=None). Nothing existing has BOTH a non-zero
delta and a missingness condition, which is what the reported table needs.

(lambda, delta) is fixed here, not re-picked: it comes from select_lever_on_val.py's
global-knee rule -- the cheapest point whose mean VAL F1 is within 2% of the best, chosen
on val alone, with a measured test regret of +0.028 against the test-side knee oracle.

Availability sampling mirrors run_design.run_rl exactly (same rng seeds, same
Bernoulli(p) draw per condition) so these rows are directly comparable with the existing
A0/A20/A40 numbers for every other design in the table.

  RLB_RUNSET=k16  RLB_CACHE_SUBDIR=botneck_16 \
    PYTHONPATH=<emb> python point_at_lever.py --lam 0.5 --delta 0.1 iemocap cmi czu_mhad mmfi
"""
import argparse
import os

import numpy as np
import pandas as pd

import environment as env
import rl_core as rl
import shapley
import dqn_policy_qprior as QPR

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = {'iemocap': [0, 1, 2], 'cmi': [0, 1, 2], 'czu_mhad': [0, 1, 2], 'mmfi': [0, 1, 2],
         'dsads': [0, 1, 2, 3], 'eav': [0, 1, 2], 'pamap2': [0, 1, 2, 3],
         'utd_mhad': [0, 1, 2]}
SEEDS = [0, 1, 2]
CONDITIONS = [('A0', 1.0), ('A20', 0.8), ('A40', 0.6)]      # missing% = 1 - availability
RUNSET = os.environ.get('RLB_RUNSET', 'k16')


def run(ds, lam, delta):
    rows, md = [], env.depth_of(ds)
    for fold in FOLDS[ds]:
        val = env.get_cache('val', ds, fold)
        test = env.get_cache('test', ds, fold)
        M = val['M']
        Nt, Nv = test['softmax'].shape[1], val['softmax'].shape[1]
        rl.FORCE_FIRST_ORDER = None                    # plain policy: free first pick
        vv = shapley.conditional_tables(val, M, md)
        QPR.EXACT_DP, QPR.STOP_BCE, QPR.RES_L2, QPR.BW_K = False, 0.0, 0.0, 0.0
        QPR.CKPT_SUFFIX, QPR.ADVICE_MODE, QPR.ABLATION = '', 'conditional', 'full'
        QPR.W_ADV = 1.0
        for seed in SEEDS:
            ck = QPR._ckpt_path(ds, fold, seed, lam, QPR.P_TRAIN, QPR.RESAMPLE)
            if not os.path.exists(ck):
                print(f'  MISS {ds} fold{fold} seed{seed} lam{lam}', flush=True)
                continue
            net, tabs = QPR.train_or_load(val, M, md, dataset=ds, fold=fold, seed=seed,
                                          lam=lam, p_train=QPR.P_TRAIN,
                                          resample=QPR.RESAMPLE, tables=vv,
                                          force_retrain=False)
            for cond, p in CONDITIONS:
                # identical rng discipline to run_design.run_rl
                av_te = None if p == 1.0 else rl.sample_availability(
                    Nt, M, p, np.random.default_rng(1000 + fold))
                av_va = None if p == 1.0 else rl.sample_availability(
                    Nv, M, p, np.random.default_rng(3000 + fold))
                rng_e = np.random.default_rng(2000 + seed)
                f1, macq, g, _s, _f = QPR.rollout(net, test, M, md, tabs, delta=delta,
                                                  avail=av_te, eps=0.0, rng=rng_e)
                vf1, vmacq, _g, _s, _f = QPR.rollout(
                    net, val, M, md, tabs, delta=delta, avail=av_va, eps=0.0,
                    rng=np.random.default_rng(4000 + seed))
                nA = float(M if av_te is None else av_te.sum(1).mean())
                rows.append(dict(dataset=ds, design='dqn_qprior', fold=fold, seed=seed,
                                 condition=cond, lever_name='lambda_delta',
                                 lever_value=lam, delta=delta,
                                 macro_f1=f1, mean_acquired=macq, mean_available=nA,
                                 ratio=macq / max(1e-9, nA), mean_gflops=g,
                                 macro_f1_val=vf1, ratio_val=vmacq / max(1e-9, nA),
                                 is_bstar=True, M=M))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--delta', type=float, required=True)
    ap.add_argument('datasets', nargs='*', default=list(FOLDS))
    a = ap.parse_args()
    out_dir = os.path.join(HERE, f'results_{RUNSET}')
    for ds in a.datasets:
        try:
            rows = run(ds, a.lam, a.delta)
            if not rows:
                print(f'SKIP {ds}: no rows', flush=True)
                continue
            p = os.path.join(out_dir, f'dqn_qprior_lever_{ds}_point.csv')
            pd.DataFrame(rows).to_csv(p, index=False)
            d = pd.DataFrame(rows)
            s = d[d.condition == 'A0']
            print(f'{ds}: {len(rows)} rows -> {os.path.basename(p)}   '
                  f'A0 F1={s.macro_f1.mean():.4f} ratio={s.ratio.mean():.3f} '
                  f'GF={s.mean_gflops.mean():.3f}', flush=True)
        except Exception as e:
            print(f'SKIP {ds}: {type(e).__name__}: {str(e)[:150]}', flush=True)


if __name__ == '__main__':
    main()
