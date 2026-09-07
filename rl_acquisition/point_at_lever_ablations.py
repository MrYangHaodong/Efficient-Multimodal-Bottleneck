"""Roll out all 5 RL ablation variants at fixed (lambda=0.5, delta=0.1, res_l2=0.1).

Variants:
  full       — conditional-Shapley prior (the full policy, reference)
  zeroq      — zero-initialized Q-network
  randorder  — random-ordering initialized prior
  nostop     — no STOP action; always acquires all M modalities (tau=1.01)
  marginal   — marginal-Shapley advice (fullshap)
  marginal_randorder — marginal-Shapley + random order

Output: results_k16/ablation_lever_<ds>_point.csv  (one per dataset)
        results_k16/ablation_lever_summary.csv       (aggregated for CSV update)

Usage:
  RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16 \\
    PYTHONPATH=/home/group/maestro_visual/Efficient-Multimodal-Bottleneck \\
    python point_at_lever_ablations.py [dataset ...]
"""
import os, sys
import numpy as np
import pandas as pd

import environment as env
import rl_core as rl
import shapley
import dqn_policy_qprior as QPR
import dqn_policy_qprior_noexit as NOEXIT

HERE   = os.path.dirname(os.path.abspath(__file__))
RUNSET = os.environ.get('RLB_RUNSET', 'k16')
OUT    = os.path.join(HERE, f'results_{RUNSET}')

LAM   = 0.5
DELTA = 0.1
RES_L2 = 0.1

FOLDS = {'iemocap': [0,1,2], 'cmi': [0,1,2], 'czu_mhad': [0,1,2], 'mmfi': [0,1,2],
         'dsads': [0,1,2,3], 'eav': [0,1,2], 'pamap2': [0,1,2,3], 'utd_mhad': [0,1,2]}
SEEDS = [0, 1, 2]
CONDITIONS = [('A0', 1.0), ('A20', 0.8), ('A40', 0.6)]

# Per-dataset W_ADV for marginal (fullshap) — read from existing checkpoint filenames
W_ADV_MAP = {
    'iemocap':  2.95,
    'mmfi':     0.58,
    'cmi':      3.87,
    'czu_mhad': 0.85,
    'dsads':    4.89,
    'eav':      2.06,
    'pamap2':   2.15,
    'utd_mhad': 1.73,
}

VARIANTS = [
    # (key, label, ablation, advice_mode, use_nostop)
    ('full',             'full (Ours) — RL policy',         'full',      'conditional', False),
    ('zeroq',            'zero-initialized Q-network',       'zeroq',     'conditional', False),
    ('randorder',        'random-ordering initialized',      'randorder', 'conditional', False),
    ('nostop',           'no stop action',                   'full',      'conditional', True),
    ('marginal',         'marginal-Shapley advice (fullshap)', 'full',    'marginal',    False),
    ('marginal_randorder','marginal-Shapley + random order', 'randorder', 'marginal',    False),
]


def _set_qpr_globals(ablation, advice_mode, w_adv):
    QPR.EXACT_DP   = False
    QPR.STOP_BCE   = 0.0
    QPR.BW_K       = 0.0
    QPR.RES_L2     = RES_L2
    QPR.CKPT_SUFFIX = ''
    QPR.ABLATION   = ablation
    QPR.ADVICE_MODE = advice_mode
    QPR.W_ADV      = w_adv
    QPR.W_COST     = 1.0
    QPR.W_CONF     = 1.0


def run_variant(ds, ablation, advice_mode, use_nostop, w_adv):
    _set_qpr_globals(ablation, advice_mode, w_adv)
    rl.FORCE_FIRST_ORDER = None

    rows = []
    md = env.depth_of(ds)
    for fold in FOLDS[ds]:
        val  = env.get_cache('val',  ds, fold)
        test = env.get_cache('test', ds, fold)
        M    = val['M']
        Nt, Nv = test['softmax'].shape[1], val['softmax'].shape[1]

        if advice_mode == 'marginal':
            # Marginal Shapley tables: built from val-as-fit, same as run_design.py §2
            fit = val
            Cc  = fit['softmax'].shape[-1]
            vv  = shapley.value_val_perclass(
                shapley.shapley_per_sample(fit, M, md), fit['y'], M, Cc)
        else:
            vv = shapley.conditional_tables(val, M, md)

        for seed in SEEDS:
            ck = QPR._ckpt_path(ds, fold, seed, LAM, QPR.P_TRAIN, QPR.RESAMPLE)
            if not os.path.exists(ck):
                print(f'  MISS {ds} fold{fold} seed{seed} lam{LAM} '
                      f'abl={ablation} adv={advice_mode}  -> {ck}', flush=True)
                continue
            net, tabs = QPR.train_or_load(val, M, md, dataset=ds, fold=fold, seed=seed,
                                          lam=LAM, p_train=QPR.P_TRAIN,
                                          resample=QPR.RESAMPLE, tables=vv,
                                          force_retrain=False)
            for cond, p in CONDITIONS:
                av_te = None if p == 1.0 else rl.sample_availability(
                    Nt, M, p, np.random.default_rng(1000 + fold))
                av_va = None if p == 1.0 else rl.sample_availability(
                    Nv, M, p, np.random.default_rng(3000 + fold))
                rng_e = np.random.default_rng(2000 + seed)

                if use_nostop:
                    # No STOP action: always acquire all available modalities
                    f1, macq, g, _sn, _fi = NOEXIT.rollout(
                        net, test, M, md, tabs, tau=1.01,
                        avail=av_te, eps=0.0, rng=rng_e)
                    vf1, vmacq, _g, _s, _f = NOEXIT.rollout(
                        net, val, M, md, tabs, tau=1.01,
                        avail=av_va, eps=0.0,
                        rng=np.random.default_rng(4000 + seed))
                else:
                    f1, macq, g, _sn, _fi = QPR.rollout(
                        net, test, M, md, tabs, delta=DELTA,
                        avail=av_te, eps=0.0, rng=rng_e)
                    vf1, vmacq, _g, _s, _f = QPR.rollout(
                        net, val, M, md, tabs, delta=DELTA,
                        avail=av_va, eps=0.0,
                        rng=np.random.default_rng(4000 + seed))

                nA = float(M if av_te is None else av_te.sum(1).mean())
                rows.append(dict(
                    dataset=ds, fold=fold, seed=seed, condition=cond,
                    macro_f1=f1, mean_acquired=macq,
                    mean_available=nA, ratio=macq / max(1e-9, nA),
                    mean_gflops=g,
                    macro_f1_val=vf1, ratio_val=vmacq / max(1e-9, nA),
                ))
    return rows


def aggregate(rows, cond):
    """Mean/std F1 across folds×seeds, mean GFLOP and MU at a condition."""
    if not rows:
        return None, None, None, None
    df = pd.DataFrame(rows)
    if 'condition' not in df.columns:
        return None, None, None, None
    df = df[df.condition == cond]
    if df.empty:
        return None, None, None, None
    # Average seeds within each fold first, then compute fold-mean and fold-std
    by_fold = df.groupby('fold').macro_f1.mean()
    mean_f1 = by_fold.mean()
    std_f1  = by_fold.std(ddof=1) if len(by_fold) > 1 else 0.0
    mean_gf = df.mean_gflops.mean()
    mean_mu = df.ratio.mean()
    return mean_f1, std_f1, mean_gf, mean_mu


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else list(FOLDS)
    os.makedirs(OUT, exist_ok=True)

    summary_rows = []

    for ds in datasets:
        print(f'\n{"="*60}\n  {ds.upper()}\n{"="*60}', flush=True)
        all_variant_rows = {}

        for vkey, vlabel, ablation, advice_mode, use_nostop in VARIANTS:
            w_adv = W_ADV_MAP[ds] if advice_mode == 'marginal' else 1.0
            print(f'  [{vkey}]', flush=True)
            try:
                rows = run_variant(ds, ablation, advice_mode, use_nostop, w_adv)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'  !! {vkey}/{ds} FAILED: {e}', flush=True)
                rows = []
            all_variant_rows[vkey] = rows

            for cond in ['A0', 'A40']:
                mf1, sf1, mgf, mmu = aggregate(rows, cond)
                if mf1 is not None:
                    print(f'    {cond}: F1={mf1:.4f}±{sf1:.4f}  GF={mgf:.3f}  MU={mmu:.2f}',
                          flush=True)
                    summary_rows.append(dict(
                        dataset=ds, variant=vkey, label=vlabel,
                        condition=cond, lam=LAM, delta=DELTA,
                        mean_f1=mf1, std_f1=sf1,
                        mean_gflops=mgf, mean_mu=mmu,
                    ))

            # Save per-dataset per-variant CSV
            if rows:
                p = os.path.join(OUT, f'ablation_lever_{ds}_{vkey}_point.csv')
                pd.DataFrame(rows).to_csv(p, index=False)

    # Write summary
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(os.path.join(OUT, 'ablation_lever_summary.csv'), index=False)
    print(f'\nSummary -> {OUT}/ablation_lever_summary.csv', flush=True)
    return sdf


if __name__ == '__main__':
    main()
