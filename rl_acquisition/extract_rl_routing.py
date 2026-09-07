#!/usr/bin/env python
"""Extract the trained dqn_qprior policy's REAL per-sample acquisition routing at A0
(0% missingness) for each (dataset, fold), and write a portable JSON per cell that
the energy_2 bundle consumes for its seqA+RL adaptiveness.

For every (ds, fold) we load the B*-selected (is_bstar) net + q-prior tables and run
a DETERMINISTIC (eps=0) rollout over the test cache.  Each test sample ends at a node
whose key is its ORDERED acquired-modality path (indices into cache['mods']).  We
aggregate the distinct ordered subsets and their frequencies — this is the real,
per-sample, data-dependent routing that replaces the old scalar k=round(mean_acquired).

Output: rl_routing/<ds>_fold<fold>.json
  {dataset, fold, M, mods, mean_acquired, mean_acquired_csv, mean_gflops,
   depth_hist:{d:count}, per_modality_freq:{mod:freq},
   subsets:[{order:[mod,...], count, freq}, ...]}   # sorted by count desc

  RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16 \
  PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES=0 python extract_rl_routing.py
"""
import os, sys, json, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('RLB_RUNSET', 'k16')
os.environ.setdefault('RLB_CACHE_SUBDIR', 'botneck_16')
sys.path.insert(0, HERE)
import environment as env          # noqa: E402
import dqn_policy_qprior as QPR    # noqa: E402
import rl_core as rl               # noqa: E402

ALL_DS = ['iemocap', 'cmi', 'czu_mhad', 'mmfi']
OUT = os.path.join(HERE, 'rl_routing')


def bstar_config(ds, fold):
    d = pd.read_csv(os.path.join(HERE, 'results_k16', f'dqn_qprior_{ds}_point.csv'))
    r = d[(d.is_bstar) & (d.condition == 'A0') & (d.fold == fold)].iloc[0]
    return int(r['seed']), float(r['lever_value']), float(r['mean_acquired']), \
        (float(r['mean_gflops']) if 'mean_gflops' in r else float('nan'))


def extract(ds, fold):
    val = env.get_cache('val', ds, fold)
    test = env.get_cache('test', ds, fold)
    M, md = val['M'], env.depth_of(ds)
    mods = list(test['mods'])
    seed, lam, macq_csv, g_csv = bstar_config(ds, fold)
    net, vv = QPR.train_or_load(val, M, md, dataset=ds, fold=fold, seed=seed,
                                lam=lam, tables=None)
    # deterministic (eps=0) deployable policy
    f1, macq, g, sn, first = QPR.rollout(net, test, M, md, vv, delta=0.0,
                                         avail=None, eps=0.0,
                                         rng=np.random.default_rng(0))
    nodes = test['nodes']
    depth, used, ch = rl.node_meta(nodes, M, md)
    keys = [nodes[int(s)] for s in sn]                       # ordered mod-index paths
    N = len(keys)

    def names(k):
        return [mods[int(i)] for i in k.split(',')] if k else []

    cnt = collections.Counter(keys)
    subsets = [{'order': names(k), 'count': int(c), 'freq': c / N}
               for k, c in cnt.most_common()]
    permod = collections.Counter()
    for k, c in cnt.items():
        for nm in names(k):
            permod[nm] += c
    per_mod_freq = {m: permod.get(m, 0) / N for m in mods}
    depth_hist = {int(dp): int(sum(1 for s in sn if depth[int(s)] == dp))
                  for dp in sorted(set(int(depth[int(s)]) for s in sn))}
    return dict(dataset=ds, fold=fold, M=int(M), mods=mods, n_samples=int(N),
                seed=seed, lam=lam, macro_f1=float(f1),
                mean_acquired=float(macq), mean_acquired_csv=float(macq_csv),
                mean_gflops=float(g), mean_gflops_csv=float(g_csv),
                depth_hist=depth_hist, per_modality_freq=per_mod_freq,
                subsets=subsets,
                note='deterministic (eps=0) rollout of is_bstar dqn_qprior policy at A0')


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in ALL_DS:
        for fold in env.FOLDS[ds]:
            try:
                r = extract(ds, fold)
                json.dump(r, open(os.path.join(OUT, f'{ds}_fold{fold}.json'), 'w'), indent=2)
                top = r['subsets'][0]
                print(f'  {ds:9s} f{fold}: N={r["n_samples"]} mean_acq={r["mean_acquired"]:.3f} '
                      f'(csv {r["mean_acquired_csv"]:.3f})  {len(r["subsets"])} distinct subsets; '
                      f'top={top["count"]}x[{",".join(top["order"])}]', flush=True)
            except Exception as e:
                print(f'  {ds} f{fold}: FAIL {str(e)[:100]}', flush=True)
    print(f'wrote rl_routing/ -> {OUT}')


if __name__ == '__main__':
    main()
