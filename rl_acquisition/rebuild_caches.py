"""EXPLICIT cache rebuild at MAX_DEPTH = M (protocol §1). Nothing else triggers a build.

    python rebuild_caches.py iemocap                        # val + test  (the default)
    python rebuild_caches.py cmi --splits train,val,test    # only if you really need train

ONLY val + test are built. NOTHING reads the train cache: the protocol fits on val (§2 — the
backbone memorises its train split, so train-cache predictions carry no transferable signal),
selects the lever on val, and reports on test. The train split is 62% of IEMOCAP's samples and
55% of CMI's, so building it would more than double the job for nothing.

GFLOPs is derived additively from the M per-modality constants in the existing d=4 cache
(|gflops[S] - sum_m c_m| < 3e-5, protocol §8.3) — no fvcore needed.
"""
import argparse
import time

import environment as env

ap = argparse.ArgumentParser()
ap.add_argument('dataset', nargs='?', default='iemocap')
ap.add_argument('--splits', default='val,test',
                help='default val,test — the train cache is never read (see module docstring)')
a = ap.parse_args()
ds = a.dataset
assert ds in env.FOLDS, f'unknown dataset {ds}; one of {list(env.FOLDS)}'
splits = [s.strip() for s in a.splits.split(',')]

t0 = time.time()
print(f'[rebuild] {ds}: depth {env.depth_of(ds)} (pv3 on disk is {env.PV3_DEPTH})  '
      f'splits={splits}  folds={env.FOLDS[ds]}', flush=True)
for fold in env.FOLDS[ds]:
    for split in splits:
        t = time.time()
        c = env.build_cache(split, ds, fold)
        print(f'[rebuild] {ds} fold{fold} {split}: {len(c["nodes"])} nodes  N={c["softmax"].shape[1]}'
              f'  {time.time()-t:.0f}s  (total {(time.time()-t0)/60:.1f} min)', flush=True)
print(f'[rebuild] {ds} DONE in {(time.time()-t0)/60:.1f} min', flush=True)
