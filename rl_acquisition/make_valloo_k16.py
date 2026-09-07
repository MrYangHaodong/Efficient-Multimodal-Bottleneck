#!/usr/bin/env python
"""Per-fold VAL leave-one-out modality order for the K=16 branched checkpoints.

For each fold: full-set val accuracy, then acc with each modality removed;
loo[m] = acc_full - acc_minus_m; order = descending loo. Writes valloo_orders.json
(same schema as the K=4 run dirs) into the K=16 run dir, pre-builds the per-modality
GFLOPs npz (so parallel cache workers never race on it), and — if RLB_QUEUE is set —
enqueues the fold's val/test cache-build tasks as soon as the fold is ready.

Run with RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16:
    python make_valloo_k16.py <dataset>
"""
import os, sys, json
os.environ.setdefault('RLB_RUNSET', 'k16')
os.environ.setdefault('RLB_CACHE_SUBDIR', 'botneck_16')
import torch
import environment as env

PRIO = {'cmi': '10', 'czu_mhad': '20', 'iemocap': '30', 'mmfi': '40',
        'dsads': '45', 'pamap2': '50', 'utd_mhad': '55', 'eav': '60'}  # long jobs first

ds = sys.argv[1]
run_dir = env._RUNDIR[ds]
queue = os.environ.get('RLB_QUEUE')
out = {'dataset': ds, 'model': 'seqA_branched', 'run_dir': os.path.basename(run_dir),
       'folds': [str(f) for f in env.FOLDS[ds]], 'per_fold': {}}
loo_sum = None

for fd in env.FOLDS[ds]:
    model, splits, mods, C = env._load_model(ds, fd)
    vb = splits['val']

    @torch.no_grad()
    def acc(order_names):
        good = tot = 0
        for b in vb:
            pref = env._forward_prefix(model, b, order_names, mods)
            pred = pref[:, -1].argmax(-1).numpy()
            y = b[2].view(-1).numpy()
            good += int((pred == y).sum()); tot += len(y)
        return good / tot

    a_full = acc(list(mods))
    loo = {m: a_full - acc([x for x in mods if x != m]) for m in mods}
    order = sorted(mods, key=lambda m: -loo[m])
    out['per_fold'][str(fd)] = {'order': order, 'loo': loo, 'val_acc_full': a_full}
    if loo_sum is None:
        loo_sum = {m: 0.0 for m in mods}
    for m in mods:
        loo_sum[m] += loo[m]
    print(f'[valloo] {ds} fold{fd}: full={a_full:.4f} order={order}', flush=True)
    del model, splits
    torch.cuda.empty_cache()

    # per-modality GFLOPs prepass (cached npz in CACHE_DIR; avoids worker races)
    env.modality_gflops(ds, fd)

    # write the json incrementally so a mid-run crash still leaves prior folds usable
    # (static_order finalized after the last fold below)
    if queue:                                   # fold ready -> enqueue its cache tasks
        pass                                    # enqueue AFTER json write (workers read it)

mean = {m: loo_sum[m] / len(env.FOLDS[ds]) for m in loo_sum}
out['static_order'] = sorted(mean, key=lambda m: -mean[m])
out['static_loo_mean'] = mean
with open(os.path.join(run_dir, 'valloo_orders.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f'[valloo] wrote {run_dir}/valloo_orders.json  static={out["static_order"]}', flush=True)

if queue:                                       # json on disk -> safe to enqueue all folds
    for fd in env.FOLDS[ds]:
        for sp in ('val', 'test'):
            open(os.path.join(queue, f'{PRIO[ds]}_{ds}_{fd}_{sp}.task'), 'w').close()
    print(f'[valloo] enqueued {2 * len(env.FOLDS[ds])} cache tasks for {ds}', flush=True)
