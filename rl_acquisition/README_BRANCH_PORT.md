# RL_AAAI_branch — seqA_branched port of RL_AAAI_final

This is a copy of `RL_AAAI_final` with the RL **backbone swapped from the batched
sequential seqA to the branched model** `models.seqA_branched.SeqABranched`
(per-modality independent encoder+fusion branches, interleaved enc→fuse).

The RL *strategy is unchanged* — still `dqn_policy_qprior` (Shapley bootstraps Q,
the net learns the residual), same reward/cost/state, same split hygiene, same
baselines. Only the frozen backbone that produces the cached prefix-tree changed.

## What was modified (all in `environment.py`)

| touch point | batched (original) | branched (this copy) |
|---|---|---|
| checkpoints | `runs/…/*_pv3_mbtAvg_prefixup_randord/best_model_loo_fold{N}.pth` | `runs/dropout_robustness_seqA_dropaug/2026-07-18_{ds}_pv3_branch_seqAbranch/best_model_fold{N}.pth` |
| model build | `build_data(...)` → `DualVideoBottleneckModelV6Downsample` | `SeqABranched(...)` built from `results_fold{N}.json`; data still via `build_data` (high dict only) |
| prefix forward | toggle `model.model.bottleneck_fusion.seq_modality_order` + `return_prefix_logits` | `model(inputs, order=…, return_prefix=True)` |
| per-modality GFLOPs | read from batched pv3 d=4 cache | fvcore on the branched model, `{m}` alone (additive by construction) — cached to `cache/{ds}_fold{fd}_branch_modgflops.npz` |
| val-LOO order | from pv3 manifest | branched `valloo_orders.json` (`static_order`) in each run dir |
| cache source | `CACHE_SOURCE='pv3'` (reuse batched caches) | `CACHE_SOURCE='build'` — the batched pv3 caches cannot be reused (different backbone) |

Everything else — `rl_core.py`, `shapley.py`, `baselines.py`, `run_design.py`,
`dqn_policy_qprior*.py`, `runs.sh` — is untouched: they operate purely on the
cached prefix-tree, which is backbone-agnostic.

## Prerequisites

The branched checkpoints must exist (they do, from `training/seqA_branched.py`):
`runs/dropout_robustness_seqA_dropaug/2026-07-18_{ds}_pv3_branch_seqAbranch/` with
`best_model_fold{N}.pth`, `results_fold{N}.json`, `valloo_orders.json`, and
`shapley_orders.json`.

## Build the caches (required — the branched softmax/GFLOPs differ from batched)

The batched pv3 caches are NOT reused. Build the branched prefix-tree per dataset
(depth = M; this is the expensive step, ~15 min IEMOCAP … hours for CMI/CZU):

```
cd RL_AAAI_branch
python rebuild_caches.py iemocap     # builds val + test at depth M
python rebuild_caches.py cmi
# czu_mhad, mmfi likewise
```

Then run a dataset end-to-end exactly as before:

```
./runs.sh <dataset> <gpu>
```

## Notes / caveats

- **Shapley order** for the Q-prior advice is computed by `shapley.py` off the
  branched cache — it will differ from the batched model's Shapley (the branched
  Shapley we already measured: e.g. MM-Fi `depth > infra1 > lidar > mmwave > wifi`).
- **`seq_random_order=False`** and sparse-attn disabled at eval → the branched
  value function is deterministic (required for a stable cache).
- Branched per-modality GFLOPs are additive because each branch runs independently
  with a fixed-K bottleneck fusion; verified the marginal is position-independent.
- Verified working: model load (strict), prefix forward `(B,K,C)`, fvcore GFLOPs,
  val-LOO order, and all RL modules import against the branched environment.
