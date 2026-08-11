# measurement/ — latency measurement for seqA+RL vs adaptive-multimodal baselines

Self-contained latency + F1 benchmarks for **seqA+RL** against **AdaMML**,
**DynMM**, and **DyMo** across 7 datasets, on GPU and CPU, with and without
data-loading cost included.

Each measurement is **one self-contained Python file**. Every script inlines the
full harness — dataset config, model loaders, RL policy, stacked-BMM encoder,
walk variants, baseline forwards — so no script imports another. The only local
import is the `models/` package (the trained architectures, ~4.7k lines, too
large to inline).

| Script | What it measures |
|---|---|
| `bench_gpu.py` | GPU latency + F1, compute only |
| `bench_cpu.py` | CPU latency + F1, compute only, with `--threads` pinning |
| `bench_loading.py` | Latency **including** disk → RAM → GPU loading |
| `sweep_delta.py` | Accuracy/compute frontier as the RL stop bias δ varies |
| `bench_fp16.py` | Does fp16 or an entropy early exit help at batch size 1? |
| `prepare_samples.py` | *(supporting)* writes the per-modality sample files `bench_loading.py` reads |

---

## The two timing models

This is the distinction that matters most when reading results.

**Compute only** (`bench_gpu.py`, `bench_cpu.py`, `sweep_delta.py`,
`bench_fp16.py`) — every tensor is moved to the device *before* the timer
starts. A timed call measures model forward time and nothing else. This isolates
architecture cost.

**Loading inclusive** (`bench_loading.py`) — the timer starts *before* anything
is read from disk, so each call measures the full pipeline a deployed system
pays: `np.load` → `.to(device)` → forward.

The second is where sequential acquisition earns its keep. AdaMML, DynMM and
DyMo all need every modality present before inference begins, so they pay M file
reads per sample. seqA+RL needs only one modality to start: the next file is not
read until the policy has decided it wants that modality, and files it never
acquires are never read at all.

`bench_loading.py` therefore runs seqA+RL in two modes so the effect is isolated
cleanly — both produce identical routes and F1, differing only in **when** data
arrives:

- `sequential` — read each modality on demand, interleaved with the RL
  decisions. Only acquired modalities are ever read.
- `bundled` — read all M modalities upfront, then walk. Pays the same I/O as the
  baselines, so `bundled − sequential` is exactly what sequential loading buys.

---

## Running

```bash
cd runs/measurement
export PYTHONPATH=/home/group/maestro_visual

# GPU, compute only
CUDA_VISIBLE_DEVICES=0 python bench_gpu.py --n 50
CUDA_VISIBLE_DEVICES=0 python bench_gpu.py --n 0 --variant both   # full test sets

# CPU, 8 threads
CUDA_VISIBLE_DEVICES="" python bench_cpu.py --n 50 --threads 8

# Loading-inclusive (prepare samples once first)
python prepare_samples.py --n 50
CUDA_VISIBLE_DEVICES=0 python bench_loading.py --n 50 --mode both

# Operating point for one dataset
CUDA_VISIBLE_DEVICES=0 python sweep_delta.py --datasets utd_mhad --n 200

# fp16 / early-exit study
CUDA_VISIBLE_DEVICES=0 python bench_fp16.py --datasets utd_mhad --n 200
```

Common flags: `--n` samples per fold (`0` = full test set), `--warmup`,
`--datasets`, `--out`. Results are written to `results_*.json`.

`prepare_samples.py --n 50` writes ~7 GB and is plenty for latency, which is
stable across samples. Use `--n 0` for full test sets when F1 also needs to be
trustworthy.

---

## Walk variants

Both produce **identical routes and F1** — they differ only in how the encoder
is scheduled.

- **A0** — encode + fuse each acquired modality sequentially *inside* the RL
  loop. This is what a real sequential-acquisition system pays: a modality is
  only encoded once the policy has decided to acquire it. Required by
  `bench_loading.py`, since only A0 can interleave I/O with decisions.
- **C2** — pre-encode all M modalities upfront in one grouped-BMM pass
  (`StackedBmmEncoder` stacks per-branch weights into `[M, D, D]` tensors), then
  run a fuse-only walk that just indexes the precomputed tokens. Faster when the
  policy acquires most modalities anyway; wasteful when it stops early.

`build_stacked_encoder` asserts parity against per-branch `encode()` on every
build, so a silent weight-stacking bug cannot pass unnoticed.

## The δ lever

`OURS_DELTA` is the test-time stop bias — the walk stops when
`Q[STOP] + δ ≥ max Q[acquire]`. Raising δ stops earlier (fewer modalities, lower
latency, usually lower F1); lowering it acquires more. It is a **deployment
lever on an already-trained policy**, so `sweep_delta.py` traces the whole
frontier without retraining.

## Operating points

Pinned in `DS_CFG` in each script, matching the comparison CSV:

| Dataset | δ | Folds | seqA checkpoint | RL policy |
|---|---|---|---|---|
| IEMOCAP | 0.1 | 3 | standard `pv3_branch_16` | `rl_policy_net_lam0.5/` |
| MM-Fi\* | 0.1 | 3 | rebalanced (2026-07-24) | `rl_policy_net_mmfi_rebal/` |
| CMI | 0.1 | 3 | standard | `rl_policy_net_lam0.5/` |
| CZU-MHAD | 0.1 | 3 | standard | `rl_policy_net_lam0.5/` |
| DSADS | 0.1 | 4 | standard | `rl_policy_net_k16p_lam0.5/` |
| EAV | 0.1 | 3 | standard | `rl_policy_net_k16p_lam0.5/` |
| UTD-MHAD\*\* | **0.0** | 3 | dropaug (2026-08-09) | `rl_policy_net_utd_mhad_dropaug/` |

\* MM-Fi uses the rebalanced backbone, which needs `num_fusion_distill=4` —
only the main repo's `SeqABranched` supports it, so `_build_seqa_mmfi_rebal`
temporarily shadows the local `models/` package to import it.

\*\* UTD-MHAD's δ=0.0 came from `sweep_delta.py`: F1 rose 0.869 → 0.914 for
+0.17 modalities and ~1.2 ms.

DynMM and DyMo checkpoints exist for 5 of the 7 datasets — they are skipped for
DSADS and EAV (`has_dynmm` / `has_dymo` in `DS_CFG`).

---

## Layout

```
measurement/
├── bench_gpu.py  bench_cpu.py  bench_loading.py
├── sweep_delta.py  bench_fp16.py  prepare_samples.py
├── models/       seqA, AdaMML, DynMM, DyMo architectures (copied)
├── common/       DynMM branch definitions (copied)
├── selector/     required by models/seqA.py (copied)
├── dymo_sg/      cached DyMo subset Gaussians (copied)
├── rl_policy_net_*/   trained RL policy bundles (copied)
├── checkpoints/  all trained weights (3.2 GB, real files — see below)
│   ├── seqA_mmfi_rebalanced/…            MM-Fi rebalanced backbone
│   └── dropout_robustness_seqA_dropaug/… UTD-MHAD dropaug backbone
├── samples/      per-modality .npy for bench_loading.py — NOT shipped, generate locally
└── data/         test tensors — NOT shipped, point DATA_DIR at your copy
```

This directory is fully self-contained: every checkpoint the suite loads lives
under `checkpoints/`, including the two seqA backbones that sat outside the tree
in the original layout (they are nested one level down so the `DsCfg` glob for
`*<ds>_pv3_branch_16_seqAbranch` still resolves to the standard checkpoints and
not to these).

**Not included: test data.** `data/` (the batched test tensors) and `samples/`
(the per-modality files `bench_loading.py` reads) are excluded — together they
are ~21 GB. `bench_gpu.py` / `bench_cpu.py` / `sweep_delta.py` / `bench_fp16.py`
need `data/`; `bench_loading.py` additionally needs `samples/`, which
`prepare_samples.py` generates from `data/`. Point `DATA_DIR` at your copy of
the tensors to run any of them.

Note the repo `.gitignore` excludes `*.pth`, `*.pt` and `*.npz`, so the weights
under `checkpoints/`, `dymo_sg/` and `rl_policy_net_*/` are present on disk but
untracked — a fresh clone gets the scripts, not the weights.

## Reference results

`results_loading_sequential_fulltest.json` — full test sets, sequential loading,
the run behind the headline numbers (geomean 1.14× vs AdaMML, 1.78× vs DynMM,
5.12× vs DyMo). `results_loading_bundled_50.json` — the earlier 50-sample
bundled-loading run, kept for comparison.
