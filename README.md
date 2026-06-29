# Efficient Multimodal Bottleneck

Code for **efficient multimodal time-series fusion** built on a **sequential gated-bottleneck**
backbone (`seqA`) with **modality-level early exit**, studied under **cross-subject distribution
shift**. The central thesis: per-sample routing/early-exit signals largely **collapse to a static
schedule under shift**, and **distillation is the lever** that makes modality pruning / early-exit
actually save compute.

Datasets: IEMOCAP, DaliaHAR, PAMAP2, DSADS, EAV, CMU-MOSEI.

## Repository structure

```
models/                  ONE file per model (11): seqA.py (sequential gated-bottleneck fusion) +
                         dmbf, l2r, mmee, crema, adamml, dynmm, shaspec, multimodn, decalign,
                         crossattn. (+ helpers fusion_bmm_parallel.py, crossmodal_transformer.py)
training/                ONE trainer per model (11): train each on all 6 datasets via
                         `--dataset {iemocap,daliahar,pamap2,dsads,eav,mosei}`. e.g.
                         `python training/seqA.py --dataset iemocap --fold 0 --results_dir runs/`
data/                    Dataset loaders (IEMOCAP / CMU-MOSEI / EAV / HAR; SAX tokenization, …)
utils/                   Configs, helpers, perf/FLOP utilities
selector/                Modality-selection / routing components
common/                  Shared loader factory (ee_data.build_loaders — unifies all 6 datasets,
                         float + SAX), loader_utils, compute_vlo_vs_random, dynmm_branches.json, …
analysis/                Adaptive early-exit curves, LSMI R/U/S interaction estimation,
                         val-LOO / Shapley pruning, GFLOPs–accuracy frontiers, aggregation
scripts/                 Shell orchestrators (run_*.sh): multi-GPU, multi-fold/seed, resume-guarded
external_baselines/      Vendored reference implementations (CREMA, multi-modal-early-exit)
results/
  figures/               Key plots (GFLOPs–accuracy frontiers, exit-scheme comparisons, …)
  tables/                Key result JSONs / summary tables (aggregated metrics, exit curves)
environment.yml          Conda environment (PyTorch 2.5.1 / CUDA 12.1 / Python 3.10)
```

Every model is one `models/<m>.py` + one `training/<m>.py`; all trainers share
`common.ee_data.build_loaders` (one data pipeline for all 6 datasets).

## Notes
- This repo is **code + key results/plots only** — trained checkpoints (`*.pth`), packed data
  (`*.npz`/`*.pkl`), archives, and logs are excluded (see `.gitignore`).
- Scripts use absolute dataset paths and the `maestro` conda env from the original working tree;
  adjust paths and add the repo root to `PYTHONPATH` (shared helpers live in `common/`) to run.

## Setup
```bash
conda env create -f environment.yml -n maestro
conda activate maestro
```
