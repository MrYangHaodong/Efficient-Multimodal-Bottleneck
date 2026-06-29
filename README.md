# Efficient Multimodal Bottleneck

Code for **efficient multimodal time-series fusion** built on a **sequential gated-bottleneck**
backbone (`seqA`) with **modality-level early exit**, studied under **cross-subject distribution
shift**. The central thesis: per-sample routing/early-exit signals largely **collapse to a static
schedule under shift**, and **distillation is the lever** that makes modality pruning / early-exit
actually save compute.

Datasets: IEMOCAP, DaliaHAR, PAMAP2, DSADS, EAV, CMU-MOSEI.

## Repository structure

```
multimodal_model/        Model architectures (importable package)
                           - seqA sequential gated-bottleneck fusion (v6_downsample…seqfusion)
                           - baselines: DMBF, L2R, MMEE, CREMA, AdaMML, DynMM (+ faithful original
                             DynMM: dynmm_orig_baseline.py), ShaSpec, MultiModN, DecAlign, …
data/                    Dataset loaders (IEMOCAP / CMU-MOSEI / EAV / HAR; SAX tokenization, …)
utils/                   Configs, helpers, perf/FLOP utilities
selector/                Modality-selection / routing components
common/                  Shared training helpers (ee_data.py loader factory,
                         compute_vlo_vs_random.py, qmf_pdf_late_fusion.py) + configs

training/
  seqA/                  Core seqA training scripts (main_v6_*_seqfusion_late_fusion.py, per dataset)
  baselines/             Baseline-method training scripts (main_<method>_*.py)
analysis/                Adaptive early-exit curves, LSMI R/U/S interaction estimation,
                         val-LOO / Shapley pruning, GFLOPs–accuracy frontiers, aggregation
scripts/                 Shell orchestrators (run_*.sh): multi-GPU, multi-fold/seed, resume-guarded
external_baselines/      Vendored reference implementations (CREMA, multi-modal-early-exit)

results/
  figures/               Key plots (GFLOPs–accuracy frontiers, exit-scheme comparisons, …)
  tables/                Key result JSONs / summary tables (aggregated metrics, exit curves)
environment.yml          Conda environment (PyTorch 2.5.1 / CUDA 12.1 / Python 3.10)
```

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
