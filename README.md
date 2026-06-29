# Efficient Multimodal Bottleneck

Code for **efficient multimodal time-series fusion** built on a **sequential gated-bottleneck**
backbone (`seqA`) with **modality-level early exit**, studied under **cross-subject distribution
shift**. The central thesis: per-sample routing/early-exit signals largely **collapse to a static
schedule under shift**, and **distillation is the lever** that makes modality pruning/early-exit
actually save compute.

Datasets: IEMOCAP, DaliaHAR, PAMAP2, DSADS, EAV, CMU-MOSEI.

## Repository structure

```
multimodal_model/    Model architectures (importable package)
                       - the seqA sequential gated-bottleneck fusion model (v6_downsample…seqfusion)
                       - baseline architectures: DMBF, L2R, MMEE, CREMA, AdaMML, DynMM (+ faithful
                         original-DynMM: dynmm_orig_baseline.py), ShaSpec, MultiModN, DecAlign, …
data/                Dataset loaders (IEMOCAP / CMU-MOSEI / EAV / HAR via dataset_builder, SAX, etc.)
utils/               Configs, helpers, perf/FLOP utilities
selector/            Modality-selection / routing components
common/              Shared training helpers (ee_data.py loader factory, compute_vlo_vs_random.py,
                     qmf_pdf_late_fusion.py) and config files (dynmm_branches.json, …)

training/            Core seqA training scripts (main_v6_*_seqfusion_late_fusion.py, per dataset)
baselines/           Baseline-method training scripts (main_<method>_*.py)
analysis/            Analysis + plotting: adaptive early-exit curves, LSMI R/U/S interaction
                     estimation, Shapley / val-LOO pruning, GFLOPs–accuracy frontiers, aggregation
scripts/             Shell orchestrators (run_*.sh): multi-GPU, multi-fold/seed, resume-guarded
external_baselines/  Vendored reference implementations (CREMA, multi-modal-early-exit)

results/
  figures/           Key plots (GFLOPs–accuracy frontiers, exit-scheme comparisons, …)
  tables/            Key result JSONs / summary tables (aggregated metrics, exit curves)
model_chkpt/         Pretrained IEMOCAP V6+TierA checkpoint + Shapley bundle (~78 MB, whitelisted)
environment.yml      Conda environment (PyTorch 2.5.1 / CUDA 12.1 / Python 3.10)
```

## Notes
- This repo contains **code + key results/plots only** — trained checkpoints (`*.pth`), packed
  data (`*.npz`/`*.pkl`), and logs are excluded (see `.gitignore`); the one exception is the
  whitelisted IEMOCAP checkpoint bundle under `model_chkpt/`.
- Scripts use absolute dataset paths and the `maestro` conda env from the original working tree;
  adjust paths and add the repo root to `PYTHONPATH` (helper imports moved to `common/`) to run.

## Setup
```bash
conda env create -f environment.yml -n maestro
conda activate maestro
```
