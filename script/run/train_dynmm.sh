#!/usr/bin/env bash
# DynMM branch-gated adaptive baseline — embed_dim=128.
#
# Two-phase: 30 expert epochs then 10 gate epochs. Branches: 1mod,full.
# NB: DynMM uses --embed_dim (not --d_model).
# Pass PHASE2_ONLY=1 to retrain only the gate on existing experts.
#
# Usage:
#   bash script/run/train_dynmm.sh <dataset> [GPU_ID]
#   bash script/run/train_dynmm.sh iemocap 0
#
# Datasets: iemocap, cmi, czu_mhad, mmfi, utd_mhad, eav, dsads
# Folds are run sequentially on the chosen GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
RESULTS="${RESULTS:-$REPO/model_chkpt/baselines}"

DATASET="${1:?usage: bash script/run/train_dynmm.sh <dataset> [GPU_ID]}"
GPU="${2:-0}"

export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$RESULTS"

# ── per-dataset settings ─────────────────────────────────────────────────────
case "$DATASET" in
  iemocap) FOLDS="0 1 2"; SEQ=100; BS=32; WL= ;;
  cmi) FOLDS="0 1 2"; SEQ=128; BS=32; WL= ;;
  czu_mhad) FOLDS="0 1 2 3 4"; SEQ=128; BS=32; WL= ;;
  mmfi) FOLDS="0 1 2"; SEQ=128; BS=16; WL= ;;
  utd_mhad) FOLDS="0 1 2"; SEQ=128; BS=32; WL=1 ;;
  eav) FOLDS="0 1 2"; SEQ=128; BS=32; WL=1 ;;
  dsads) FOLDS="0 1 2 3"; SEQ=128; BS=32; WL=2 ;;
  *) echo "unknown dataset: $DATASET (expected one of: iemocap, cmi, czu_mhad, mmfi, utd_mhad, eav, dsads)" >&2; exit 1 ;;
esac

# float datasets use the resampled path; SAX HAR datasets use word_length.
DATA_ARGS=(--max_seq_len "$SEQ" --time_compression_ratio 4 --split_mode cross_subject)
[ -n "$WL" ] && DATA_ARGS+=(--word_length "$WL")

# Gate-only retrain reuses the trained experts in $RESULTS.
PHASE_ARGS=(--expert_epochs 30 --gate_epochs 10)
[ "${PHASE2_ONLY:-0}" = "1" ] && PHASE_ARGS=(--phase2_only --gate_epochs 40 --reg 0.1 --tau 1.0)

for FOLD in $FOLDS; do
  echo "==== dynmm $DATASET fold$FOLD -> GPU $GPU ===="
  "$PYTHON" "$REPO/script/training/dynmm.py" \
    --dataset "$DATASET" --fold "$FOLD" \
    "${DATA_ARGS[@]}" "${PHASE_ARGS[@]}" \
    --embed_dim 128 --num_layers 6 --nhead 8 \
    --branches 1mod,full --resource_mode expensive \
    --drop_warmup 10 --drop_ramp 20 --max_drop_prob 0.4 \
    --lr 1e-4 --batch_size "$BS" --num_workers 4 --cuda_pick cuda:0 \
    --results_dir "$RESULTS" --exp_name dynmm_$DATASET
done

echo "==== DONE: dynmm $DATASET all folds ===="
