#!/usr/bin/env bash
# MultiModN sequential-module baseline — d_model=128 fair config.
#
# Arch: d_model 128, nhead 8, 6 encoder / 4 fusion / 2 predictor layers.
# Dropout curriculum: warmup 10 / ramp 20 / max 0.4, 40 epochs, random order.
#
# Usage:
#   bash script/run/train_multimodn.sh <dataset> [GPU_ID]
#   bash script/run/train_multimodn.sh iemocap 0
#
# Datasets: iemocap, cmi, czu_mhad, mmfi, utd_mhad, eav, dsads
# Folds are run sequentially on the chosen GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
RESULTS="${RESULTS:-$REPO/model_chkpt/baselines}"

DATASET="${1:?usage: bash script/run/train_multimodn.sh <dataset> [GPU_ID]}"
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

for FOLD in $FOLDS; do
  echo "==== multimodn $DATASET fold$FOLD -> GPU $GPU ===="
  "$PYTHON" "$REPO/script/training/multimodn.py" \
    --dataset "$DATASET" --fold "$FOLD" \
    "${DATA_ARGS[@]}" \
    --d_model 128 --nhead 8 --num_layers 6 \
    --num_layers_fus 4 --num_layers_pred 2 \
    --num_epochs 40 --drop_warmup 10 --drop_ramp 20 --max_drop_prob 0.4 \
    --seq_random_order \
    --lr 1e-4 --batch_size "$BS" --num_workers 4 --cuda_pick cuda:0 \
    --results_dir "$RESULTS" --exp_name multimodn
done

echo "==== DONE: multimodn $DATASET all folds ===="
