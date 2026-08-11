#!/usr/bin/env bash
# Vanilla MBT (non-adaptive bottleneck fusion) — d_model=128 fair config.
#
# Arch: d_model 128, 6 layers, nhead 8, mlp_dim 128, fusion_layer 3, K=4
# bottlenecks. Dropout curriculum: warmup 10 / ramp 20 / max 0.4, 40 epochs.
#
# Usage:
#   bash script/run/train_mbt.sh <dataset> [GPU_ID]
#   bash script/run/train_mbt.sh iemocap 0
#
# Datasets: iemocap, cmi, czu_mhad, mmfi, utd_mhad, eav, dsads
# Folds are run sequentially on the chosen GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
RESULTS="${RESULTS:-$REPO/model_chkpt/baselines}"

DATASET="${1:?usage: bash script/run/train_mbt.sh <dataset> [GPU_ID]}"
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
  echo "==== mbt $DATASET fold$FOLD -> GPU $GPU ===="
  "$PYTHON" "$REPO/script/training/mbt.py" \
    --dataset "$DATASET" --fold "$FOLD" \
    "${DATA_ARGS[@]}" \
    --d_model 128 --num_layers 6 --nhead 8 --mlp_dim 128 \
    --fusion_layer 3 --n_bottlenecks 4 \
    --num_epochs 40 --warmup_epochs 10 --ramp_epochs 20 --max_drop_prob 0.4 \
    --lr 1e-4 --batch_size "$BS" --num_workers 4 --cuda_pick cuda:0 \
    --results_dir "$RESULTS" --exp_name mbt_$DATASET
done

echo "==== DONE: mbt $DATASET all folds ===="
