#!/usr/bin/env bash
# seqA branched bottleneck model (ours) — pv3_branch_16, K=16.
#
# mbtAvg base + prefix supervision with k-prop KD + EMA self-distillation
# teacher + modality-dropout curriculum. Arch: d_model 128, K=16 bottlenecks.
# Per-dataset seed / layer depth / schedule follow the published recipes.
#
# MOD_DROP defaults to 0.4 (the dropaug recipe). Set MOD_DROP=0.2 and
# N_FUSION_DISTILL=4 to reproduce the baselines_d128_fair variant instead.
#
# Usage:
#   bash script/run/train_seqA_branched.sh <dataset> [GPU_ID]
#   bash script/run/train_seqA_branched.sh iemocap 0
#
# Datasets: iemocap, cmi, czu_mhad, mmfi, utd_mhad, eav, dsads
# Folds are run sequentially on the chosen GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
RESULTS="${RESULTS:-$REPO/model_chkpt/baselines}"

DATASET="${1:?usage: bash script/run/train_seqA_branched.sh <dataset> [GPU_ID]}"
GPU="${2:-0}"

export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$RESULTS"

# ── per-dataset settings ─────────────────────────────────────────────────────
case "$DATASET" in
  iemocap) FOLDS="0 1 2"; SEQ=100; BS=32; WL=; SEED=239; NL=2; NLPM=2; WU=10; RAMP=20; EP=40;  EXTRA="--loss weighted_ce" ;;
  cmi) FOLDS="0 1 2"; SEQ=128; BS=32; WL=; SEED=9;   NL=2; NLPM=2; WU=10; RAMP=20; EP=40;  EXTRA="" ;;
  czu_mhad) FOLDS="0 1 2 3 4"; SEQ=128; BS=32; WL=; SEED=9;   NL=2; NLPM=2; WU=20; RAMP=40; EP=100; EXTRA="" ;;
  mmfi) FOLDS="0 1 2"; SEQ=128; BS=16; WL=; SEED=239; NL=2; NLPM=2; WU=20; RAMP=40; EP=100; EXTRA="" ;;
  utd_mhad) FOLDS="0 1 2"; SEQ=128; BS=32; WL=1; SEED=9;   NL=3; NLPM=3; WU=20; RAMP=40; EP=100; EXTRA="" ;;
  eav) FOLDS="0 1 2"; SEQ=128; BS=32; WL=1; SEED=169; NL=2; NLPM=2; WU=20; RAMP=40; EP=100; EXTRA="--lr 2e-5" ;;
  dsads) FOLDS="0 1 2 3"; SEQ=128; BS=32; WL=2; SEED=169; NL=3; NLPM=3; WU=20; RAMP=40; EP=100; EXTRA="" ;;
  *) echo "unknown dataset: $DATASET (expected one of: iemocap, cmi, czu_mhad, mmfi, utd_mhad, eav, dsads)" >&2; exit 1 ;;
esac

# float datasets use the resampled path; SAX HAR datasets use word_length.
DATA_ARGS=(--max_seq_len "$SEQ" --time_compression_ratio 4 --split_mode cross_subject)
[ -n "$WL" ] && DATA_ARGS+=(--word_length "$WL")

MOD_DROP="${MOD_DROP:-0.4}"
N_FUSION_DISTILL="${N_FUSION_DISTILL:-0}"

for FOLD in $FOLDS; do
  echo "==== seqA_branch $DATASET fold$FOLD -> GPU $GPU (seed $SEED) ===="
  "$PYTHON" "$REPO/script/training/seqA_branched.py" \
    --dataset "$DATASET" --fold "$FOLD" --seed_num "$SEED" \
    "${DATA_ARGS[@]}" $EXTRA \
    --fusion_mode sequential --seq_agg_mode mean --seq_random_order \
    --prefix_supervision --prefix_ds_weight 1.0 \
    --prefix_kd_weight 0.5 --prefix_kd_weight_mode k_prop \
    --phase2_ema_teacher --ema_decay 0.999 \
    --d_model 128 --num_layers "$NL" --num_layers_per_modal "$NLPM" \
    --n_bottlenecks 16 --n_fusion_distill "$N_FUSION_DISTILL" --base_factor 5 \
    --use_sparse_attn False --sparse_attn_variant opt --strat_block_size 8 \
    --downsample_min_len 4 --head_mode gap --use_batched_fusion False \
    --dropout_signal uniform --max_modality_drop "$MOD_DROP" \
    --warmup_epochs "$WU" --ramp_epochs "$RAMP" --num_epochs "$EP" \
    --lr 1e-4 --batch_size "$BS" --num_workers 4 --cuda_pick cuda:0 \
    --results_dir "$RESULTS" --exp_name pv3_branch_16
done

echo "==== DONE: seqA_branched $DATASET all folds ===="
