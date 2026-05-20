#!/usr/bin/env bash
# Sequential ShapDistill sweep over 3 folds for a single target_k value.
# Designed to run on one free GPU while the main sweep uses others.
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/egg8711/miniconda3/envs/maestro/bin/python
TARGET_K=${TARGET_K:-2}
GPU=${GPU:-3}
EXP_TAG=${EXP_TAG:-v6_shapley_distill_kl_tk${TARGET_K}}
EXP=./model_chkpt/IEMOCAP/multimodal_model/2026-05-17_IEMOCAP_v6_fusion_only_dyna

mkdir -p logs

for FOLD in 0 1 2; do
  "$PY" -u script/main_v6_iemocap_selector_finetune_shapley.py \
    --fold="$FOLD" --cuda_pick="cuda:${GPU}" \
    --frozen_ckpt="$EXP/best_model_fold${FOLD}.pth" \
    --shapley_train_npz="$EXP/shapley_train_fold${FOLD}_phi.npz" \
    --target_k="$TARGET_K" --selector_downsample_factor=4 \
    --tau=0.5 --lambda_kl=1.0 --lambda_probe=0.05 \
    --num_epochs=30 --batch_size=32 --lr=3e-4 \
    --num_workers=4 \
    --results_dir=./model_chkpt/IEMOCAP/selector/ \
    --exp_name="$EXP_TAG" \
    > "logs/iemocap_shapdistill_${EXP_TAG}_fold${FOLD}.log" 2>&1
  echo "[$(date +%T)] fold $FOLD done"
done
echo "All 3 folds done for target_k=$TARGET_K"
