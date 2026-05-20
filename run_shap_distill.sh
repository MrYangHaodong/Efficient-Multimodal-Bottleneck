#!/usr/bin/env bash
# Shapley-distilled selector on DaliaHAR (frozen 5.14 dyna fusion).
# 3 folds in parallel on cuda:0/1/2.
set -uo pipefail

cd "$(dirname "$0")"
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
FUSION_DIR=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna
RESULTS_DIR=./model_chkpt/DaliaHAR/selector/
TAU=${TAU:-0.5}
LAMBDA_KL=${LAMBDA_KL:-1.0}
LAMBDA_CE=${LAMBDA_CE:-0.0}
LAMBDA_CE_SOFT=${LAMBDA_CE_SOFT:-0.0}
NUM_EPOCHS=${NUM_EPOCHS:-30}
TAG=${TAG:-tau${TAU/./p}}
EXP=${EXP_NAME:-v6_shapley_distill_ds4_${TAG}}
mkdir -p logs
echo "Running EXP=$EXP  tau=$TAU  lambda_kl=$LAMBDA_KL  lambda_ce=$LAMBDA_CE  lambda_ce_soft=$LAMBDA_CE_SOFT"

run_fold () {
  local FOLD="$1"
  local GPU="$2"
  "$PY" script/main_v6_daliahar_selector_finetune_shapley.py \
    --fold="$FOLD" --cuda_pick="cuda:${GPU}" \
    --frozen_ckpt="${FUSION_DIR}/best_model_fold${FOLD}.pth" \
    --shapley_train_npz="${FUSION_DIR}/shapley_train_fold${FOLD}_phi.npz" \
    --target_k=4 --selector_downsample_factor=4 \
    --tau="$TAU" --lambda_kl="$LAMBDA_KL" \
    --num_epochs="$NUM_EPOCHS" --batch_size=64 \
    --lr=3e-4 --lr_schedule=cosine --lambda_probe=0.05 \
    --lambda_ce="$LAMBDA_CE" --lambda_ce_soft="$LAMBDA_CE_SOFT" \
    --exp_name="$EXP" --results_dir="$RESULTS_DIR" \
    > "logs/shap_distill_${EXP}_fold${FOLD}.log" 2>&1
}

run_fold 0 0 &
run_fold 1 1 &
run_fold 2 2 &
wait
echo "All 3 Shapley-distill folds finished."
