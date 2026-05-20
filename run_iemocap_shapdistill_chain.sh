#!/usr/bin/env bash
# Chain target_k=1 and target_k=6 sweeps on cuda:0/1/2 once the current
# target_k=4 sweep finishes.  Each GPU runs its own fold sequentially.
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/egg8711/miniconda3/envs/maestro/bin/python
FOLD="$1"
GPU="$2"
EXP=./model_chkpt/IEMOCAP/multimodal_model/2026-05-17_IEMOCAP_v6_fusion_only_dyna
EXP_DIR=./model_chkpt/IEMOCAP/selector

mkdir -p logs

# Wait for the current target_k=4 run to finish for THIS fold.
DEFAULT_TK4="${EXP_DIR}/2026-05-17_IEMOCAP_v6_shapley_distill_kl/results_fold${FOLD}.json"
echo "[$(date +%T)] cuda:${GPU} fold ${FOLD}: waiting for ${DEFAULT_TK4}"
while [ ! -f "$DEFAULT_TK4" ]; do
  sleep 30
done
echo "[$(date +%T)] cuda:${GPU} fold ${FOLD}: target_k=4 done; starting target_k=1"

# target_k=1
"$PY" -u script/main_v6_iemocap_selector_finetune_shapley.py \
  --fold="$FOLD" --cuda_pick="cuda:${GPU}" \
  --frozen_ckpt="$EXP/best_model_fold${FOLD}.pth" \
  --shapley_train_npz="$EXP/shapley_train_fold${FOLD}_phi.npz" \
  --target_k=1 --selector_downsample_factor=4 \
  --tau=0.5 --lambda_kl=1.0 --lambda_probe=0.05 \
  --num_epochs=30 --batch_size=32 --lr=3e-4 \
  --num_workers=4 \
  --results_dir="$EXP_DIR" \
  --exp_name=v6_shapley_distill_kl_tk1 \
  > "logs/iemocap_shapdistill_kl_tk1_fold${FOLD}.log" 2>&1
echo "[$(date +%T)] cuda:${GPU} fold ${FOLD}: target_k=1 done; starting target_k=6"

# target_k=6
"$PY" -u script/main_v6_iemocap_selector_finetune_shapley.py \
  --fold="$FOLD" --cuda_pick="cuda:${GPU}" \
  --frozen_ckpt="$EXP/best_model_fold${FOLD}.pth" \
  --shapley_train_npz="$EXP/shapley_train_fold${FOLD}_phi.npz" \
  --target_k=6 --selector_downsample_factor=4 \
  --tau=0.5 --lambda_kl=1.0 --lambda_probe=0.05 \
  --num_epochs=30 --batch_size=32 --lr=3e-4 \
  --num_workers=4 \
  --results_dir="$EXP_DIR" \
  --exp_name=v6_shapley_distill_kl_tk6 \
  > "logs/iemocap_shapdistill_kl_tk6_fold${FOLD}.log" 2>&1
echo "[$(date +%T)] cuda:${GPU} fold ${FOLD}: target_k=6 done; chain complete"
