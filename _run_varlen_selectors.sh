#!/bin/bash
# Launch 12 selector trainings: 2 fusion variants (linear/dyna) × 2 phi sources (train/val) × 3 folds.
# Each GPU runs a chain of 4 sequential jobs (one per (variant, phi) pair) for its assigned fold.
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/egg8711/miniconda3/envs/maestro/bin/python
EXP_LIN=./model_chkpt/IEMOCAP/multimodal_model/2026-05-18_IEMOCAP_v6_fusion_varlen_linear_d256_lr1e4
EXP_DYN=./model_chkpt/IEMOCAP/multimodal_model/2026-05-18_IEMOCAP_v6_fusion_varlen_dyna_d256_lr1e4
EXPDIR=./model_chkpt/IEMOCAP/selector
mkdir -p "$EXPDIR" logs

run_one() {
  local fold=$1 gpu=$2 variant=$3 phi_src=$4 ckpt=$5
  local npz_split tag use_val
  if [ "$phi_src" = "train" ]; then
    npz_split=train; use_val=False; tag=train
  else
    npz_split=val;   use_val=True;  tag=val
  fi
  local exp_name="v6_varlen_${variant}_${tag}phi"
  local log_file="logs/varlen_sel_${variant}_${tag}_fold${fold}.log"
  local ckpt_path="${ckpt}/best_model_fold${fold}.pth"
  local npz_path="${ckpt}/shapley_${npz_split}_fold${fold}_phi.npz"
  echo "[$(date +%T)] start: variant=$variant phi=$phi_src fold=$fold cuda:$gpu"
  "$PY" -u script/main_v6_iemocap_selector_finetune_shapley_varlen.py \
    --fold="$fold" --cuda_pick="cuda:$gpu" \
    --frozen_ckpt="$ckpt_path" \
    --shapley_train_npz="$npz_path" \
    --use_val_split="$use_val" \
    --target_k=1 --selector_downsample_factor=4 \
    --tau=0.5 --lambda_kl=1.0 --lambda_probe=0.05 \
    --num_epochs=30 --batch_size=32 --lr=3e-4 --num_workers=2 \
    --max_seq_len=384 \
    --d_model=256 --internal_dim=256 \
    --results_dir="$EXPDIR" \
    --exp_name="$exp_name" \
    > "$log_file" 2>&1
  echo "[$(date +%T)] done: variant=$variant phi=$phi_src fold=$fold"
}

# Launch 3 fold chains in parallel; each chain runs 4 jobs sequentially on its assigned GPU.
for f in 0 1 2; do
  (
    run_one "$f" "$f" "lin" "train" "$EXP_LIN"
    run_one "$f" "$f" "lin" "val"   "$EXP_LIN"
    run_one "$f" "$f" "dyn" "train" "$EXP_DYN"
    run_one "$f" "$f" "dyn" "val"   "$EXP_DYN"
    echo "[$(date +%T)] chain fold $f done"
  ) > "logs/varlen_sel_chain_fold${f}.log" 2>&1 &
done
wait
echo "[$(date +%T)] ALL 12 selector trainings done"
