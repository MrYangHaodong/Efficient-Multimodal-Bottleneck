#!/usr/bin/env bash
# Faithful DyMo (prototype + per-subset Gaussian + calibration greedy) training,
# cross-subject, 4 datasets x 3 folds. Settings aligned to the suite (per-dataset
# transform/d_model/batch baked into main_dymo_proto.py; 3-enc+3-fusion backbone).
# 4 groups (one dataset each), each = fold {0,1,2} + --aggregate; <=1 group/GPU.
# Usage:  GPUS="0 1 2 3" bash run_clean_dymo_proto.sh
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh
conda activate maestro || { echo "FATAL: maestro env"; exit 1; }
echo "python -> $(which python)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/train"; RES="$HERE/results_3p3"; LOGS="$RES/logs"; mkdir -p "$LOGS"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"; NGPU=${#GPU_ARR[@]}
DATASETS=(daliahar pamap2 wesad iemocap)

ds_args() { [ "$1" = "iemocap" ] && echo "--split_mode cross_subject" || echo ""; }
resdir()  { echo "$RES/dymo_proto_$1"; }
fold_done() { compgen -G "$(resdir "$1")"/*/results_fold"$2".json > /dev/null 2>&1; }

echo "== preflight: --help =="; PF=0
for ds in "${DATASETS[@]}"; do
  if ! ( cd "$TRAIN" && CUDA_VISIBLE_DEVICES="" python main_dymo_proto.py --dataset "$ds" --help \
         >/dev/null 2>"$LOGS/preflight_dymoproto_${ds}.err" ); then echo "  FAIL $ds"; PF=1; else echo "  ok $ds"; fi
done
[ "$PF" -ne 0 ] && { echo "Aborting."; exit 1; }
echo "== preflight passed =="

run_group() {   # $1=ds $2=gpu
  local ds="$1" gpu="$2" rd; rd="$(resdir "$ds")"; local dsa; dsa="$(ds_args "$ds")"
  local log="$LOGS/dymo_proto_${ds}.log"
  echo "[$(date +%H:%M:%S)] START dymo_proto/$ds on cuda:$gpu -> $log"
  {
    echo "### dymo_proto / $ds (cuda:$gpu) $(date)"
    for f in 0 1 2; do
      if fold_done "$ds" "$f"; then echo "  [skip] fold $f done"; continue; fi
      echo "  >>> fold $f"
      ( cd "$TRAIN" && python main_dymo_proto.py --dataset "$ds" $dsa --fold "$f" \
          --cuda_pick "cuda:$gpu" --num_epochs 100 \
          --results_dir "$rd" --exp_name dymo_proto )
    done
    echo "  >>> aggregate"
    ( cd "$TRAIN" && python main_dymo_proto.py --dataset "$ds" $dsa --aggregate \
        --results_dir "$rd" --exp_name dymo_proto )
  } >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  dymo_proto/$ds"
}

declare -a GLIST=("${DATASETS[@]}")
echo "== ${#GLIST[@]} groups over $NGPU GPUs (${GPUS}) =="
for g in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$g]}"
  ( for i in "${!GLIST[@]}"; do
      [ $(( i % NGPU )) -eq "$g" ] && run_group "${GLIST[$i]}" "$gpu"
    done ) &
done
wait
echo "== ALL DYMO_PROTO GROUPS COMPLETE =="
