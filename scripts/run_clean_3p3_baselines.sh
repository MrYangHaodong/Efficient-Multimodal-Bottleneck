#!/usr/bin/env bash
# Retrain crossattn, MultiModN, DyMo at 3-enc + 3-fusion, cross-subject,
# 4 datasets x 3 folds, inside clean_models/.  Results -> results_3p3/.
#
#   crossattn : main_crossattn_v6_<ds>.py     --num_layers_per_modal 3 --num_layers 3
#   multimodn : main_multimodn_<ds>_dyna.py   --num_layers 3 --num_layers_fus 3
#   dymo      : main_dynmm_dymo_<ds>_late.py  --method dymo --num_layers_per_modal 3 --num_layers 3
#   HAR (daliahar/pamap2/wesad): d_model=64, folds via dataset_cfg
#   IEMOCAP                    : d_model=128, --split_mode cross_subject
#
# 12 (method,ds) groups, each = fold {0,1,2} sequential + --aggregate.
# Groups are assigned round-robin to GPUS and run sequentially within a GPU
# (<=1 group per GPU at a time -> no iemocap@d128 co-location OOM).
#
# Usage:  GPUS="0 1 2 3" bash run_clean_3p3_baselines.sh
set -u

# --- activate the maestro conda env (numpy/torch live here) ---
source /home/egg8711/miniconda3/etc/profile.d/conda.sh
conda activate maestro || { echo "FATAL: cannot activate maestro env"; exit 1; }
echo "python -> $(which python)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/train"
RES="$HERE/results_3p3"
LOGS="$RES/logs"
mkdir -p "$LOGS"

GPUS="${GPUS:-0 1 2 3}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

METHODS=(crossattn multimodn dymo)
DATASETS=(daliahar pamap2 wesad iemocap)

# ---- per-(method,ds) command builder for a single fold -------------------
main_script() {  # $1=method $2=ds -> script name
  case "$1" in
    crossattn) echo "main_crossattn_v6_$2.py" ;;
    multimodn) echo "main_multimodn_${2}_dyna.py" ;;
    dymo)      echo "main_dynmm_dymo_${2}_late_fusion.py" ;;
  esac
}

depth_args() {   # $1=method
  case "$1" in
    crossattn) echo "--num_layers_per_modal 3 --num_layers 3" ;;
    multimodn) echo "--num_layers 3 --num_layers_fus 3" ;;
    dymo)      echo "--method dymo --num_layers_per_modal 3 --num_layers 3" ;;
  esac
}

ds_args() {      # $1=ds
  if [ "$1" = "iemocap" ]; then echo "--d_model 128 --split_mode cross_subject"
  else echo "--d_model 64"; fi
}

resdir() { echo "$RES/$1_$2"; }          # $1=method $2=ds
expname() { echo "$1_3p3"; }             # $1=method

# fold result already present anywhere under results_dir?
fold_done() {    # $1=method $2=ds $3=fold
  local d; d="$(resdir "$1" "$2")"
  compgen -G "$d"/*/results_fold"$3".json > /dev/null 2>&1
}

# ---- preflight: every main parses & imports (--help) ---------------------
echo "== preflight: --help for all 12 mains =="
PF_FAIL=0
for m in "${METHODS[@]}"; do for ds in "${DATASETS[@]}"; do
  s="$(main_script "$m" "$ds")"
  if ! ( cd "$TRAIN" && CUDA_VISIBLE_DEVICES="" python "$s" --help >/dev/null 2>"$LOGS/preflight_${m}_${ds}.err" ); then
    echo "  PREFLIGHT FAIL: $s  (see $LOGS/preflight_${m}_${ds}.err)"; PF_FAIL=1
  else echo "  ok: $s"; fi
done; done
if [ "$PF_FAIL" -ne 0 ]; then echo "Aborting: fix preflight failures first."; exit 1; fi
echo "== preflight passed =="

# ---- run one (method,ds) group on one GPU: folds 0,1,2 then aggregate ----
run_group() {    # $1=method $2=ds $3=gpu
  local m="$1" ds="$2" gpu="$3"
  local s; s="$(main_script "$m" "$ds")"
  local rd; rd="$(resdir "$m" "$ds")"
  local en; en="$(expname "$m")"
  local da; da="$(depth_args "$m")"
  local dsa; dsa="$(ds_args "$ds")"
  local log="$LOGS/${m}_${ds}.log"
  echo "[$(date +%H:%M:%S)] START group $m/$ds on cuda:$gpu -> $log"
  {
    echo "### $m / $ds  (cuda:$gpu)  $(date)"
    for f in 0 1 2; do
      if fold_done "$m" "$ds" "$f"; then echo "  [skip] fold $f already done"; continue; fi
      echo "  >>> fold $f"
      # --num_epochs 100 forced for ALL methods (multimodn mains default to 200).
      ( cd "$TRAIN" && python "$s" $da $dsa --fold "$f" --cuda_pick "cuda:$gpu" \
          --num_epochs 100 --results_dir "$rd" --exp_name "$en" )
    done
    echo "  >>> aggregate"
    ( cd "$TRAIN" && python "$s" $da $dsa --aggregate \
        --results_dir "$rd" --exp_name "$en" )
  } >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  group $m/$ds"
}

# ---- assign groups round-robin to GPUs; serialize within a GPU ----------
# NB: do NOT name this array GROUPS -- that is a bash special (the user's GIDs).
declare -a GLIST=()
for m in "${METHODS[@]}"; do for ds in "${DATASETS[@]}"; do GLIST+=("$m:$ds"); done; done
echo "== ${#GLIST[@]} groups over $NGPU GPUs (${GPUS}) =="

for g in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$g]}"
  (
    for i in "${!GLIST[@]}"; do
      if [ $(( i % NGPU )) -eq "$g" ]; then
        IFS=':' read -r m ds <<< "${GLIST[$i]}"
        run_group "$m" "$ds" "$gpu"
      fi
    done
  ) &
done
wait
echo "== ALL GROUPS COMPLETE =="
