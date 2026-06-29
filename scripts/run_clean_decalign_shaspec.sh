#!/usr/bin/env bash
# Train DecAlign + ShaSpec, cross-subject, 4 datasets x 3 folds, inside
# clean_models/.  Settings are aligned to crossattn/dymo/multimodn/seqA:
#   daliahar : sax,        d_model=64,  batch=64
#   pamap2   : sax_noisy,  d_model=64,  batch=64
#   wesad    : sax,        d_model=64,  batch=64  (decalign: num_prototypes=3, M=10 OT)
#   iemocap  : float,      d_model=128, batch=32  (--split_mode cross_subject)
# All forced --num_epochs 100. Per-dataset transform/d_model/batch are baked
# into each main's defaults; here we only force epochs + iemocap split_mode.
#
# 8 (method,ds) groups, each = fold {0,1,2} sequential + --aggregate.
# <=1 group per GPU at a time (DecAlign is memory-heavy: iemocap 116M /
# wesad 140M params -> needs a near-empty GPU; do NOT share).
#
# Usage:  GPUS="0 3" bash run_clean_decalign_shaspec.sh
set -u

source /home/egg8711/miniconda3/etc/profile.d/conda.sh
conda activate maestro || { echo "FATAL: cannot activate maestro env"; exit 1; }
echo "python -> $(which python)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/train"
RES="$HERE/results_3p3"
LOGS="$RES/logs"
mkdir -p "$LOGS"

GPUS="${GPUS:-0 3}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

METHODS=(decalign shaspec)
DATASETS=(daliahar pamap2 wesad iemocap)

main_script() {  # $1=method $2=ds
  case "$1" in
    decalign) echo "main_decalign_${2}_dyna.py" ;;
    shaspec)  echo "main_shaspec_${2}_dyna.py" ;;
  esac
}
ds_args() {      # $1=ds  (iemocap is session-disjoint; d_model/transform/batch from main defaults)
  [ "$1" = "iemocap" ] && echo "--split_mode cross_subject" || echo ""
}
resdir()  { echo "$RES/$1_$2"; }       # $1=method $2=ds
expname() { echo "$1_aligned"; }       # $1=method

fold_done() {    # $1=method $2=ds $3=fold
  local d; d="$(resdir "$1" "$2")"
  compgen -G "$d"/*/results_fold"$3".json > /dev/null 2>&1
}

# ---- preflight: every main parses & imports ----
echo "== preflight: --help for all 8 mains =="
PF_FAIL=0
for m in "${METHODS[@]}"; do for ds in "${DATASETS[@]}"; do
  s="$(main_script "$m" "$ds")"
  if ! ( cd "$TRAIN" && CUDA_VISIBLE_DEVICES="" python "$s" --help >/dev/null 2>"$LOGS/preflight_${m}_${ds}.err" ); then
    echo "  PREFLIGHT FAIL: $s (see $LOGS/preflight_${m}_${ds}.err)"; PF_FAIL=1
  else echo "  ok: $s"; fi
done; done
[ "$PF_FAIL" -ne 0 ] && { echo "Aborting: fix preflight first."; exit 1; }
echo "== preflight passed =="

run_group() {    # $1=method $2=ds $3=gpu
  local m="$1" ds="$2" gpu="$3"
  local s; s="$(main_script "$m" "$ds")"
  local rd; rd="$(resdir "$m" "$ds")"
  local en; en="$(expname "$m")"
  local dsa; dsa="$(ds_args "$ds")"
  local log="$LOGS/${m}_${ds}.log"
  echo "[$(date +%H:%M:%S)] START $m/$ds on cuda:$gpu -> $log"
  {
    echo "### $m / $ds  (cuda:$gpu)  $(date)"
    for f in 0 1 2; do
      if fold_done "$m" "$ds" "$f"; then echo "  [skip] fold $f done"; continue; fi
      echo "  >>> fold $f"
      ( cd "$TRAIN" && python "$s" $dsa --fold "$f" --cuda_pick "cuda:$gpu" \
          --num_epochs 100 --results_dir "$rd" --exp_name "$en" )
    done
    echo "  >>> aggregate"
    ( cd "$TRAIN" && python "$s" $dsa --aggregate --results_dir "$rd" --exp_name "$en" )
  } >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $m/$ds"
}

# ---- round-robin groups -> GPUs; serialize within a GPU ----
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
echo "== ALL DECALIGN+SHASPEC GROUPS COMPLETE =="
