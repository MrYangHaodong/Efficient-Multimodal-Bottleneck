#!/usr/bin/env bash
# Train QMF + PDF (late-fusion baselines), cross-subject, 4 datasets x 3 folds.
# Both come from main_qmfpdf_<ds>_late_fusion.py via --fusion_method {qmf,pdf}.
# Settings already canonical in the mains (daliahar sax/d64/bs64, pamap2
# sax_noisy, wesad sax/d64/bs64, iemocap float/d128/bs32, epochs=100); here we
# only force --num_epochs 100 + iemocap --split_mode cross_subject.
#
# 8 (method,ds) groups, each = fold {0,1,2} + --aggregate; <=1 group/GPU.
# Usage:  GPUS="0 1 2 3" bash run_clean_qmf_pdf.sh
set -u

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

METHODS=(qmf pdf)
DATASETS=(daliahar pamap2 wesad iemocap)

main_script() { echo "main_qmfpdf_${1}_late_fusion.py"; }   # $1=ds
ds_args()  { [ "$1" = "iemocap" ] && echo "--split_mode cross_subject" || echo ""; }
resdir()   { echo "$RES/$1_$2"; }    # $1=method $2=ds
expname()  { echo "$1_aligned"; }    # $1=method

fold_done() {    # $1=method $2=ds $3=fold
  local d; d="$(resdir "$1" "$2")"
  compgen -G "$d"/*/results_fold"$3".json > /dev/null 2>&1
}

echo "== preflight: --help for 4 qmfpdf mains =="
PF_FAIL=0
for ds in "${DATASETS[@]}"; do
  s="$(main_script "$ds")"
  if ! ( cd "$TRAIN" && CUDA_VISIBLE_DEVICES="" python "$s" --help >/dev/null 2>"$LOGS/preflight_qmfpdf_${ds}.err" ); then
    echo "  PREFLIGHT FAIL: $s"; PF_FAIL=1
  else echo "  ok: $s"; fi
done
[ "$PF_FAIL" -ne 0 ] && { echo "Aborting."; exit 1; }
echo "== preflight passed =="

run_group() {    # $1=method(qmf|pdf) $2=ds $3=gpu
  local m="$1" ds="$2" gpu="$3"
  local s; s="$(main_script "$ds")"
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
      ( cd "$TRAIN" && python "$s" --fusion_method "$m" $dsa --fold "$f" \
          --cuda_pick "cuda:$gpu" --num_epochs 100 --results_dir "$rd" --exp_name "$en" )
    done
    echo "  >>> aggregate"
    ( cd "$TRAIN" && python "$s" --fusion_method "$m" $dsa --aggregate \
        --results_dir "$rd" --exp_name "$en" )
  } >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $m/$ds"
}

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
echo "== ALL QMF+PDF GROUPS COMPLETE =="
