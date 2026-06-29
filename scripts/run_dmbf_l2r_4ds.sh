#!/bin/bash
# DMBF + L2R on the 4 HAR/IEMOCAP datasets (cross-subject folds), 100 epochs (matching
# the other HAR baselines). SAX for daliahar/pamap2/dsads, float for iemocap (via ee_data).
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
COMMON="--num_epochs 100 --num_workers 4"
declare -A MAIN=( [dmbf]="main_dmbf_4ds.py" [l2r]="main_l2r_4ds.py" )
declare -A RD=( [dmbf]="./results_dmbf_4ds" [l2r]="./results_l2r_4ds" )
declare -A EXP=( [dmbf]="dmbf" [l2r]="l2r" )
mkdir -p results_dmbf_4ds/logs results_l2r_4ds/logs

JOBS=()
for meth in dmbf l2r; do
  for ds in iemocap daliahar pamap2; do for f in 0 1 2; do JOBS+=("$meth|$ds|$f"); done; done
  for f in 0 1 2 3; do JOBS+=("$meth|dsads|$f"); done
done
echo "Total jobs: ${#JOBS[@]}"

run_one() {  # $1=method $2=dataset $3=fold $4=gpu
  local meth=$1 ds=$2 f=$3 g=$4 main=${MAIN[$1]} rd=${RD[$1]} exp=${EXP[$1]}
  local dir="$rd/${exp}_${ds}/fold${f}"; local lg="$rd/logs/${meth}_${ds}_f${f}.log"
  [ -f "$dir/results_fold${f}.json" ] && { echo "[gpu$g] SKIP $meth $ds f$f"; return; }
  echo "[gpu$g] START $meth $ds f$f"
  $PY $main --dataset $ds --fold $f $COMMON --cuda_pick "cuda:$g" --results_dir $rd > "$lg" 2>&1
  echo "[gpu$g] DONE  $meth $ds f$f (exit $?)"
}
gpu_queue() { local g=$1 i=0; for job in "${JOBS[@]}"; do
    if [ $((i % 4)) -eq "$g" ]; then IFS='|' read -r m d f <<< "$job"; run_one "$m" "$d" "$f" "$g"; fi
    i=$((i+1)); done; }
echo "Launching across 4 GPUs ..."
for g in 0 1 2 3; do gpu_queue "$g" & done
wait
echo "ALL_DMBF_L2R_4DS_DONE"
