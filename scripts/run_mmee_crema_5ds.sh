#!/bin/bash
# MMEE (depth-axis anytime exit) + CREMA-style (modality-axis exit) baselines on 5 datasets:
# IEMOCAP / DaliaHAR / PAMAP2 / DSADS (cross-subject FOLDS) + CMU-MOSEI (5 SEEDS).
# Default per-main recipes; MOSEI uses --max_seq_len 50 and seed-distinguished exp_name.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
COMMON="--num_epochs 100 --num_workers 4"
MOSEI_SEEDS=(7 21 88 365 1024)
declare -A MAIN=( [mmee]="main_mmee_4ds.py" [crema]="main_crema_4ds.py" )
declare -A RD=( [mmee]="./results_mmee" [crema]="./results_crema" )
declare -A EXP=( [mmee]="mmee" [crema]="crema_style" )
mkdir -p results_mmee/logs results_crema/logs

# Build job list: "method|dataset|kind|id"
JOBS=()
for meth in mmee crema; do
  for ds in iemocap daliahar pamap2; do for f in 0 1 2; do JOBS+=("$meth|$ds|fold|$f"); done; done
  for f in 0 1 2 3; do JOBS+=("$meth|dsads|fold|$f"); done
  for s in "${MOSEI_SEEDS[@]}"; do JOBS+=("$meth|mosei|seed|$s"); done
done
NJOBS=${#JOBS[@]}
echo "Total jobs: $NJOBS"

run_one() {  # $1=method $2=dataset $3=kind $4=id $5=gpu
  local meth=$1 ds=$2 kind=$3 id=$4 g=$5
  local main=${MAIN[$meth]} rd=${RD[$meth]} exp=${EXP[$meth]}
  if [ "$kind" = fold ]; then
    local dir="$rd/${exp}_${ds}/fold${id}"; local lg="$rd/logs/${meth}_${ds}_f${id}.log"
    [ -f "$dir/results_fold${id}.json" ] && { echo "[gpu$g] SKIP $meth $ds f$id"; return; }
    echo "[gpu$g] START $meth $ds f$id"
    $PY $main --dataset $ds --fold $id $COMMON --cuda_pick "cuda:$g" --results_dir $rd > "$lg" 2>&1
    echo "[gpu$g] DONE  $meth $ds f$id (exit $?)"
  else  # mosei seed
    local dir="$rd/${exp}_s${id}_mosei/run"; local lg="$rd/logs/${meth}_mosei_s${id}.log"
    [ -f "$dir/results.json" ] && { echo "[gpu$g] SKIP $meth mosei s$id"; return; }
    echo "[gpu$g] START $meth mosei s$id"
    $PY $main --dataset mosei --max_seq_len 50 --seed $id --exp_name "${exp}_s${id}" $COMMON \
        --cuda_pick "cuda:$g" --results_dir $rd > "$lg" 2>&1
    echo "[gpu$g] DONE  $meth mosei s$id (exit $?)"
  fi
}

gpu_queue() {  # $1 = gpu id; runs every job whose index%4==gpu
  local g=$1 i=0
  for job in "${JOBS[@]}"; do
    if [ $((i % 4)) -eq "$g" ]; then
      IFS='|' read -r meth ds kind id <<< "$job"
      run_one "$meth" "$ds" "$kind" "$id" "$g"
    fi
    i=$((i+1))
  done
}

echo "Launching across 4 GPUs ..."
for g in 0 1 2 3; do gpu_queue "$g" & done
wait
echo "ALL_MMEE_CREMA_5DS_DONE"
