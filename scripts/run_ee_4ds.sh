#!/bin/bash
# Full 100-epoch runs: MMEE + CREMA-style on iemocap/daliahar/pamap2/dsads x folds.
# 2 models x (3+3+3+4)=13 = 26 jobs, 4-GPU sequential queues. T matched to seqA curves.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
RD=./results_ee; LOGD=$RD/logs; mkdir -p "$LOGD"
COMMON="--num_epochs 100 --num_workers 4"

MODELS=(mmee crema)
declare -A FOLDS=( [iemocap]="0 1 2" [daliahar]="0 1 2" [pamap2]="0 1 2" [dsads]="0 1 2 3" )
declare -A BS=(    [iemocap]=32      [daliahar]=24     [pamap2]=8        [dsads]=48 )
declare -A WL=(    [iemocap]=2       [daliahar]=1      [pamap2]=2        [dsads]=1 )   # word_length (sax T)

# build job list
declare -a JOB_GPU JOB
i=0
for model in "${MODELS[@]}"; do
  for ds in iemocap daliahar pamap2 dsads; do
    for f in ${FOLDS[$ds]}; do
      JOB_GPU[$i]=$(( i % 4 )); JOB[$i]="$model|$ds|$f"; i=$((i+1))
    done
  done
done
NJOBS=$i

run_gpu_queue() {
  local g=$1
  for j in "${!JOB[@]}"; do
    [ "${JOB_GPU[$j]}" -ne "$g" ] && continue
    local model="${JOB[$j]%%|*}"; local rest="${JOB[$j]#*|}"
    local ds="${rest%%|*}"; local f="${rest##*|}"
    local out="$RD/${model}/${model}_${ds}/fold${f}/results_fold${f}.json"
    local log="$LOGD/${model}_${ds}_f${f}.log"
    if [ -f "$out" ]; then echo "[gpu$g] SKIP ${model}/${ds}/f${f} (done)"; continue; fi
    echo "[gpu$g] START ${model}/${ds}/f${f} (bs=${BS[$ds]} wl=${WL[$ds]})"
    $PY main_${model}_4ds.py --dataset "$ds" --fold "$f" $COMMON \
        --batch_size "${BS[$ds]}" --word_length "${WL[$ds]}" \
        --results_dir "$RD/${model}" --exp_name "${model}" --cuda_pick "cuda:$g" > "$log" 2>&1
    echo "[gpu$g] DONE  ${model}/${ds}/f${f} (exit $?)"
  done
}

echo "Launching $NJOBS jobs across 4 GPUs ..."
for g in 0 1 2 3; do run_gpu_queue "$g" & done
wait
echo "ALL_EE_4DS_DONE"
