#!/bin/bash
# Full 100-epoch CMU-MOSEI runs: DMBF + Learning-to-Route, 3 seeds each (fold 0/1/2 ->
# seed 42/43/44, split_id fixed 0). 6 jobs across 4 GPU queues, resume-guarded.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
LOGD=./results_dmbf_l2r_logs; mkdir -p "$LOGD"
COMMON="--num_epochs 100 --batch_size 32 --num_workers 4 --mosei_task 2"
declare -A SEED=( [0]=42 [1]=43 [2]=44 )

# job list: "script|resdir|fold"
JOBS=(
  "main_dmbf_mosei.py|./results_dmbf_mosei|0"
  "main_dmbf_mosei.py|./results_dmbf_mosei|1"
  "main_dmbf_mosei.py|./results_dmbf_mosei|2"
  "main_l2r_mosei.py|./results_l2r_mosei|0"
  "main_l2r_mosei.py|./results_l2r_mosei|1"
  "main_l2r_mosei.py|./results_l2r_mosei|2"
)

run_queue() {
  local g=$1
  for i in "${!JOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r script resdir fold <<< "${JOBS[$i]}"
    local tag="$(basename $script .py)_f${fold}"
    # resume guard: skip if a results_fold json already exists under any dated dir
    if ls ${resdir}/*/results_fold${fold}.json >/dev/null 2>&1; then
      echo "[gpu$g] SKIP $tag (done)"; continue
    fi
    echo "[gpu$g] START $tag (seed=${SEED[$fold]})"
    $PY $script $COMMON --fold "$fold" --seed_num "${SEED[$fold]}" \
        --results_dir "$resdir" --cuda_pick "cuda:$g" > "$LOGD/${tag}.log" 2>&1
    echo "[gpu$g] DONE  $tag (exit $?)"
  done
}

echo "Launching ${#JOBS[@]} jobs across 4 GPUs ..."
for g in 0 1 2 3; do run_queue "$g" & done
wait
echo "ALL_DMBF_L2R_MOSEI_DONE"
