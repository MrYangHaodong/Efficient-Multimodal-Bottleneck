#!/bin/bash
# Full EAV runs for DMBF + AdaMML + MMEE, 3 cross-subject folds each = 9 jobs.
# 4-GPU flock-free round-robin queue, resume-guarded, detached.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
RD=./results_eav; LOGD=$RD/logs; mkdir -p "$LOGD"

# name|fold  ->  command built per-model below
JOBS=()
for f in 0 1 2; do JOBS+=("dmbf|$f" "adamml|$f" "mmee|$f"); done

cmd_of() {  # $1=name $2=fold $3=gpu
  local n="$1" f="$2" g="$3"
  case "$n" in
    dmbf)   echo "$PY main_dmbf_eav.py --fold $f --num_epochs 100 --batch_size 32 --num_workers 4 --results_dir $RD/dmbf --cuda_pick cuda:$g";;
    adamml) echo "$PY main_adamml_ours_eav.py --fold $f --warmup_epochs 40 --joint_epochs 40 --finetune_epochs 20 --batch_size 32 --results_dir $RD/adamml --exp_name adamml --cuda_pick cuda:$g";;
    mmee)   echo "$PY main_mmee_4ds.py --dataset eav --fold $f --num_epochs 100 --batch_size 32 --num_workers 4 --max_seq_len 128 --results_dir $RD/mmee --exp_name mmee --cuda_pick cuda:$g";;
  esac
}
done_of() {  # results path glob to skip if present
  local n="$1" f="$2"
  case "$n" in
    dmbf)   ls $RD/dmbf/*/results_fold$f.json 2>/dev/null;;
    adamml) ls $RD/adamml/*/results_fold$f.json 2>/dev/null;;
    mmee)   ls $RD/mmee/mmee_eav/fold$f/results_fold$f.json 2>/dev/null;;
  esac
}

run_queue() {
  local g=$1
  for i in "${!JOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r name fold <<< "${JOBS[$i]}"
    if done_of "$name" "$fold" >/dev/null 2>&1; then echo "[gpu$g] SKIP ${name}/f${fold}"; continue; fi
    echo "[gpu$g] START ${name}/f${fold}"
    eval "$(cmd_of "$name" "$fold" "$g")" > "$LOGD/${name}_f${fold}.log" 2>&1
    echo "[gpu$g] DONE  ${name}/f${fold} (exit $?)"
  done
}

echo "Launching ${#JOBS[@]} EAV extra jobs across 4 GPUs ..."
for g in 0 1 2 3; do run_queue "$g" & done
wait
echo "ALL_EAV_EXTRA_DONE"
