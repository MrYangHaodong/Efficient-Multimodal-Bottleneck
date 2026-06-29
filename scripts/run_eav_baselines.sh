#!/bin/bash
# Full 100-epoch EAV runs (cross-subject): seqA, seqA-prefixsup, multimodn, shaspec,
# crossattn, dymo_proto, decalign — 3 folds each = 21 jobs, 4-GPU queue, resume-guarded.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
RD=./results_eav; LOGD=$RD/logs; mkdir -p "$LOGD"
COMMON="--num_epochs 100 --batch_size 32 --num_workers 4 --max_seq_len 128 --split_mode cross_subject"

# name|script|extra_args
CFGS=(
  "seqA|main_v6_eav_seqfusion_late_fusion.py|"
  "seqA_prefixsup|main_v6_eav_seqfusion_late_fusion.py|--prefix_supervision"
  "multimodn|main_multimodn_eav_dyna.py|"
  "shaspec|main_shaspec_eav_dyna.py|"
  "crossattn|main_crossattn_v6_eav.py|"
  "dymo|main_dynmm_dymo_eav_late_fusion.py|"
  "decalign|main_decalign_eav_dyna.py|"
)

# build flat job list: name|script|extra|fold
declare -a JOBS
for c in "${CFGS[@]}"; do
  for f in 0 1 2; do JOBS+=("$c|$f"); done
done
NJOBS=${#JOBS[@]}

run_queue() {
  local g=$1
  for i in "${!JOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r name script extra fold <<< "${JOBS[$i]}"
    local resdir="$RD/$name"
    if ls ${resdir}/*/results_fold${fold}.json >/dev/null 2>&1; then
      echo "[gpu$g] SKIP ${name}/f${fold} (done)"; continue
    fi
    echo "[gpu$g] START ${name}/f${fold}"
    $PY $script $COMMON $extra --fold "$fold" --cuda_pick "cuda:$g" \
        --results_dir "$resdir" --exp_name "$name" > "$LOGD/${name}_f${fold}.log" 2>&1
    echo "[gpu$g] DONE  ${name}/f${fold} (exit $?)"
  done
}

echo "Launching $NJOBS EAV jobs across 4 GPUs ..."
for g in 0 1 2 3; do run_queue "$g" & done
wait
echo "ALL_EAV_BASELINES_DONE"
