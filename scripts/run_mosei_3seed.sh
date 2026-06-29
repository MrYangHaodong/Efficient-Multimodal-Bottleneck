#!/bin/bash
# CMU-MOSEI 3-seed training: v6 seqA, crossattn, shaspec, dymo_proto, multimodn, decalign.
# Acc-2, weight_decay=1e-4 (unified), 3 seeds; seqA/crossattn 3-enc+3-fusion.
# 18 runs (6 models x 3 seeds) balanced across 4 GPUs, each GPU sequential.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
RD=./results_mosei_3seed
LOGD=$RD/logs
mkdir -p "$LOGD"
SEEDS=(42 1337 2025)
COMMON="--weight_decay 1e-4 --mosei_task 2 --num_epochs 100 --num_workers 4 --results_dir $RD"

# tag -> "script + structural flags"
declare -A MODEL=(
  [seqA]="main_v6_mosei_seqfusion_late_fusion.py --use_sparse_attn False --n_bottlenecks 4 --n_fusion_distill -1 --num_layers_per_modal 3 --num_layers 3"
  [crossattn]="main_crossattn_v6_mosei.py --use_sparse_attn True --sparse_attn_variant orig --num_layers_per_modal 3 --num_layers 3"
  [shaspec]="main_shaspec_mosei_dyna.py --num_layers 3"
  [multimodn]="main_multimodn_mosei_dyna.py --num_layers 3 --num_layers_fus 3"
  [decalign]="main_decalign_mosei_dyna.py --num_layers 3"
  [dymo]="main_dymo_proto.py --dataset mosei --max_seq_len 50"
)
ORDER=(seqA crossattn shaspec multimodn decalign dymo)

# Build the 18-job list, assign GPU = jobindex % 4.
declare -a JOB_GPU JOB_DESC
i=0
for tag in "${ORDER[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOB_GPU[$i]=$((i % 4))
    JOB_DESC[$i]="$tag|$s"
    i=$((i+1))
  done
done
NJOBS=$i

run_gpu_queue() {  # $1 = gpu id
  local g=$1
  for j in "${!JOB_DESC[@]}"; do
    [ "${JOB_GPU[$j]}" -ne "$g" ] && continue
    local tag="${JOB_DESC[$j]%|*}"; local s="${JOB_DESC[$j]#*|}"
    local cmd="${MODEL[$tag]}"
    local exp="${tag}_acc2_s${s}"
    local log="$LOGD/${exp}.log"
    # resume: skip runs that already produced a results.json
    if compgen -G "$RD/*_${exp}*/results.json" > /dev/null 2>&1; then
      echo "[gpu$g] SKIP  $exp (results.json exists)"; continue
    fi
    echo "[gpu$g] START $exp"
    $PY $cmd $COMMON --seed_num "$s" --exp_name "$exp" --cuda_pick "cuda:$g" > "$log" 2>&1
    echo "[gpu$g] DONE  $exp (exit $?)"
  done
}

echo "Launching $NJOBS jobs across 4 GPUs ..."
for g in 0 1 2 3; do run_gpu_queue "$g" & done
wait
echo "ALL_MOSEI_3SEED_DONE"
