#!/bin/bash
# CMU-MOSEI 5-NEW-seed training (Acc-2): crossattn, decalign, dymo_proto, multimodn,
# + our seqA in TWO variants (original; prefix-supervision = new training strategy).
# 6 configs x 5 seeds = 30 runs, balanced across 4 GPUs (each GPU sequential queue).
# Seeds {7,21,88,365,1024} are distinct from the earlier 3-seed run {42,1337,2025}.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
RD=./results_mosei_5seed
LOGD=$RD/logs
mkdir -p "$LOGD"
SEEDS=(7 21 88 365 1024)
COMMON="--weight_decay 1e-4 --mosei_task 2 --num_epochs 100 --num_workers 4 --results_dir $RD"

# tag -> "script + structural flags"  (same recipes as run_mosei_3seed.sh)
declare -A MODEL=(
  [seqA]="main_v6_mosei_seqfusion_late_fusion.py --use_sparse_attn False --n_bottlenecks 4 --n_fusion_distill -1 --num_layers_per_modal 3 --num_layers 3"
  [seqA_ps]="main_v6_mosei_seqfusion_late_fusion.py --use_sparse_attn False --n_bottlenecks 4 --n_fusion_distill -1 --num_layers_per_modal 3 --num_layers 3 --prefix_supervision"
  [crossattn]="main_crossattn_v6_mosei.py --use_sparse_attn True --sparse_attn_variant orig --num_layers_per_modal 3 --num_layers 3"
  [multimodn]="main_multimodn_mosei_dyna.py --num_layers 3 --num_layers_fus 3"
  [decalign]="main_decalign_mosei_dyna.py --num_layers 3"
  [dymo]="main_dymo_proto.py --dataset mosei --max_seq_len 50"
  [shaspec]="main_shaspec_mosei_dyna.py --num_layers 3"
)
ORDER=(shaspec seqA seqA_ps crossattn multimodn decalign dymo)

declare -a JOB_GPU JOB_DESC
i=0
for tag in "${ORDER[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOB_GPU[$i]=$((i % 4)); JOB_DESC[$i]="$tag|$s"; i=$((i+1))
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
    if compgen -G "$RD/*_${exp}*/results.json" > /dev/null 2>&1 || compgen -G "$RD/*_${exp}_*/results_fold*.json" > /dev/null 2>&1; then
      echo "[gpu$g] SKIP  $exp (results exist)"; continue
    fi
    echo "[gpu$g] START $exp"
    $PY $cmd $COMMON --seed_num "$s" --exp_name "$exp" --cuda_pick "cuda:$g" > "$log" 2>&1
    echo "[gpu$g] DONE  $exp (exit $?)"
  done
}

echo "Launching $NJOBS jobs across 4 GPUs ..."
for g in 0 1 2 3; do run_gpu_queue "$g" & done
wait
echo "ALL_MOSEI_5SEED_DONE"
