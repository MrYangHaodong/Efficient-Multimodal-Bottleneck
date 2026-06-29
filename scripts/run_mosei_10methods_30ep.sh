#!/bin/bash
# CMU-MOSEI benchmark, 30-epoch training, 5 seeds {7,21,88,365,1024}, 10 methods:
# multimodn, shaspec, decalign, crossattn, dymo, seqA, seqA_ps, MMEE, DMBF, L2R.
# (CREMA excluded per request.) Competitive/established per-method recipes; only epochs->30.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
SEEDS=(7 21 88 365 1024)
GLV="--weight_decay 1e-4 --mosei_task 2 --num_workers 4 --num_epochs 30"
RDG=./results_mosei_5seed
mkdir -p "$RDG/logs" ./results_mmee/logs ./results_dmbf_mosei ./results_l2r_mosei ./logs_mosei30

METHODS=(multimodn shaspec decalign crossattn dymo seqA seqA_ps mmee dmbf l2r)

cmd_of() {  # $1=method $2=seed -> full python args
  local m=$1 s=$2
  case "$m" in
    seqA)      echo "main_v6_mosei_seqfusion_late_fusion.py --use_sparse_attn False --n_bottlenecks 4 --n_fusion_distill -1 --num_layers_per_modal 3 --num_layers 3 $GLV --seed_num $s --exp_name seqA_acc2_s$s --results_dir $RDG";;
    seqA_ps)   echo "main_v6_mosei_seqfusion_late_fusion.py --use_sparse_attn False --n_bottlenecks 4 --n_fusion_distill -1 --num_layers_per_modal 3 --num_layers 3 --prefix_supervision $GLV --seed_num $s --exp_name seqA_ps_acc2_s$s --results_dir $RDG";;
    crossattn) echo "main_crossattn_v6_mosei.py --use_sparse_attn True --sparse_attn_variant orig --num_layers_per_modal 3 --num_layers 3 $GLV --seed_num $s --exp_name crossattn_acc2_s$s --results_dir $RDG";;
    multimodn) echo "main_multimodn_mosei_dyna.py --num_layers 3 --num_layers_fus 3 $GLV --seed_num $s --exp_name multimodn_acc2_s$s --results_dir $RDG";;
    shaspec)   echo "main_shaspec_mosei_dyna.py --num_layers 3 $GLV --seed_num $s --exp_name shaspec_acc2_s$s --results_dir $RDG";;
    decalign)  echo "main_decalign_mosei_dyna.py --num_layers 3 $GLV --seed_num $s --exp_name decalign_acc2_s$s --results_dir $RDG";;
    dymo)      echo "main_dymo_proto.py --dataset mosei --max_seq_len 50 $GLV --seed_num $s --exp_name dymo_acc2_s$s --results_dir $RDG";;
    mmee)      echo "main_mmee_4ds.py --dataset mosei --max_seq_len 50 --num_epochs 30 --num_workers 4 --seed $s --exp_name mmee_s$s --results_dir ./results_mmee";;
    dmbf)      echo "main_dmbf_mosei.py --num_epochs 30 --batch_size 32 --num_workers 4 --mosei_task 2 --seed_num $s --exp_name dmbf_mosei_s$s --results_dir ./results_dmbf_mosei";;
    l2r)       echo "main_l2r_mosei.py --num_epochs 30 --batch_size 32 --num_workers 4 --mosei_task 2 --seed_num $s --exp_name l2r_mosei_s$s --results_dir ./results_l2r_mosei";;
  esac
}

JOBS=()
for m in "${METHODS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$m|$s"); done; done
echo "Total jobs: ${#JOBS[@]}  (10 methods x 5 seeds), 30 epochs"

run_queue() {  # $1=gpu
  local g=$1 i=0
  for job in "${JOBS[@]}"; do
    if [ $((i % 4)) -eq "$g" ]; then
      local m="${job%%|*}" s="${job#*|}"
      local lg="./logs_mosei30/${m}_s${s}.log"
      echo "[gpu$g] START $m s$s"
      $PY $(cmd_of "$m" "$s") --cuda_pick "cuda:$g" > "$lg" 2>&1
      echo "[gpu$g] DONE  $m s$s (exit $?)"
    fi
    i=$((i+1))
  done
}
echo "Launching across 4 GPUs ..."
for g in 0 1 2 3; do run_queue "$g" & done
wait
echo "ALL_MOSEI_10METHODS_30EP_DONE"
