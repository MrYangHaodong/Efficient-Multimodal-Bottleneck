#!/bin/bash
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
RD=./results_eav_unimodal; LOGD=$RD/logs; mkdir -p "$LOGD"
# diagnostic: do the NON-audio modalities carry emotion signal? (all 8400 trials incl Listening)
declare -A MODS=( [eeg]="eeg" [video]="video" [eegvideo]="eeg video" )
JOBS=()
for name in eeg video eegvideo; do for f in 0 1 2; do JOBS+=("$name|$f"); done; done
run_q(){ local g=$1
  for i in "${!JOBS[@]}"; do [ $((i%4)) -ne "$g" ] && continue
    IFS='|' read -r name f <<< "${JOBS[$i]}"
    ls $RD/$name/*/results_fold$f.json >/dev/null 2>&1 && { echo "[gpu$g] SKIP $name/f$f"; continue; }
    echo "[gpu$g] START $name/f$f mods=[${MODS[$name]}]"
    $PY main_v6_eav_seqfusion_late_fusion.py --modalities ${MODS[$name]} --fold $f --num_epochs 100 \
      --batch_size 32 --max_seq_len 128 --split_mode cross_subject --cuda_pick cuda:$g \
      --results_dir $RD/$name --exp_name $name > "$LOGD/${name}_f${f}.log" 2>&1
    echo "[gpu$g] DONE  $name/f$f (exit $?)"
  done; }
echo "Launching ${#JOBS[@]} unimodal/bimodal diagnostic jobs ..."
for g in 0 1 2 3; do run_q "$g" & done; wait
echo "ALL_EAV_UNIMODAL_DONE"
