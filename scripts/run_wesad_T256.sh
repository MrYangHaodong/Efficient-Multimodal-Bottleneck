#!/usr/bin/env bash
# Retrain all 9 trainable methods on WESAD at T=256 (SAX word_length=1),
# 3 folds + aggregate, cross-subject. Same recipe as the T=128 runs + --word_length 1
# (crossattn also uses --sparse_attn_variant orig). Results -> results_3p3/T256_wesad/<method>/.
# Usage:  GPUS="0 1 2 3" bash run_wesad_T256.sh
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh
conda activate maestro || { echo "FATAL: maestro env"; exit 1; }
echo "python -> $(which python)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/train"; RES="$HERE/results_3p3/T256_wesad"; LOGS="$RES/logs"; mkdir -p "$LOGS"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"; NGPU=${#GPU_ARR[@]}
read -r -a METHODS <<< "${METHODS_LIST:-seqA crossattn multimodn dymo dymo_proto decalign shaspec qmf pdf}"

main_of() { case "$1" in
  seqA) echo "main_v6_wesad_seqfusion_late_fusion.py";;
  crossattn) echo "main_crossattn_v6_wesad.py";;
  multimodn) echo "main_multimodn_wesad_dyna.py";;
  dymo) echo "main_dynmm_dymo_wesad_late_fusion.py";;
  dymo_proto) echo "main_dymo_proto.py";;
  decalign) echo "main_decalign_wesad_dyna.py";;
  shaspec) echo "main_shaspec_wesad_dyna.py";;
  qmf|pdf) echo "main_qmfpdf_wesad_late_fusion.py";;
esac; }
extra_of() { case "$1" in
  seqA) echo "--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3 --word_length 1";;
  crossattn) echo "--num_layers_per_modal 3 --num_layers 3 --d_model 64 --sparse_attn_variant orig --word_length 1";;
  multimodn) echo "--num_layers 3 --num_layers_fus 3 --d_model 64 --word_length 1";;
  dymo) echo "--method dymo --num_layers_per_modal 3 --num_layers 3 --d_model 64 --word_length 1";;
  dymo_proto) echo "--dataset wesad --word_length 1";;
  decalign) echo "--transform sax --word_length 1 --batch_size 32";;  # M=10 + T=256: bs=32 probed ~30GB (bs=64 OOMs)
  shaspec) echo "--word_length 1";;
  qmf) echo "--fusion_method qmf --word_length 1";;
  pdf) echo "--fusion_method pdf --word_length 1";;
esac; }
resdir() { echo "$RES/$1"; }
fold_done() { compgen -G "$(resdir "$1")"/*/results_fold"$2".json >/dev/null 2>&1; }

echo "== preflight: --help =="; PF=0
for m in "${METHODS[@]}"; do s="$(main_of "$m")"
  ( cd "$TRAIN" && CUDA_VISIBLE_DEVICES="" python "$s" --help >/dev/null 2>"$LOGS/pf_${m}.err" ) \
    && echo "  ok $m" || { echo "  FAIL $m"; PF=1; }
done
[ "$PF" -ne 0 ] && { echo "Aborting."; exit 1; }
echo "== preflight passed =="

run_group() {  # $1=method $2=gpu
  local m="$1" gpu="$2" s rd ex log; s="$(main_of "$m")"; rd="$(resdir "$m")"
  ex="$(extra_of "$m")"; log="$LOGS/${m}.log"
  echo "[$(date +%H:%M:%S)] START $m (WESAD T=256) on cuda:$gpu -> $log"
  {
    echo "### $m / wesad T=256 (cuda:$gpu) $(date)"
    for f in 0 1 2; do
      if fold_done "$m" "$f"; then echo "  [skip] fold $f done"; continue; fi
      echo "  >>> fold $f"
      ( cd "$TRAIN" && python "$s" $ex --fold "$f" --cuda_pick "cuda:$gpu" \
          --num_epochs 100 --results_dir "$rd" --exp_name "${m}_T256" )
    done
    echo "  >>> aggregate"
    ( cd "$TRAIN" && python "$s" $ex --aggregate --results_dir "$rd" --exp_name "${m}_T256" )
  } >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $m"
}

declare -a GLIST=("${METHODS[@]}")
echo "== ${#GLIST[@]} methods over $NGPU GPUs (${GPUS}) =="
for g in "${!GPU_ARR[@]}"; do gpu="${GPU_ARR[$g]}"
  ( for i in "${!GLIST[@]}"; do
      [ $(( i % NGPU )) -eq "$g" ] && run_group "${GLIST[$i]}" "$gpu"
    done ) &
done
wait
echo "== ALL WESAD T=256 DONE =="
