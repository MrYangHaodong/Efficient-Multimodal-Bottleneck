#!/usr/bin/env bash
# Original / vanilla MBT baseline on 4 datasets. Same backbone & training regime as
# seqA, but MBT topology: simultaneous (joint) fusion + symmetric MEAN bottleneck
# update + NO fusion-layer distilling conv (n_fusion_distill=0). This is the faithful
# "TRUE vanilla MBT (joint+mean+nodist)". Diff vs seqA recipe = ONLY:
#   --fusion_mode sequential  -> --fusion_mode joint
#   (remove) --seq_random_order
#   (add) --bottleneck_agg_mode mean   (add) --n_fusion_distill 0
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
OUT="$HERE/results_3p3/mbt_vanilla"; LOG="$OUT/logs"; mkdir -p "$LOG"; EP=100
main_of(){ case "$1" in
  daliahar) echo main_v6_daliahar_seqfusion_late_fusion.py;; pamap2) echo main_v6_pamap2_seqfusion_late_fusion.py;;
  dsads) echo main_v6_dsads_seqfusion_late_fusion.py;; iemocap) echo main_v6_iemocap_seqfusion_late_fusion.py;; esac; }
# seqA recipe with fusion_mode=joint, bottleneck_agg_mode=mean, n_fusion_distill=0, no seq_random_order.
recipe_of(){ case "$1" in
  daliahar) echo "--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=joint --bottleneck_agg_mode mean --n_fusion_distill 0 --head_mode gap --num_layers_per_modal 3 --num_layers 3 --word_length 1";;
  pamap2)   echo "--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=joint --bottleneck_agg_mode mean --n_fusion_distill 0 --head_mode gap --num_layers_per_modal 3 --num_layers 3 --transform sax_noisy --input_length_override 512";;
  dsads)    echo "--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=joint --bottleneck_agg_mode mean --n_fusion_distill 0 --head_mode gap --num_layers_per_modal 3 --num_layers 3 --transform sax --word_length 1 --dropout_signal uniform";;
  iemocap)  echo "--d_model 128 --nhead 8 --base_factor 5 --num_layers_per_modal 3 --num_layers 3 --internal_dim 128 --downsample_min_len 64 --n_fusion_distill 0 --head_mode gap --dropout 0.1 --batch_size 32 --lr 3e-5 --max_modality_drop 0.4 --use_sparse_attn False --fusion_mode joint --bottleneck_agg_mode mean --max_seq_len 128 --time_compression_ratio 4 --split_mode cross_subject";;
esac; }

JOBS=()
for f in 0 1 2; do JOBS+=("daliahar|$f" "pamap2|$f" "iemocap|$f"); done
for f in 0 1 2 3; do JOBS+=("dsads|$f"); done
echo "jobs: ${#JOBS[@]}"
run_job(){ local job="$1" g="$2" ds f main rec rd lg
  ds="${job%%|*}"; f="${job#*|}"; main="$(main_of "$ds")"; rec="$(recipe_of "$ds")"
  rd="$OUT/$ds/mbt"; lg="$LOG/${ds}_f${f}.log"
  compgen -G "$rd"/*/results_fold"$f".json >/dev/null 2>&1 && { echo "[skip] $ds f$f"; return; }
  echo "[$(date +%H:%M:%S)] START $ds f$f cuda:$g"
  ( cd "$TRAIN" && python "$main" $rec --fold "$f" --cuda_pick "cuda:$g" \
      --num_epochs $EP --results_dir "$rd" --exp_name "mbt_vanilla" ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $ds f$f"; }
QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop(){ exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF" 2>/dev/null)"; tail -n +2 "$QF" > "$QF.tmp" 2>/dev/null && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
worker(){ local g="$1" j; while :; do j="$(pop)"; [ -z "$j" ] && break; run_job "$j" "$g"; done; }
read -r -a WG <<< "${GPUS:-0 1 2 3}"
for g in "${WG[@]}"; do worker "$g" & done
wait; rm -f "$QF" "$QL"
echo "ALL VANILLA-MBT DONE $(date)"
