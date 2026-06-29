#!/usr/bin/env bash
# Retrain seqA + crossattn/multimodn/shaspec/decalign on PAMAP2 at FULL-RES T=512
# with CLEAN sax (not sax_noisy), cross-subject, 3 folds. Makes pamap2 consistent
# with daliahar/wesad (clean sax, full-res). seqA: --input_length_override 512;
# baselines: no override -> sequence_length=512. Results -> results_3p3/pamap2_T512/<method>/.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
RES="$HERE/results_3p3/pamap2_T512"; LOG="$RES/logs"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
read -r -a METHODS <<< "${METHODS_LIST:-seqA crossattn multimodn shaspec decalign}"

main_of() { case "$1" in
  seqA) echo "main_v6_pamap2_seqfusion_late_fusion.py";;
  crossattn) echo "main_crossattn_v6_pamap2.py";;
  multimodn) echo "main_multimodn_pamap2_dyna.py";;
  shaspec) echo "main_shaspec_pamap2_dyna.py";;
  decalign) echo "main_decalign_pamap2_dyna.py";;
esac; }
extra_of() { case "$1" in
  seqA) echo "--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3 --transform sax --input_length_override 512";;
  crossattn) echo "--num_layers_per_modal 3 --num_layers 3 --d_model 64 --sparse_attn_variant orig --transform sax";;
  multimodn) echo "--num_layers 3 --num_layers_fus 3 --d_model 64";;  # main has no --transform (sax hardcoded)
  shaspec) echo "--d_model 64";;                                       # main has no --transform (sax hardcoded)
  decalign) echo "--transform sax --batch_size 64";;
esac; }
resdir() { echo "$RES/$1"; }
fold_done() { compgen -G "$(resdir "$1")"/*/results_fold"$2".json >/dev/null 2>&1; }

echo "== preflight =="; PF=0
for m in "${METHODS[@]}"; do s="$(main_of "$m")"
  ( cd "$TRAIN" && CUDA_VISIBLE_DEVICES="" python "$s" --help >/dev/null 2>"$LOG/pf_${m}.err" ) \
    && echo "  ok $m" || { echo "  FAIL $m (see $LOG/pf_${m}.err)"; PF=1; }
done
[ "$PF" -ne 0 ] && { echo "Aborting."; exit 1; }

JOBS=()
for m in "${METHODS[@]}"; do for f in 0 1 2; do
  fold_done "$m" "$f" && { echo "[skip] $m f$f"; continue; }
  JOBS+=("$m|$f"); done; done
echo "pamap2 sax T512 jobs: ${#JOBS[@]} -> ${JOBS[*]}"

run_one() {  # $1=method $2=fold $3=gpu
  local m="$1" f="$2" g="$3" s ex lg; s="$(main_of "$m")"; ex="$(extra_of "$m")"; lg="$LOG/${m}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START $m f$f on cuda:$g"
  ( cd "$TRAIN" && python "$s" $ex --fold "$f" --cuda_pick "cuda:$g" \
      --num_epochs 100 --results_dir "$(resdir "$m")" --exp_name "${m}_pamap2_sax_T512" ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $m f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r m f <<< "$job"; run_one "$m" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== PAMAP2 sax T512 ALL DONE $(date) =="
