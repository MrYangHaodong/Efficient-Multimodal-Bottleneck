#!/usr/bin/env bash
# Step I of ModalityDynMM-on-seqA: train the 3 expert seqA networks (1mod / partial
# / full) per dataset, on nested modality subsets chosen by val-LOO (dynmm_branches.json).
# 18 jobs = {daliahar,wesad} x {1mod,partial,full} x 3 folds. Same seqA T=256 recipe
# + --modalities <subset>. GPU-pinned flock queue. skip-done. T=256 (word_length=1).
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
BJSON="$TRAIN/dynmm_branches.json"
LOG="$HERE/results_3p3/T256/logs"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
EP="${EP:-100}"
RECIPE="--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3 --word_length 1"

main_of()  { [ "$1" = daliahar ] && echo main_v6_daliahar_seqfusion_late_fusion.py || echo main_v6_wesad_seqfusion_late_fusion.py; }
resroot()  { [ "$1" = daliahar ] && echo "$HERE/results_3p3/T256" || echo "$HERE/results_3p3/T256_wesad"; }
mods_of()  { python -c "import json;print(' '.join(json.load(open('$BJSON'))['$1']['$2']))"; }
resdir()   { echo "$(resroot "$1")/dynmm_experts/$2"; }
fold_done(){ compgen -G "$(resdir "$1" "$2")"/*/results_fold"$3".json >/dev/null 2>&1; }

JOBS=()
for ds in daliahar wesad; do for br in 1mod partial full; do for f in 0 1 2; do
  fold_done "$ds" "$br" "$f" && { echo "[skip] $ds/$br f$f"; continue; }
  JOBS+=("$ds|$br|$f"); done; done; done
echo "remaining expert jobs: ${#JOBS[@]} -> ${JOBS[*]}"

run_one() {  # $1=ds $2=branch $3=fold $4=gpu
  local ds="$1" br="$2" f="$3" g="$4" mods rd s lg
  mods="$(mods_of "$ds" "$br")"; rd="$(resdir "$ds" "$br")"; s="$(main_of "$ds")"
  lg="$LOG/dynmm_expert_${ds}_${br}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START expert $ds/$br f$f mods=[$mods] on cuda:$g"
  ( cd "$TRAIN" && python "$s" $RECIPE --modalities $mods --fold "$f" --cuda_pick "cuda:$g" \
      --num_epochs "$EP" --results_dir "$rd" --exp_name "dynmm_${br}_T256" ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  expert $ds/$br f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r ds br f <<< "$job"; run_one "$ds" "$br" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== DynMM experts (Step I) ALL DONE $(date) =="
