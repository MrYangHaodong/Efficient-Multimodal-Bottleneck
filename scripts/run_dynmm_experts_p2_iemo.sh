#!/usr/bin/env bash
# Step I of ModalityDynMM-on-seqA for pamap2 + iemocap (T=128, each dataset's own
# seqA recipe). 18 jobs = {pamap2,iemocap} x {1mod,partial,full} x 3 folds, subsets
# from dynmm_branches.json (val-LOO). GPU-pinned flock queue, skip-done.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
BJSON="$TRAIN/dynmm_branches.json"
LOG="$HERE/results_3p3/logs_dynmm"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
EP="${EP:-100}"
PAMAP2_RECIPE="--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3 --transform sax_noisy"
IEMO_RECIPE="--model_arch=v6 --max_seq_len=128 --base_factor=5 --lr=3e-5 --dropout=0.1 --max_modality_drop=0.4 --batch_size=32 --split_mode=cross_subject --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3"

main_of()  { [ "$1" = pamap2 ] && echo main_v6_pamap2_seqfusion_late_fusion.py || echo main_v6_iemocap_seqfusion_late_fusion.py; }
recipe_of(){ [ "$1" = pamap2 ] && echo "$PAMAP2_RECIPE" || echo "$IEMO_RECIPE"; }
resroot()  { echo "$HERE/results_3p3/dynmm_$1"; }
mods_of()  { python -c "import json;print(' '.join(json.load(open('$BJSON'))['$1']['$2']))"; }
resdir()   { echo "$(resroot "$1")/dynmm_experts/$2"; }
fold_done(){ compgen -G "$(resdir "$1" "$2")"/*/results_fold"$3".json >/dev/null 2>&1; }

JOBS=()
for ds in pamap2 iemocap; do for br in 1mod partial full; do for f in 0 1 2; do
  fold_done "$ds" "$br" "$f" && { echo "[skip] $ds/$br f$f"; continue; }
  JOBS+=("$ds|$br|$f"); done; done; done
echo "remaining p2/iemo expert jobs: ${#JOBS[@]} -> ${JOBS[*]}"

run_one() {  # $1=ds $2=branch $3=fold $4=gpu
  local ds="$1" br="$2" f="$3" g="$4" mods rd s rec lg
  mods="$(mods_of "$ds" "$br")"; rd="$(resdir "$ds" "$br")"; s="$(main_of "$ds")"; rec="$(recipe_of "$ds")"
  lg="$LOG/expert_${ds}_${br}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START expert $ds/$br f$f mods=[$mods] on cuda:$g"
  ( cd "$TRAIN" && python "$s" $rec --modalities $mods --fold "$f" --cuda_pick "cuda:$g" \
      --num_epochs "$EP" --results_dir "$rd" --exp_name "dynmm_${br}_T128" ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  expert $ds/$br f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r ds br f <<< "$job"; run_one "$ds" "$br" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== DynMM p2/iemo experts (Step I) ALL DONE $(date) =="
