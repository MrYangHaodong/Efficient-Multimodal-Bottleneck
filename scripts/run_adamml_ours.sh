#!/usr/bin/env bash
# AdaMML baseline (whole-sample modality selection, per-modality 6-layer Transformer
# + logits late fusion + Gumbel policy) on the HAR datasets. 9 jobs =
# {daliahar,wesad,pamap2} x 3 folds. GPU-pinned flock queue, skip-done.
# (iemocap handled separately — feats-list data.) gamma = efficiency penalty (sweep via GAMMA env).
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
LOG="$HERE/results_3p3/logs_adamml"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
GAMMA="${GAMMA:-0.5}"; WU="${WU:-40}"; JT="${JT:-40}"; FT="${FT:-20}"
DM_OF()   { [ "$1" = iemocap ] && echo 128 || echo 64; }
BS_OF()   { [ "$1" = iemocap ] && echo 32 || echo 64; }
MAIN_OF() { [ "$1" = iemocap ] && echo main_adamml_ours_iemocap.py || echo main_adamml_ours.py; }
DSLIST="${DSLIST:-daliahar wesad pamap2 iemocap}"

resdir()  { echo "$HERE/results_3p3/adamml_$1/adamml_ours"; }
fold_done(){ [ -f "$(resdir "$1")/results_fold$2.json" ]; }

JOBS=()
for ds in $DSLIST; do for f in 0 1 2; do
  fold_done "$ds" "$f" && { echo "[skip] $ds f$f"; continue; }
  JOBS+=("$ds|$f"); done; done
echo "AdaMML jobs: ${#JOBS[@]} -> ${JOBS[*]}"

run_one() {  # $1=ds $2=fold $3=gpu
  local ds="$1" f="$2" g="$3" lg="$LOG/adamml_${ds}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START adamml $ds f$f on cuda:$g"
  ( cd "$TRAIN" && python "$(MAIN_OF "$ds")" --dataset "$ds" --fold "$f" --cuda_pick "cuda:$g" \
      --d_model "$(DM_OF "$ds")" --num_layers 6 --warmup_epochs "$WU" --joint_epochs "$JT" \
      --finetune_epochs "$FT" --gamma "$GAMMA" --batch_size "$(BS_OF "$ds")" \
      --results_dir "$HERE/results_3p3/adamml_${ds}" --exp_name adamml_ours ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  adamml $ds f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r ds f <<< "$job"; run_one "$ds" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== AdaMML (HAR) ALL DONE $(date) =="
