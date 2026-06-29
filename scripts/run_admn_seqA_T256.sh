#!/usr/bin/env bash
# ADMN baseline built on the trained seqA backbone (faithful 2-stage), T=256.
# 6 jobs = {daliahar, wesad} x 3 folds. GPU-pinned workers + flock queue
# (one job per GPU). skip-done via results_fold{f}.json. Default GPUS="1 2 3"
# to leave GPU0 for the last 2 finishing T=256 folds.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
LOG="$HERE/results_3p3/T256/logs"; mkdir -p "$LOG"
GPUS="${GPUS:-1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
S1="${S1:-30}"; CTRL="${CTRL:-40}"   # from-seqA FT converges fast; longer overfits under shift

resdir() { [ "$1" = daliahar ] && echo "$HERE/results_3p3/T256/admn_seqA" \
                                || echo "$HERE/results_3p3/T256_wesad/admn_seqA"; }
fold_done() { [ -f "$(resdir "$1")/admn_seqA_T256/results_fold$2.json" ]; }

JOBS=()
for ds in daliahar wesad; do for f in 0 1 2; do
  fold_done "$ds" "$f" && { echo "[skip] $ds f$f"; continue; }
  JOBS+=("$ds|$f"); done; done
echo "remaining ADMN-seqA jobs: ${#JOBS[@]} -> ${JOBS[*]}"

run_one() {  # $1=ds $2=fold $3=gpu
  local ds="$1" f="$2" g="$3" lg="$LOG/admn_seqA_${ds}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START admn_seqA $ds f$f on cuda:$g"
  ( cd "$TRAIN" && python main_admn_seqA.py --dataset "$ds" --fold "$f" \
      --cuda_pick "cuda:$g" --stage1_epochs "$S1" --ctrl_epochs "$CTRL" \
      --results_dir "$(resdir "$ds")" --exp_name admn_seqA_T256 ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  admn_seqA $ds f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r ds f <<< "$job"; run_one "$ds" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== ADMN-seqA T256 ALL DONE $(date) =="
