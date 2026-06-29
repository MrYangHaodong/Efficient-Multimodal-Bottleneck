#!/usr/bin/env bash
# Step II of ModalityDynMM-on-seqA: train the per-sample gate over the 3 frozen
# expert seqA nets. 6 jobs = {daliahar,wesad} x 3 folds. Only runs a (ds,fold)
# once all 3 experts (1mod/partial/full) for that fold are trained. GPU-pinned
# flock queue, skip-done. Run AFTER run_dynmm_experts_T256.sh.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
LOG="$HERE/results_3p3/T256/logs"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
REG="${REG:-0.1}"; GEP="${GEP:-40}"

resroot()    { [ "$1" = daliahar ] && echo "$HERE/results_3p3/T256" || echo "$HERE/results_3p3/T256_wesad"; }
gate_resdir(){ echo "$(resroot "$1")/dynmm_seqA"; }
expert_ok()  { compgen -G "$(resroot "$1")/dynmm_experts/$2"/*/best_model_fold"$3".json >/dev/null 2>&1 \
               || compgen -G "$(resroot "$1")/dynmm_experts/$2"/*/results_fold"$3".json >/dev/null 2>&1; }
gate_done()  { [ -f "$(gate_resdir "$1")/dynmm_seqA_T256/results_fold$2.json" ]; }

JOBS=()
for ds in daliahar wesad; do for f in 0 1 2; do
  gate_done "$ds" "$f" && { echo "[skip] gate $ds f$f"; continue; }
  if expert_ok "$ds" 1mod "$f" && expert_ok "$ds" partial "$f" && expert_ok "$ds" full "$f"; then
    JOBS+=("$ds|$f")
  else echo "[wait] experts not ready: $ds f$f"; fi
done; done
echo "runnable gate jobs: ${#JOBS[@]} -> ${JOBS[*]:-（none yet）}"
[ "${#JOBS[@]}" -eq 0 ] && { echo "No experts ready; rerun after experts finish."; exit 0; }

run_one() {  # $1=ds $2=fold $3=gpu
  local ds="$1" f="$2" g="$3" lg="$LOG/dynmm_gate_${ds}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START gate $ds f$f on cuda:$g"
  ( cd "$TRAIN" && python main_dynmm_seqA_gate.py --dataset "$ds" --fold "$f" --cuda_pick "cuda:$g" \
      --gate_epochs "$GEP" --reg "$REG" --results_dir "$(gate_resdir "$ds")" --exp_name dynmm_seqA_T256 ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  gate $ds f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r ds f <<< "$job"; run_one "$ds" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== DynMM gate (Step II) ALL DONE $(date) =="
