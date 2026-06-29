#!/usr/bin/env bash
# Step II of ModalityDynMM-on-seqA, ALL 4 datasets. Trains the per-sample gate over
# each dataset's 3 frozen expert seqA nets. 12 jobs = {daliahar,wesad,pamap2,iemocap}
# x 3 folds. A (ds,fold) runs only once its 3 experts (1mod/partial/full) exist.
# GPU-pinned flock queue, skip-done. Run AFTER the expert runners.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
LOG="$HERE/results_3p3/logs_dynmm"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
REG="${REG:-0.1}"; GEP="${GEP:-40}"

# per-dataset: resroot | gate main | exp_name | batch_size
declare -A RESROOT=( [daliahar]="results_3p3/T256" [wesad]="results_3p3/T256_wesad"
                     [pamap2]="results_3p3/dynmm_pamap2" [iemocap]="results_3p3/dynmm_iemocap" )
declare -A GMAIN=(  [daliahar]="main_dynmm_seqA_gate.py" [wesad]="main_dynmm_seqA_gate.py"
                    [pamap2]="main_dynmm_seqA_gate.py" [iemocap]="main_dynmm_seqA_gate_iemocap.py" )
declare -A EXPN=(   [daliahar]="dynmm_seqA_T256" [wesad]="dynmm_seqA_T256"
                    [pamap2]="dynmm_seqA_T128" [iemocap]="dynmm_seqA_T128" )
declare -A BS=(     [daliahar]=64 [wesad]=64 [pamap2]=64 [iemocap]=32 )

expert_ok(){ compgen -G "$HERE/${RESROOT[$1]}/dynmm_experts/$2"/*/best_model_fold"$3".json >/dev/null 2>&1 \
             || compgen -G "$HERE/${RESROOT[$1]}/dynmm_experts/$2"/*/results_fold"$3".json >/dev/null 2>&1; }
gate_done(){ [ -f "$HERE/${RESROOT[$1]}/dynmm_seqA/${EXPN[$1]}/results_fold$2.json" ]; }

JOBS=()
for ds in daliahar wesad pamap2 iemocap; do for f in 0 1 2; do
  gate_done "$ds" "$f" && { echo "[skip] gate $ds f$f"; continue; }
  if expert_ok "$ds" 1mod "$f" && expert_ok "$ds" partial "$f" && expert_ok "$ds" full "$f"; then
    JOBS+=("$ds|$f")
  else echo "[wait] experts not ready: $ds f$f"; fi
done; done
echo "runnable gate jobs: ${#JOBS[@]} -> ${JOBS[*]:-（none yet）}"
[ "${#JOBS[@]}" -eq 0 ] && { echo "No experts ready yet; rerun after experts finish."; exit 0; }

run_one() {  # $1=ds $2=fold $3=gpu
  local ds="$1" f="$2" g="$3" lg="$LOG/gate_${ds}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START gate $ds f$f on cuda:$g"
  ( cd "$TRAIN" && python "${GMAIN[$ds]}" --dataset "$ds" --fold "$f" --cuda_pick "cuda:$g" \
      --gate_epochs "$GEP" --reg "$REG" --batch_size "${BS[$ds]}" \
      --results_dir "$HERE/${RESROOT[$ds]}/dynmm_seqA" --exp_name "${EXPN[$ds]}" ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  gate $ds f$f"
}

QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for gpu in "${GPU_ARR[@]}"; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r ds f <<< "$job"; run_one "$ds" "$f" "$gpu"; done ) &
done
wait; rm -f "$QF" "$QL"
echo "== DynMM gate (Step II, all 4 ds) ALL DONE $(date) =="
