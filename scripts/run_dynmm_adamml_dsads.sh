#!/usr/bin/env bash
# DSADS: AdaMML (4 scenarios) + DynMM (3 seqA experts x 4 scenarios -> gate).
# Phase 1 queue (4 GPUs): 4 AdaMML + 12 experts = 16 independent jobs.
# Phase 2 queue (4 GPUs): 4 DynMM gates (need their fold's 3 experts).
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
BJSON="$TRAIN/dynmm_branches.json"
ADARES="$HERE/results_3p3/adamml_dsads"
DYNRES="$HERE/results_3p3/dynmm_dsads"
LOG="$DYNRES/logs"; mkdir -p "$LOG" "$ADARES"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GA <<< "$GPUS"
REC="--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3 --transform sax --word_length 1 --dropout_signal uniform"
mods_of(){ python -c "import json;print(' '.join(json.load(open('$BJSON'))['dsads']['$1']))"; }

run_job(){  # "ada|f" or "exp|br|f"
  local job="$1" g="$2" kind f br mods rd lg
  kind="${job%%|*}"
  if [ "$kind" = ada ]; then
    f="${job#ada|}"; lg="$LOG/adamml_f${f}.log"
    [ -f "$ADARES/adamml_ours/results_fold${f}.json" ] && { echo "[skip] ada f$f"; return; }
    echo "[$(date +%H:%M:%S)] START ada f$f cuda:$g"
    ( cd "$TRAIN" && python main_adamml_ours.py --dataset dsads --fold "$f" --cuda_pick "cuda:$g" \
        --d_model 64 --num_layers 6 --warmup_epochs 40 --joint_epochs 40 --finetune_epochs 20 \
        --gamma 0.5 --batch_size 64 --results_dir "$ADARES" --exp_name adamml_ours ) >> "$lg" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  ada f$f"
  else
    br="$(echo "$job" | cut -d'|' -f2)"; f="$(echo "$job" | cut -d'|' -f3)"
    rd="$DYNRES/dynmm_experts/$br"; lg="$LOG/expert_${br}_f${f}.log"
    compgen -G "$rd"/*/results_fold"$f".json >/dev/null 2>&1 && { echo "[skip] exp $br f$f"; return; }
    mods="$(mods_of "$br")"
    echo "[$(date +%H:%M:%S)] START exp $br f$f mods=[$mods] cuda:$g"
    ( cd "$TRAIN" && python main_v6_dsads_seqfusion_late_fusion.py $REC --modalities $mods --fold "$f" \
        --cuda_pick "cuda:$g" --num_epochs 100 --results_dir "$rd" --exp_name "dynmm_${br}_dsads" ) >> "$lg" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  exp $br f$f"
  fi
}

# ---- build phase-1 job list ----
JOBS=()
for f in 0 1 2 3; do JOBS+=("ada|$f"); done
for br in 1mod partial full; do for f in 0 1 2 3; do JOBS+=("exp|$br|$f"); done; done
echo "phase1 jobs: ${#JOBS[@]}"
QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop(){ exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF" 2>/dev/null)"; tail -n +2 "$QF" > "$QF.tmp" 2>/dev/null && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
worker(){ local g="$1" j; while :; do j="$(pop)"; [ -z "$j" ] && break; run_job "$j" "$g"; done; }
for g in "${GA[@]}"; do worker "$g" & done; wait
rm -f "$QF" "$QL"
echo "== phase1 (AdaMML + experts) DONE $(date) =="

# ---- phase 2: gates ----
for f in 0 1 2 3; do
  g="${GA[$((f % ${#GA[@]}))]}"
  ( cd "$TRAIN" && python main_dynmm_seqA_gate.py --dataset dsads --fold "$f" --cuda_pick "cuda:$g" \
      --word_length 1 --gate_epochs 40 --reg 0.1 --batch_size 64 \
      --results_dir "$DYNRES/dynmm_seqA" --exp_name dynmm_seqA_dsads ) > "$LOG/gate_f${f}.log" 2>&1 &
done
wait
echo "== phase2 (DynMM gate) DONE $(date) =="
echo "ALL DONE $(date)"
