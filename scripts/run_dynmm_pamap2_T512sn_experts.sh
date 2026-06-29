#!/usr/bin/env bash
# DynMM pamap2 experts @ T=512 sax_noisy (match seqA@512). 3 branches x 3 folds.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
BJSON="$TRAIN/dynmm_branches.json"; RES="$HERE/results_3p3/dynmm_pamap2"; LOG="$RES/logs"; mkdir -p "$LOG"
GPUS="${GPUS:-0 1 2 3}"; read -r -a GA <<< "$GPUS"
REC="--model_arch=v6 --d_model=64 --nhead=8 --base_factor=10 --batch_size=64 --lr=1e-4 --dropout=0.1 --max_modality_drop=0.4 --use_sparse_attn=False --fusion_mode=sequential --seq_random_order --num_layers_per_modal 3 --num_layers 3 --transform sax_noisy --input_length_override 512 --dropout_signal grad"
mods_of(){ python -c "import json;print(' '.join(json.load(open('$BJSON'))['pamap2']['$1']))"; }
resdir(){ echo "$RES/dynmm_experts/$1"; }
fold_done(){ compgen -G "$(resdir "$1")"/*/results_fold"$2".json >/dev/null 2>&1; }
JOBS=()
for br in 1mod partial full; do for f in 0 1 2; do fold_done "$br" "$f" && continue; JOBS+=("$br|$f"); done; done
echo "expert jobs: ${#JOBS[@]} -> ${JOBS[*]}"
run_one(){ local br="$1" f="$2" g="$3" mods rd lg; mods="$(mods_of "$br")"; rd="$(resdir "$br")"; lg="$LOG/expert_${br}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START $br f$f mods=[$mods] cuda:$g"
  ( cd "$TRAIN" && python main_v6_pamap2_seqfusion_late_fusion.py $REC --modalities $mods --fold "$f" --cuda_pick "cuda:$g" --num_epochs 100 --results_dir "$rd" --exp_name "dynmm_${br}_T512sn" ) >> "$lg" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $br f$f"; }
QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"; QL="${QF}.lock"
pop(){ exec 9>"$QL"; flock 9; local l; l="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$l"; }
for g in "${GA[@]}"; do ( while :; do j="$(pop)"; [ -z "$j" ] && break; IFS='|' read -r br f <<< "$j"; run_one "$br" "$f" "$g"; done ) & done
wait; rm -f "$QF" "$QL"; echo "== DynMM pamap2 T512sn experts DONE $(date) =="
