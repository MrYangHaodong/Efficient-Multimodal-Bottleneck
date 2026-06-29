#!/bin/bash
# ModalityDynMM-on-seqA for EAV. Phase I: 9 seqA experts (1mod/partial/full x 3 folds)
# on val-LOO modality subsets (dynmm_branches.json["eav"]). Phase II: 3 per-sample gates.
# 4-GPU queues, resume-guarded, detached.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
ROOT=../results_3p3/dynmm_eav
LOGD=$ROOT/logs; mkdir -p "$LOGD"
BJSON=dynmm_branches.json
mods_of(){ $PY -c "import json;print(' '.join(json.load(open('$BJSON'))['eav']['$1']))"; }

# ---------------- Phase I: experts ----------------
EJOBS=()
for br in 1mod partial full; do for f in 0 1 2; do EJOBS+=("$br|$f"); done; done

expert_queue() {
  local g=$1
  for i in "${!EJOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r br f <<< "${EJOBS[$i]}"
    local rd="$ROOT/dynmm_experts/$br"
    if ls $rd/*/results_fold$f.json >/dev/null 2>&1; then echo "[gpu$g] SKIP expert $br/f$f"; continue; fi
    local mods; mods="$(mods_of "$br")"
    echo "[gpu$g] START expert $br/f$f mods=[$mods]"
    $PY main_v6_eav_seqfusion_late_fusion.py --modalities $mods --fold $f --num_epochs 100 \
        --batch_size 32 --max_seq_len 128 --split_mode cross_subject --cuda_pick cuda:$g \
        --results_dir "$rd" --exp_name "dynmm_${br}_T128" > "$LOGD/expert_${br}_f${f}.log" 2>&1
    echo "[gpu$g] DONE  expert $br/f$f (exit $?)"
  done
}
echo "=== Phase I: ${#EJOBS[@]} experts ==="
for g in 0 1 2 3; do expert_queue "$g" & done
wait
echo "ALL_EAV_DYNMM_EXPERTS_DONE"

# ---------------- Phase II: gates ----------------
gate_queue() {
  local g=$1
  for f in 0 1 2; do
    [ $(( f % 4 )) -ne "$g" ] && continue
    if [ -f "$ROOT/dynmm_seqA/dynmm_seqA_eav/results_fold$f.json" ]; then echo "[gpu$g] SKIP gate f$f"; continue; fi
    echo "[gpu$g] START gate f$f"
    $PY main_dynmm_seqA_gate_eav.py --dataset eav --fold $f --cuda_pick cuda:$g \
        --gate_epochs 40 --reg 0.1 --batch_size 32 \
        --results_dir "$ROOT/dynmm_seqA" --exp_name dynmm_seqA_eav > "$LOGD/gate_f${f}.log" 2>&1
    echo "[gpu$g] DONE  gate f$f (exit $?)"
  done
}
echo "=== Phase II: 3 gates ==="
for g in 0 1 2 3; do gate_queue "$g" & done
wait
echo "ALL_EAV_DYNMM_DONE"
