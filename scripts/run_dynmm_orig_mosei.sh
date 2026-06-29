#!/bin/bash
# Faithful original-DynMM (MultiBench Transformer + Concat late fusion, NO distillation), 3 nested
# branches, on CMU-MOSEI Acc-2. Seeds {7,21,365}. Phase A: 9 experts (3 branches x 3 seeds).
# Phase B: 3 gates. Per-seed isolation via dynmm_orig_mosei_s<seed>/. 4-GPU queues, resume-guarded.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
SEEDS=(7 21 365)
BR=(1mod partial full)
declare -A MODS=( [1mod]="text" [partial]="text vision" [full]="vision audio text" )
LOGD=../results_3p3/logs_dynmm_orig_mosei; mkdir -p "$LOGD"

JOBS=()
for s in "${SEEDS[@]}"; do for b in "${BR[@]}"; do JOBS+=("$s|$b"); done; done

phaseA(){
  local g=$1
  for i in "${!JOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r s b <<< "${JOBS[$i]}"
    local rd=../results_3p3/dynmm_orig_mosei_s${s}/experts/$b
    if [ -f "$rd/e_${b}/best_model.pth" ]; then echo "[gpu$g] SKIP expert s$s/$b"; continue; fi
    echo "[gpu$g] START expert s$s/$b mods=[${MODS[$b]}]"
    $PY main_dynmm_orig_expert_mosei.py --modalities ${MODS[$b]} --seed $s --num_epochs 30 \
        --batch_size 32 --max_seq_len 50 --tcr 1 --cuda_pick cuda:$g \
        --results_dir "$rd" --exp_name e_${b} > "$LOGD/expert_s${s}_${b}.log" 2>&1
    echo "[gpu$g] DONE  expert s$s/$b (exit $?)"
  done
}
echo "=== Phase A: ${#JOBS[@]} experts ==="
for g in 0 1 2 3; do phaseA "$g" & done
wait
echo "PHASE_A_DONE"

phaseB(){
  local g=$1
  for idx in "${!SEEDS[@]}"; do
    [ $(( idx % 4 )) -ne "$g" ] && continue
    local s=${SEEDS[$idx]}
    local out=../results_3p3/dynmm_orig_mosei_s${s}/gate
    if [ -f "$out/dynmm_orig_gate/results_fold0.json" ]; then echo "[gpu$g] SKIP gate s$s"; continue; fi
    echo "[gpu$g] START gate s$s"
    $PY main_dynmm_orig_gate_mosei.py --seed $s --gate_epochs 40 --reg 0.1 --batch_size 32 \
        --max_seq_len 50 --tcr 1 --cuda_pick cuda:$g \
        --results_dir "$out" --exp_name dynmm_orig_gate > "$LOGD/gate_s${s}.log" 2>&1
    echo "[gpu$g] DONE  gate s$s (exit $?)"
  done
}
echo "=== Phase B: ${#SEEDS[@]} gates ==="
for g in 0 1 2 3; do phaseB "$g" & done
wait
echo "ALL_DYNMM_ORIG_MOSEI_DONE"
