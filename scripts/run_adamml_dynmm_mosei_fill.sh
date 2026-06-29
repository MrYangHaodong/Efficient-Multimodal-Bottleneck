#!/bin/bash
# Fill missing MOSEI baselines: AdaMML (3 seeds) + DynMM full pipeline (3 seeds: experts + gate).
# Seeds {7,21,365} to match DMBF/MMEE. Per-seed expert isolation via dynmm_mosei_s<seed>/.
# 4-GPU queues, resume-guarded, detached.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
SEEDS=(7 21 365)
BR=(1mod partial full)
declare -A MODS=( [1mod]="text" [partial]="text vision" [full]="vision audio text" )
LOGD=../results_3p3/logs_adamml_dynmm_mosei_fill; mkdir -p "$LOGD"

# ---------------- Phase A: 9 DynMM experts + 3 AdaMML (12 jobs) ----------------
JOBS=()
for s in "${SEEDS[@]}"; do for b in "${BR[@]}"; do JOBS+=("expert|$s|$b"); done; done
for s in "${SEEDS[@]}"; do JOBS+=("adamml|$s|-"); done

phaseA(){
  local g=$1
  for i in "${!JOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r kind s b <<< "${JOBS[$i]}"
    if [ "$kind" = expert ]; then
      local rd=../results_3p3/dynmm_mosei_s${s}/dynmm_experts/$b
      if ls $rd/*/best_model_cold_fold0.pth >/dev/null 2>&1; then echo "[gpu$g] SKIP expert s$s/$b"; continue; fi
      echo "[gpu$g] START expert s$s/$b mods=[${MODS[$b]}]"
      $PY main_v6_mosei_seqfusion_late_fusion.py --modalities ${MODS[$b]} --fold 0 \
          --num_epochs 30 --batch_size 32 --max_seq_len 50 --seed_num $s --cuda_pick cuda:$g \
          --results_dir "$rd" --exp_name dynmm_${b}_mosei > "$LOGD/expert_s${s}_${b}.log" 2>&1
      echo "[gpu$g] DONE  expert s$s/$b (exit $?)"
    else
      local rd=../results_3p3/adamml_mosei
      if [ -f "$rd/adamml_s${s}/results_fold0.json" ]; then echo "[gpu$g] SKIP adamml s$s"; continue; fi
      echo "[gpu$g] START adamml s$s"
      $PY main_adamml_ours_mosei.py --dataset mosei --fold 0 \
          --warmup_epochs 40 --joint_epochs 40 --finetune_epochs 20 --batch_size 32 \
          --seed $s --cuda_pick cuda:$g --results_dir "$rd" --exp_name adamml_s${s} \
          > "$LOGD/adamml_s${s}.log" 2>&1
      echo "[gpu$g] DONE  adamml s$s (exit $?)"
    fi
  done
}
echo "=== Phase A: ${#JOBS[@]} jobs (9 experts + 3 adamml) ==="
for g in 0 1 2 3; do phaseA "$g" & done
wait
echo "PHASE_A_DONE"

# ---------------- Phase B: 3 DynMM gates (need experts) ----------------
phaseB(){
  local g=$1
  for idx in "${!SEEDS[@]}"; do
    [ $(( idx % 4 )) -ne "$g" ] && continue
    local s=${SEEDS[$idx]}
    local out=../results_3p3/dynmm_mosei_s${s}/dynmm_seqA
    if [ -f "$out/dynmm_seqA_mosei/results_fold0.json" ]; then echo "[gpu$g] SKIP gate s$s"; continue; fi
    echo "[gpu$g] START gate s$s"
    $PY main_dynmm_seqA_gate_mosei.py --dataset mosei --fold 0 --cuda_pick cuda:$g \
        --gate_epochs 40 --reg 0.1 --batch_size 32 --seed $s \
        --results_dir "$out" --exp_name dynmm_seqA_mosei > "$LOGD/gate_s${s}.log" 2>&1
    echo "[gpu$g] DONE  gate s$s (exit $?)"
  done
}
echo "=== Phase B: ${#SEEDS[@]} gates ==="
for g in 0 1 2 3; do phaseB "$g" & done
wait
echo "ALL_ADAMML_DYNMM_MOSEI_FILL_DONE"
