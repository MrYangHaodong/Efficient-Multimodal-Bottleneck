#!/bin/bash
# Faithful original-DynMM (MultiBench Transformer + Concat, NO distillation), 3 nested branches,
# on iemocap/daliahar/pamap2/dsads/eav (per cross-subject fold). Phase A: experts. Phase B: gates.
# 4-GPU queues, resume-guarded, detached. Per-(ds,fold) isolation via dynmm_orig_<ds>/fold<f>/.
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
BJSON=dynmm_branches.json
BR=(1mod partial full)
declare -A NFOLD=( [iemocap]=3 [daliahar]=3 [pamap2]=3 [dsads]=4 [eav]=3 )
DSETS=(iemocap daliahar pamap2 dsads eav)
LOGD=../results_3p3/logs_dynmm_orig_5ds; mkdir -p "$LOGD"
mods_of(){ $PY -c "import json;print(' '.join(json.load(open('$BJSON'))['$1']['$2']))"; }

# ---- Phase A: experts (ds, fold, branch) ----
EJOBS=()
for ds in "${DSETS[@]}"; do for ((f=0; f<${NFOLD[$ds]}; f++)); do for b in "${BR[@]}"; do EJOBS+=("$ds|$f|$b"); done; done; done

expert_q(){
  local g=$1
  for i in "${!EJOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r ds f b <<< "${EJOBS[$i]}"
    local rd=../results_3p3/dynmm_orig_${ds}/fold${f}/experts/$b
    if [ -f "$rd/e_${b}/best_model.pth" ]; then echo "[gpu$g] SKIP $ds f$f $b"; continue; fi
    local m; m="$(mods_of "$ds" "$b")"
    echo "[gpu$g] START expert $ds f$f $b mods=[$m]"
    $PY main_dynmm_orig_expert_5ds.py --dataset $ds --modalities $m --fold $f --num_epochs 30 \
        --max_seq_len 128 --time_compression_ratio 4 --word_length 2 --split_mode cross_subject \
        --cuda_pick cuda:$g --results_dir "$rd" --exp_name e_${b} > "$LOGD/expert_${ds}_f${f}_${b}.log" 2>&1
    echo "[gpu$g] DONE  expert $ds f$f $b (exit $?)"
  done
}
echo "=== Phase A: ${#EJOBS[@]} experts ==="
for g in 0 1 2 3; do expert_q "$g" & done
wait
echo "PHASE_A_DONE"

# ---- Phase B: gates (ds, fold) ----
GJOBS=()
for ds in "${DSETS[@]}"; do for ((f=0; f<${NFOLD[$ds]}; f++)); do GJOBS+=("$ds|$f"); done; done

gate_q(){
  local g=$1
  for i in "${!GJOBS[@]}"; do
    [ $(( i % 4 )) -ne "$g" ] && continue
    IFS='|' read -r ds f <<< "${GJOBS[$i]}"
    local out=../results_3p3/dynmm_orig_${ds}/fold${f}/gate
    if [ -f "$out/dynmm_orig_gate/results_fold${f}.json" ]; then echo "[gpu$g] SKIP gate $ds f$f"; continue; fi
    echo "[gpu$g] START gate $ds f$f"
    $PY main_dynmm_orig_gate_5ds.py --dataset $ds --fold $f --gate_epochs 40 --reg 0.1 \
        --max_seq_len 128 --time_compression_ratio 4 --word_length 2 --split_mode cross_subject \
        --cuda_pick cuda:$g --results_dir "$out" --exp_name dynmm_orig_gate > "$LOGD/gate_${ds}_f${f}.log" 2>&1
    echo "[gpu$g] DONE  gate $ds f$f (exit $?)"
  done
}
echo "=== Phase B: ${#GJOBS[@]} gates ==="
for g in 0 1 2 3; do gate_q "$g" & done
wait
echo "ALL_DYNMM_ORIG_5DS_DONE"
