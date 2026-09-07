#!/usr/bin/env bash
# FORCED-FIRST-PICK (§8.8 fix) campaign on the parity backbones, 4 new datasets.
#
# At depth 0 the only legal action is the highest-Shapley AVAILABLE modality; every
# later step stays fully dynamic. Enforced in the action mask, so it binds identically
# in training (FQI targets) and at test (both rollouts) — the deployed policy is exactly
# the one the Bellman targets were computed for.
#
# Reuses the k16p caches (backbone unchanged); nets retrain because CKPT_SUFFIX gains _ff.
# Outputs: results_k16p/<design>_ff[...]_<ds>_{sweep,point}.csv
#
# Usage: bash run_rl_parity_ff_4ds.sh [GPU_A GPU_B GPU_C GPU_D]

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/tmp/rl_ff_logs
mkdir -p "$LOG"
GPUS=(${1:-0} ${2:-1} ${3:-2} ${4:-3})
DS=(dsads eav pamap2 utd_mhad)
declare -A FOLDS=([dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")

design() {  # ds variant
  local ds=$1 v=$2 extra=""
  case "$v" in
    rl2)                extra="--res_l2 0.1" ;;
    rl2calib)           extra="--res_l2 0.1 --calibrate" ;;
    zeroq|randorder|nostop) extra="--res_l2 0.1 --ablation $v" ;;
    fullshap)           extra="--res_l2 0.1 --advice_mode marginal" ;;
    fullshap_randorder) extra="--res_l2 0.1 --advice_mode marginal --ablation randorder" ;;
  esac
  echo "[$(date +%T)] start $ds/$v"
  "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
      --folds "${FOLDS[$ds]}" --seeds "0,1,2" --force_first $extra \
      > "$LOG/${ds}_${v}.log" 2>&1
  echo "[$(date +%T)] done  $ds/$v (rc=$?)"
}

for i in "${!DS[@]}"; do
  ds=${DS[$i]}; g=${GPUS[$i]}
  (
    export CUDA_VISIBLE_DEVICES=$g
    for v in rl2 rl2calib nostop zeroq randorder fullshap fullshap_randorder; do
      design "$ds" "$v"
    done
  ) > "$LOG/W_${ds}.log" 2>&1 &
done
wait
echo "=== ALL FORCED-FIRST RUNS DONE ==="
ls results_k16p/*_ff*_point.csv 2>/dev/null | wc -l | xargs echo "point CSVs (expect 28):"
