#!/usr/bin/env bash
# FULL RL campaign on the ARCHITECTURE-PARITY seqA_branch backbones (2026-08-01,
# enc3/fus3/n_fusion_distill=0) for the 4 new datasets.
#
# Runset k16p keeps everything separate from the k16 (distill=4) campaign:
#   caches       cache/botneck_16_parity/
#   policies     checkpoints_k16p/
#   results      results_k16p/
#
# Phases (barrier between A and B: W_ADV must be measured on the NEW caches):
#   A  make_valloo -> rebuild_caches                        (per dataset, 4 GPUs)
#   A' calib_w_adv_k16p.py -> patches W_ADV_MAP_K16P
#   B  8 design runs per dataset, sequential within a GPU:
#        baselines
#        dqn_qprior                       --res_l2 0.1      (adopted lever)
#        dqn_qprior  --calibrate          --res_l2 0.1
#        ablations zeroq / randorder / nostop                --res_l2 0.1
#        fullshap (advice_mode marginal) [+ randorder]       --res_l2 0.1
#      nostop only rolls out the main net, so it runs after it.
#
# Usage: bash run_rl_parity_4ds.sh [GPU_A GPU_B GPU_C GPU_D]

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
rm -rf __pycache__
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/tmp/rl_parity_logs
mkdir -p "$LOG"

GPUS=(${1:-1} ${2:-3} ${3:-6} ${4:-7})
DS=(dsads eav pamap2 utd_mhad)
declare -A FOLDS=([dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")

# ── Phase A: valloo + caches ────────────────────────────────────────────────
for i in "${!DS[@]}"; do
  ds=${DS[$i]}; g=${GPUS[$i]}
  (
    export CUDA_VISIBLE_DEVICES=$g
    echo "[$(date +%T)] $ds A1 make_valloo (GPU $g)"
    "$PY" make_valloo_k16.py "$ds" > "$LOG/${ds}_valloo.log" 2>&1 || exit 1
    echo "[$(date +%T)] $ds A2 rebuild_caches"
    "$PY" rebuild_caches.py "$ds"  > "$LOG/${ds}_cache.log"  2>&1 || exit 1
    echo "[$(date +%T)] $ds phase A done"
  ) > "$LOG/A_${ds}.log" 2>&1 &
done
wait
echo "=== PHASE A COMPLETE ==="
for ds in "${DS[@]}"; do tail -1 "$LOG/A_${ds}.log"; done

# ── Phase A': recalibrate W_ADV on the parity caches ────────────────────────
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" calib_w_adv_k16p.py > "$LOG/w_adv.log" 2>&1
echo "=== W_ADV RECALIBRATED ==="; cat "$LOG/w_adv.log"

# ── Phase B: all designs ────────────────────────────────────────────────────
design() {  # ds variant
  local ds=$1 v=$2 extra="" design=dqn_qprior
  case "$v" in
    baselines)          design=baselines; extra="" ;;
    rl2)                extra="--res_l2 0.1" ;;
    rl2calib)           extra="--res_l2 0.1 --calibrate" ;;
    zeroq|randorder|nostop) extra="--res_l2 0.1 --ablation $v" ;;
    fullshap)           extra="--res_l2 0.1 --advice_mode marginal" ;;
    fullshap_randorder) extra="--res_l2 0.1 --advice_mode marginal --ablation randorder" ;;
  esac
  echo "[$(date +%T)] start $ds/$v"
  "$PY" run_design.py --design $design --dataset "$ds" \
      --folds "${FOLDS[$ds]}" --seeds "0,1,2" $extra \
      > "$LOG/${ds}_${v}.log" 2>&1
  echo "[$(date +%T)] done  $ds/$v (rc=$?)"
}

for i in "${!DS[@]}"; do
  ds=${DS[$i]}; g=${GPUS[$i]}
  (
    export CUDA_VISIBLE_DEVICES=$g
    for v in baselines rl2 rl2calib nostop zeroq randorder fullshap fullshap_randorder; do
      design "$ds" "$v"
    done
  ) > "$LOG/B_${ds}.log" 2>&1 &
done
wait

echo "=== ALL PARITY RL DONE ==="
for ds in "${DS[@]}"; do
  for v in baselines rl2 rl2calib nostop zeroq randorder fullshap fullshap_randorder; do
    case "$v" in
      baselines)          f="baselines" ;;
      rl2)                f="dqn_qprior_rl20.1" ;;
      rl2calib)           f="dqn_qprior_calib_rl20.1" ;;
      fullshap)           f="dqn_qprior_fullshap_rl20.1" ;;
      fullshap_randorder) f="dqn_qprior_fullshap_rl20.1_randorder" ;;
      *)                  f="dqn_qprior_rl20.1_${v}" ;;
    esac
    p="results_k16p/${f}_${ds}_point.csv"
    [ -f "$p" ] && echo "  OK    $p" || echo "  MISS  $p"
  done
done
