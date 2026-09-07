#!/usr/bin/env bash
# Fullshap (marginal advice) + ablation (zeroq/randorder/nostop) runs for the
# 4 new datasets: dsads, eav, pamap2, utd_mhad.
# Mirrors what run_fullshap_8.sh + run_ablations_3ds.sh did for the original 4.
#
# Jobs per dataset:
#   fullshap          --advice_mode marginal              (uses calibrated W_ADV_MAP)
#   fullshap_randorder --advice_mode marginal --ablation randorder
#   zeroq             --ablation zeroq
#   randorder         --ablation randorder
#   nostop            --ablation nostop
#
# GPU assignments (5 jobs per dataset, 4 datasets = 20 jobs; round-robin over 4 GPUs):
#   GPU 2, 5, 6, 7

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
rm -rf __pycache__
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOGDIR="/tmp/rl_fullshap_4ds_logs"
mkdir -p "$LOGDIR" logs

GPUS=(2 5 6 7)

declare -a JOBS=(
  "dsads fullshap"           "dsads fullshap_randorder"  "dsads zeroq"     "dsads randorder"   "dsads nostop"
  "eav fullshap"             "eav fullshap_randorder"    "eav zeroq"       "eav randorder"     "eav nostop"
  "pamap2 fullshap"          "pamap2 fullshap_randorder" "pamap2 zeroq"    "pamap2 randorder"  "pamap2 nostop"
  "utd_mhad fullshap"        "utd_mhad fullshap_randorder" "utd_mhad zeroq" "utd_mhad randorder" "utd_mhad nostop"
)

pids=(); i=0
for job in "${JOBS[@]}"; do
  set -- $job; ds=$1; variant=$2
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))

  case "$variant" in
    fullshap)
      args="--advice_mode marginal"
      label="fullshap"
      ;;
    fullshap_randorder)
      args="--advice_mode marginal --ablation randorder"
      label="fullshap_randorder"
      ;;
    zeroq|randorder|nostop)
      args="--ablation $variant"
      label="$variant"
      ;;
  esac

  folds="0,1,2,3"
  [[ "$ds" == "eav" || "$ds" == "utd_mhad" ]] && folds="0,1,2"

  echo "launch: $ds/$label -> GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_design.py --design dqn_qprior \
      --dataset "$ds" --folds "$folds" --seeds "0,1,2" \
      $args \
      > "$LOGDIR/${ds}_${label}.log" 2>&1 &
  pids+=($!)
done

echo "Launched ${#pids[@]} jobs; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "[WARN] PID $p failed"; fail=$((fail+1)); }; done

echo ""
echo "=== ALL DONE (failures: $fail) ==="
for job in "${JOBS[@]}"; do
  set -- $job; ds=$1; variant=$2
  case "$variant" in
    fullshap)           f="results_k16/dqn_qprior_fullshap_${ds}_point.csv" ;;
    fullshap_randorder) f="results_k16/dqn_qprior_fullshap_randorder_${ds}_point.csv" ;;
    *)                  f="results_k16/dqn_qprior_${variant}_${ds}_point.csv" ;;
  esac
  if [ -f "$f" ]; then
    echo "  OK    $ds/$variant  ($(($(wc -l < "$f")-1)) rows)"
  else
    echo "  MISS  $ds/$variant"
  fi
done
