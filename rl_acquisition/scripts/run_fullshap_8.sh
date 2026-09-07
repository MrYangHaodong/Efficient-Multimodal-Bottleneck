#!/usr/bin/env bash
# FULL-Shapley (marginal advice + per-dataset W_ADV) K=16 policies: full + random-ordering,
# on all 4 datasets. λ-sweep + B* on val (same protocol as the conditional baselines already
# in results_k16/). Waits for all 8. Outputs:
#   results_k16/dqn_qprior_fullshap_<ds>_{point,sweep}.csv            (full policy)
#   results_k16/dqn_qprior_fullshap_randorder_<ds>_{point,sweep}.csv  (random-ordering)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
rm -rf __pycache__
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16 PYTHONUNBUFFERED=1
mkdir -p logs
GPUS=(0 1 2 5 6)
declare -a JOBS=(
  "iemocap full" "mmfi full" "cmi full" "czu_mhad full"
  "iemocap randorder" "mmfi randorder" "cmi randorder" "czu_mhad randorder"
)
pids=(); i=0
for job in "${JOBS[@]}"; do
  set -- $job; ds=$1; variant=$2
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  abl=""; tag="full"
  [ "$variant" = "randorder" ] && { abl="--ablation randorder"; tag="randorder"; }
  echo "launch fullshap $ds/$tag -> GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
      --advice_mode marginal $abl \
      > "logs/fullshap_${ds}_${tag}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} full-Shapley jobs; waiting..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== ALL DONE (failures: $fail) ==="
for job in "${JOBS[@]}"; do
  set -- $job; ds=$1; variant=$2
  tag="full"; [ "$variant" = "randorder" ] && tag="randorder"
  sfx=""; [ "$variant" = "randorder" ] && sfx="_randorder"
  f="results_k16/dqn_qprior_fullshap${sfx}_${ds}_point.csv"
  [ -f "$f" ] && echo "  OK  $ds/$tag ($(($(wc -l <"$f")-1)) rows)" || echo "  MISSING $ds/$tag"
done
