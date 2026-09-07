#!/usr/bin/env bash
# 3 RL ablations (zeroq / randorder / nostop) x 3 datasets (mmfi, cmi, czu_mhad),
# K=16 seqA_branch, in parallel over the roomy GPUs. Waits for all 9 to finish.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
rm -rf __pycache__
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16 PYTHONUNBUFFERED=1
mkdir -p logs
GPUS=(0 1 4 5)                     # 24GB/24GB/24GB/21GB free
declare -a JOBS=(
  "cmi zeroq"     "cmi randorder"     "cmi nostop"
  "czu_mhad zeroq" "czu_mhad randorder" "czu_mhad nostop"
  "mmfi zeroq"    "mmfi randorder"    "mmfi nostop"
)
pids=(); i=0
for job in "${JOBS[@]}"; do
  set -- $job; ds=$1; abl=$2
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  echo "launch: $ds/$abl -> GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_design.py --design dqn_qprior \
      --dataset "$ds" --ablation "$abl" > "logs/abl_${abl}_${ds}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} jobs; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== ALL DONE (failures: $fail) ==="
for job in "${JOBS[@]}"; do
  set -- $job; ds=$1; abl=$2
  f="results_k16/dqn_qprior_${abl}_${ds}_point.csv"
  if [ -f "$f" ]; then echo "  OK  $ds/$abl  ($(($(wc -l < "$f")-1)) rows)"; else echo "  MISSING $ds/$abl"; fi
done
