#!/usr/bin/env bash
# Train RL policy (dqn_qprior) on 4 new datasets: dsads, eav, pamap2, utd_mhad.
# Uses seqA_branched K=16 checkpoints from runs/dropout_robustness_seqA_dropaug/.
#
# Pipeline per dataset (GPU-parallel):
#   Stage 1: make_valloo_k16.py  — val-LOO order + per-modality GFLOPs npz
#   Stage 2: rebuild_caches.py   — depth-M prefix-tree cache (val + test)
#   Stage 3: run_design.py       — baselines + dqn_qprior (3 folds x 3 seeds)
#
# GPU assignments:
#   GPU 2 → dsads (M=5, heaviest cache)
#   GPU 5 → eav   (M=3, lightest)
#   GPU 6 → pamap2 (M=4)
#   GPU 7 → utd_mhad (M=4)
#
# Usage:
#   bash run_rl_4ds.sh
#   GPUS="0 1 2 3" bash run_rl_4ds.sh    # override GPU assignments

set -eo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maestro

LOGDIR="/tmp/rl_4ds_logs"
mkdir -p "$LOGDIR"

PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16
export RLB_CACHE_SUBDIR=botneck_16
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU assignments: ds -> GPU
declare -A DS_GPU=([dsads]=2 [eav]=5 [pamap2]=6 [utd_mhad]=7)
declare -A DS_FOLDS=([dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")

PIDS=()

for DS in dsads eav pamap2 utd_mhad; do
  G="${DS_GPU[$DS]}"
  FOLDS="${DS_FOLDS[$DS]}"
  (
    export CUDA_VISIBLE_DEVICES="$G"
    export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16

    echo "[$(date +%T)] [$DS GPU$G] Stage 1: make_valloo"
    "$PY" make_valloo_k16.py "$DS" \
      >> "$LOGDIR/${DS}_valloo.log" 2>&1
    echo "[$(date +%T)] [$DS GPU$G] Stage 1 done"

    echo "[$(date +%T)] [$DS GPU$G] Stage 2: rebuild caches"
    "$PY" rebuild_caches.py "$DS" \
      >> "$LOGDIR/${DS}_cache.log" 2>&1
    echo "[$(date +%T)] [$DS GPU$G] Stage 2 done"

    echo "[$(date +%T)] [$DS GPU$G] Stage 3: RL designs (folds=$FOLDS)"
    "$PY" run_design.py --design baselines --dataset "$DS" \
      --folds "$FOLDS" --seeds "0,1,2" \
      >> "$LOGDIR/${DS}_design_baselines.log" 2>&1
    "$PY" run_design.py --design dqn_qprior --dataset "$DS" \
      --folds "$FOLDS" --seeds "0,1,2" \
      >> "$LOGDIR/${DS}_design_dqn.log" 2>&1
    echo "[$(date +%T)] [$DS GPU$G] Stage 3 done"

    echo "[$(date +%T)] [$DS GPU$G] ALL DONE"
  ) >> "$LOGDIR/${DS}_launcher.log" 2>&1 &
  PIDS+=($!)
  echo "Launched $DS on GPU $G (PID=${PIDS[-1]})"
done

echo ""
echo "4 workers launched: ${PIDS[*]}"
echo "Logs: $LOGDIR/<ds>_{valloo,cache,design_*}.log"
echo ""

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "[WARN] PID $pid exited with error"; FAIL=1; }
done

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "=== All 4 datasets complete. Results in $BASE/results_k16/ ==="
else
  echo "=== Some datasets failed — check $LOGDIR/*.log ==="
fi
