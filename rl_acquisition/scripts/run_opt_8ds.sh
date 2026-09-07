#!/usr/bin/env bash
# Combined-optimization retrain of dqn_qprior on all 8 datasets (K=16 runset).
# Adopted lever: --res_l2 0.1 (residual shrinkage toward the Shapley prior — helped 3/4
# new datasets, +0.029 B_star mean on pamap2, no meaningful harm).
# Also run --res_l2 0.1 --calibrate so the rl2-vs-rl2+calib pick per dataset is
# evidence-based (calibration alone helped dsads/eav/utd_mhad but hurt pamap2).
#
# Jobs (12): original 4 get rl2 AND rl2+calib; new 4 already have rl2-alone, so only
# rl2+calib. Sequential per dataset (one GPU each) to avoid OOM on the deep trees.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOGDIR="/tmp/rl_opt8_logs"
mkdir -p "$LOGDIR"

declare -A DS_FOLDS=([iemocap]="0,1,2" [cmi]="0,1,2" [czu_mhad]="0,1,2" [mmfi]="0,1,2" \
                     [dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")

run_one() {  # ds variant
  local ds=$1 variant=$2 extra=""
  case "$variant" in
    rl2)       extra="--res_l2 0.1" ;;
    rl2calib)  extra="--res_l2 0.1 --calibrate" ;;
  esac
  echo "[$(date +%T)] start $ds/$variant"
  "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
      --folds "${DS_FOLDS[$ds]}" --seeds "0,1,2" $extra \
      > "$LOGDIR/${ds}_${variant}.log" 2>&1
  echo "[$(date +%T)] done  $ds/$variant (rc=$?)"
}

# worker per GPU, sequential within
( export CUDA_VISIBLE_DEVICES=0; run_one cmi rl2;      run_one cmi rl2calib      ) > "$LOGDIR/w_gpu0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; run_one czu_mhad rl2; run_one czu_mhad rl2calib ) > "$LOGDIR/w_gpu1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; run_one iemocap rl2;  run_one iemocap rl2calib  ) > "$LOGDIR/w_gpu2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; run_one mmfi rl2;     run_one mmfi rl2calib     ) > "$LOGDIR/w_gpu3.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=4; run_one dsads rl2calib; run_one pamap2 rl2calib ) > "$LOGDIR/w_gpu4.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=7; run_one eav rl2calib; run_one utd_mhad rl2calib ) > "$LOGDIR/w_gpu7.log" 2>&1 &

wait
echo "=== ALL WORKERS DONE ==="
for f in results_k16/dqn_qprior_rl20.1_{iemocap,cmi,czu_mhad,mmfi}_point.csv \
         results_k16/dqn_qprior_calib_rl20.1_{iemocap,cmi,czu_mhad,mmfi,dsads,eav,pamap2,utd_mhad}_point.csv; do
  [ -f "$f" ] && echo "  OK    $f" || echo "  MISS  $f"
done
