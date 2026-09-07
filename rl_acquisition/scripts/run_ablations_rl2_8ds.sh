#!/usr/bin/env bash
# All 5 ablations (zeroq / randorder / nostop / fullshap / fullshap_randorder) of the
# ADOPTED config (--res_l2 0.1) on all 8 datasets. 40 jobs, per-GPU sequential queues.
#
# cmi notes: its 4 TRAINING ablations get a dedicated GPU each (cheapest way to hide the
# ~13h/variant cost); its nostop only ROLLS OUT the main rl2 nets, so it waits for the
# main cmi/rl2 run (run_opt_8ds.sh, GPU 0) to finish writing them.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOGDIR="/tmp/rl_abl8_logs"
mkdir -p "$LOGDIR"

declare -A DS_FOLDS=([iemocap]="0,1,2" [cmi]="0,1,2" [czu_mhad]="0,1,2" [mmfi]="0,1,2" \
                     [dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")

run_one() {  # ds variant
  local ds=$1 variant=$2 extra=""
  case "$variant" in
    zeroq|randorder|nostop) extra="--ablation $variant" ;;
    fullshap)               extra="--advice_mode marginal" ;;
    fullshap_randorder)     extra="--advice_mode marginal --ablation randorder" ;;
  esac
  echo "[$(date +%T)] start $ds/$variant"
  "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
      --folds "${DS_FOLDS[$ds]}" --seeds "0,1,2" --res_l2 0.1 $extra \
      > "$LOGDIR/${ds}_${variant}.log" 2>&1
  echo "[$(date +%T)] done  $ds/$variant (rc=$?)"
}

ALL5="zeroq randorder nostop fullshap fullshap_randorder"

# small datasets first (fast), then one cmi training-ablation each
( export CUDA_VISIBLE_DEVICES=1; for v in $ALL5; do run_one dsads    $v; done; run_one cmi zeroq              ) > "$LOGDIR/w_gpu1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; for v in $ALL5; do run_one eav      $v; done; run_one cmi randorder          ) > "$LOGDIR/w_gpu2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; for v in $ALL5; do run_one pamap2   $v; done; run_one cmi fullshap           ) > "$LOGDIR/w_gpu3.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=4; for v in $ALL5; do run_one utd_mhad $v; done; run_one cmi fullshap_randorder ) > "$LOGDIR/w_gpu4.log" 2>&1 &
# GPU 7: the three mid-size datasets, then wait for main cmi/rl2 nets -> cmi nostop
( export CUDA_VISIBLE_DEVICES=7
  for v in $ALL5; do run_one mmfi     $v; done
  for v in $ALL5; do run_one iemocap  $v; done
  for v in $ALL5; do run_one czu_mhad $v; done
  echo "[$(date +%T)] waiting for main cmi/rl2 point.csv before cmi/nostop..."
  while [ ! -f results_k16/dqn_qprior_rl20.1_cmi_point.csv ]; do sleep 300; done
  run_one cmi nostop
) > "$LOGDIR/w_gpu7.log" 2>&1 &

wait
echo "=== ALL ABLATION WORKERS DONE ==="
for ds in iemocap cmi czu_mhad mmfi dsads eav pamap2 utd_mhad; do
  for f in "dqn_qprior_rl20.1_zeroq" "dqn_qprior_rl20.1_randorder" "dqn_qprior_rl20.1_nostop" \
           "dqn_qprior_fullshap_rl20.1" "dqn_qprior_fullshap_rl20.1_randorder"; do
    p="results_k16/${f}_${ds}_point.csv"
    [ -f "$p" ] && echo "  OK    $p" || echo "  MISS  $p"
  done
done
