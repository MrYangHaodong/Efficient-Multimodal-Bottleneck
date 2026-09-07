#!/usr/bin/env bash
# FORCED-FIRST-PICK + res_l2 RL campaign — ALL 8 datasets, all ablations.
#
# Config per job:  --force_first --res_l2 0.1   (+ per-variant flags)
#   rl2 / rl2calib / nostop / zeroq / randorder / fullshap / fullshap_randorder
#
# Scheduling: a flock-guarded FIFO work queue, longest-job-first. CMI variants cost
# ~9.5 h each while EAV costs ~10 min, so a static per-GPU split would idle most of the
# fleet; with a shared queue every GPU pulls the next job the moment it frees up.
# Extra workers can be attached later (same queue) as other GPUs free up:
#     bash run_rl_ff_all8.sh --attach 0 2
#
# nostop reuses the rl2 net, so it waits on that dataset's rl2 sentinel instead of
# retraining a second copy.
#
# Usage: bash run_rl_ff_all8.sh <GPU...>          # create queue + start workers
#        bash run_rl_ff_all8.sh --attach <GPU...> # add workers to the existing queue

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Q=/tmp/rl_ff8_queue
LOG=/tmp/rl_ff8_logs
mkdir -p "$LOG"

declare -A FOLDS=([iemocap]="0,1,2" [cmi]="0,1,2" [czu_mhad]="0,1,2" [mmfi]="0,1,2" \
                  [dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")

ATTACH=0
if [[ "${1:-}" == "--attach" ]]; then ATTACH=1; shift; fi
GPUS=("$@")
[[ ${#GPUS[@]} -eq 0 ]] && { echo "usage: $0 [--attach] <GPU...>"; exit 1; }

# ── build the queue, longest first (cost = rough GPU-minutes, for ordering only) ──
if [[ $ATTACH -eq 0 ]]; then
  : > "$Q"
  {
    # CMI dominates: its 6 training variants head the queue so they start immediately.
    for v in rl2 rl2calib zeroq randorder fullshap fullshap_randorder; do echo "cmi $v"; done
    for v in rl2 rl2calib zeroq randorder fullshap fullshap_randorder; do echo "czu_mhad $v"; done
    for v in rl2 rl2calib zeroq randorder fullshap fullshap_randorder; do echo "iemocap $v"; done
    for v in rl2 rl2calib zeroq randorder fullshap fullshap_randorder; do echo "mmfi $v"; done
    # nostop last: cheap, and each waits for its dataset's rl2 sentinel
    for d in cmi czu_mhad iemocap mmfi; do echo "$d nostop"; done
  } >> "$Q"
  echo "queued $(wc -l < "$Q") jobs"
fi

pop() {  # atomically take the first queue line
  local line
  exec 9>"$Q.lock"; flock 9
  line=$(head -1 "$Q" 2>/dev/null)
  [[ -n "$line" ]] && sed -i '1d' "$Q"
  flock -u 9; exec 9>&-
  printf '%s' "$line"
}

worker() {
  local gpu=$1 job ds v extra
  export CUDA_VISIBLE_DEVICES=$gpu
  while true; do
    job=$(pop); [[ -z "$job" ]] && break
    set -- $job; ds=$1; v=$2
    if [[ "$v" == nostop ]]; then          # needs this dataset's rl2 net on disk
      while [[ ! -f "$LOG/.done_${ds}_rl2" ]]; do sleep 60; done
    fi
    case "$v" in
      rl2)                extra="--res_l2 0.1" ;;
      rl2calib)           extra="--res_l2 0.1 --calibrate" ;;
      zeroq|randorder|nostop) extra="--res_l2 0.1 --ablation $v" ;;
      fullshap)           extra="--res_l2 0.1 --advice_mode marginal" ;;
      fullshap_randorder) extra="--res_l2 0.1 --advice_mode marginal --ablation randorder" ;;
    esac
    echo "[$(date +%T)] GPU$gpu START $ds/$v"
    "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
        --folds "${FOLDS[$ds]}" --seeds "0,1,2" --force_first $extra \
        > "$LOG/${ds}_${v}.log" 2>&1
    rc=$?
    [[ $rc -eq 0 ]] && touch "$LOG/.done_${ds}_${v}"
    echo "[$(date +%T)] GPU$gpu END   $ds/$v (rc=$rc)"
  done
  echo "[$(date +%T)] GPU$gpu queue empty, worker exit"
}

for g in "${GPUS[@]}"; do
  worker "$g" >> "$LOG/worker_gpu${g}.log" 2>&1 &
done
echo "started ${#GPUS[@]} workers on GPUs: ${GPUS[*]}"
wait
echo "=== WORKERS FINISHED ==="
