#!/usr/bin/env bash
# A -> B -> C strategy campaign, all 8 datasets.
#
#   A  --exact_dp    exact backward induction over the COMPLETE prefix tree replaces the
#                    Fitted-Q bootstrap (verified against a closed form to 4e-8)
#   B  (framing)     supported by the randorder ablation: fixed order + learned stopping
#   C  --stop_bce 1  decision-aware BCE on the stop-vs-acquire margin against the exact-DP
#                    optimal stop label — trains the one signal that is actually learnable
#   plus the previously adopted levers: --force_first, --res_l2 0.1
#
# METHOD  = ff + dp + sb + rl2
# ABLATE  = drop-dp (isolates A) / drop-sb (isolates C) / randorder (isolates ordering, B)
#           / zeroq / nostop / fullshap
#
# Shared flock queue, longest-first: all 8 MAIN runs head the queue so the headline lands
# first; CMI (~9.5 h/variant) is queued ahead of everything within each block.
#
# Usage: bash run_rl_abc_all8.sh <GPU...>          |  bash run_rl_abc_all8.sh --attach <GPU...>

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Q=/tmp/rl_abc_queue
LOG=/tmp/rl_abc_logs
mkdir -p "$LOG"
BASE="--force_first --res_l2 0.1"

declare -A FOLDS=([iemocap]="0,1,2" [cmi]="0,1,2" [czu_mhad]="0,1,2" [mmfi]="0,1,2" \
                  [dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")
ORDER=(cmi czu_mhad iemocap mmfi dsads pamap2 eav utd_mhad)   # slowest first

ATTACH=0
if [[ "${1:-}" == "--attach" ]]; then ATTACH=1; shift; fi
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && { echo "usage: $0 [--attach] <GPU...>"; exit 1; }

if [[ $ATTACH -eq 0 ]]; then
  : > "$Q"
  for d in "${ORDER[@]}"; do echo "$d main"      >> "$Q"; done   # headline first
  for d in "${ORDER[@]}"; do echo "$d randorder" >> "$Q"; done   # evidence for B
  for d in "${ORDER[@]}"; do echo "$d nodp"      >> "$Q"; done   # isolates A
  for d in "${ORDER[@]}"; do echo "$d nosb"      >> "$Q"; done   # isolates C
  for d in "${ORDER[@]}"; do echo "$d zeroq"     >> "$Q"; done
  for d in "${ORDER[@]}"; do echo "$d fullshap"  >> "$Q"; done
  for d in "${ORDER[@]}"; do echo "$d nostop"    >> "$Q"; done   # reuses main net
  echo "queued $(wc -l < "$Q") jobs"
fi

pop() { local l; exec 9>"$Q.lock"; flock 9
        l=$(head -1 "$Q" 2>/dev/null); [[ -n "$l" ]] && sed -i '1d' "$Q"
        flock -u 9; exec 9>&-; printf '%s' "$l"; }

worker() {
  local gpu=$1 job ds v extra
  export CUDA_VISIBLE_DEVICES=$gpu
  while true; do
    job=$(pop); [[ -z "$job" ]] && break
    set -- $job; ds=$1; v=$2
    [[ "$v" == nostop ]] && while [[ ! -f "$LOG/.done_${ds}_main" ]]; do sleep 60; done
    case "$v" in
      main)      extra="--exact_dp --stop_bce 1.0" ;;
      randorder) extra="--exact_dp --stop_bce 1.0 --ablation randorder" ;;
      nodp)      extra="--stop_bce 0.0" ;;                       # FQI bootstrap (A removed)
      nosb)      extra="--exact_dp" ;;                           # DP only  (C removed)
      zeroq)     extra="--exact_dp --stop_bce 1.0 --ablation zeroq" ;;
      fullshap)  extra="--exact_dp --stop_bce 1.0 --advice_mode marginal" ;;
      nostop)    extra="--exact_dp --stop_bce 1.0 --ablation nostop" ;;
    esac
    echo "[$(date +%T)] GPU$gpu START $ds/$v"
    "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
        --folds "${FOLDS[$ds]}" --seeds "0,1,2" $BASE $extra \
        > "$LOG/${ds}_${v}.log" 2>&1
    rc=$?; [[ $rc -eq 0 ]] && touch "$LOG/.done_${ds}_${v}"
    echo "[$(date +%T)] GPU$gpu END   $ds/$v (rc=$rc)"
  done
  echo "[$(date +%T)] GPU$gpu worker exit"
}

for g in "${GPUS[@]}"; do worker "$g" >> "$LOG/worker_gpu${g}.log" 2>&1 & done
echo "started ${#GPUS[@]} workers on GPUs: ${GPUS[*]}"
wait
echo "=== ABC CAMPAIGN FINISHED ==="
