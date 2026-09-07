#!/usr/bin/env bash
# Fine-grained lambda pass for the A+C policy: lambda in {1.2,1.4,1.6,1.8}.
# The coarse sweep showed the accuracy/cost knee sits between 1.0 and 2.0
# (marginal return drops from 0.48 to 0.20 F1 per unit MU across that span),
# so this resolves the frontier where the operating point actually gets chosen.
# Each lambda trains its own net; output tagged _lamfine so the coarse CSVs stand.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # module root (was a hard-coded /home/group path)
PY=/home/egg8711/miniconda3/envs/maestro/bin/python
export RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity
export PYTHONPATH="${RLB_CODE_REPO:-/files1/haodong/test/Efficient-Multimodal-Bottleneck}"   # backbone code tree (override with RLB_CODE_REPO)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=/tmp/lamfine_queue; LOG=/tmp/lamfine_logs; mkdir -p "$LOG"
declare -A FOLDS=([iemocap]="0,1,2" [cmi]="0,1,2" [czu_mhad]="0,1,2" [mmfi]="0,1,2" \
                  [dsads]="0,1,2,3" [eav]="0,1,2" [pamap2]="0,1,2,3" [utd_mhad]="0,1,2")
ATTACH=0; [[ "${1:-}" == "--attach" ]] && { ATTACH=1; shift; }
GPUS=("$@")
if [[ $ATTACH -eq 0 ]]; then
  : > "$Q"; for d in cmi czu_mhad iemocap mmfi dsads pamap2 eav utd_mhad; do echo "$d" >> "$Q"; done
  echo "queued $(wc -l < "$Q") datasets"
fi
pop(){ local l; exec 9>"$Q.lock"; flock 9; l=$(head -1 "$Q" 2>/dev/null)
       [[ -n "$l" ]] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; printf '%s' "$l"; }
worker(){ local gpu=$1 ds
  export CUDA_VISIBLE_DEVICES=$gpu
  while true; do ds=$(pop); [[ -z "$ds" ]] && break
    echo "[$(date +%T)] GPU$gpu START $ds"
    "$PY" run_design.py --design dqn_qprior --dataset "$ds" \
      --folds "${FOLDS[$ds]}" --seeds "0,1,2" \
      --force_first --exact_dp --stop_bce 1.0 --res_l2 0.1 \
      --lambdas "1.2,1.4,1.6,1.8" > "$LOG/${ds}.log" 2>&1
    rc=$?; [[ $rc -eq 0 ]] && touch "$LOG/.done_${ds}"
    echo "[$(date +%T)] GPU$gpu END   $ds (rc=$rc)"
  done; }
for g in "${GPUS[@]}"; do worker "$g" >> "$LOG/w_gpu${g}.log" 2>&1 & done
echo "started ${#GPUS[@]} workers on: ${GPUS[*]}"; wait
echo "=== LAMFINE DONE ==="
