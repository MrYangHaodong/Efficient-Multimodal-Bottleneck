#!/usr/bin/env bash
# One cache-build worker pinned to a GPU. Claims tasks (PRIO_ds_fold_split.task)
# from the shared queue via atomic mv and runs the explicit K=16 cache build.
#   Usage: cache_worker_k16.sh <GPU_ID>
set -u
GPU="$1"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CB="$BASE/cache/botneck_16"
Q="$CB/_queue"; CLAIM="$CB/_claim"; DONE="$CB/_done"; LOGS="$CB/_logs"; PREP="$CB/_prepdone"
PY=/home/egg8711/miniconda3/envs/maestro/bin/python

export RLB_RUNSET=k16 RLB_CACHE_SUBDIR=botneck_16
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
cd "$BASE"

echo "[worker$GPU] up $(date +%H:%M:%S)"
while true; do
  t=$(ls "$Q" 2>/dev/null | sort | head -1)
  if [ -z "$t" ]; then
    if [ "$(ls "$PREP" 2>/dev/null | wc -l)" -ge 4 ]; then
      echo "[worker$GPU] queue drained + all preps done -> exit"; break
    fi
    sleep 30; continue
  fi
  mv "$Q/$t" "$CLAIM/$t.gpu$GPU" 2>/dev/null || continue   # atomic claim; lost race -> retry
  base="${t%.task}"; base="${base#*_}"                     # strip PRIO_
  split="${base##*_}"; rest="${base%_*}"
  fold="${rest##*_}";  ds="${rest%_*}"                     # ds may contain '_' (czu_mhad)
  echo "[worker$GPU] BUILD $ds fold$fold $split  $(date +%H:%M:%S)"
  "$PY" -c "import environment as env; env.build_cache('$split', '$ds', $fold)" \
      > "$LOGS/${base}.gpu$GPU.log" 2>&1
  rc=$?
  mv "$CLAIM/$t.gpu$GPU" "$DONE/${t}.rc$rc"
  echo "[worker$GPU] DONE  $ds fold$fold $split  rc=$rc  $(date +%H:%M:%S)"
done
