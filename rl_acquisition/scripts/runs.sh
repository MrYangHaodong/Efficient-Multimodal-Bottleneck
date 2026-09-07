#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline for ONE dataset: build the depth-M cache, then run every design.
#
#      ./runs.sh <dataset> <gpu>
#      ./runs.sh iemocap 7
#      ./runs.sh cmi 6
#
#  Use run_all.sh to launch both in parallel on separate GPUs.
#
#  Stage 1  rebuild_caches.py  — MAX_DEPTH = M (protocol §1). iemocap ~15-35 min,
#           cmi ~3-7 h. Skipped automatically if the cache already exists.
#  Stage 2  run_design.py x6   — baselines + 5 RL designs, 3 folds x 3 seeds x A0/A20/A40.
#
#  Per design (protocol §12):
#      results/<design>_<ds>_point.csv   headline at B_star
#      results/<design>_<ds>_sweep.csv   full lever frontier
#      figures/<design>_<ds>_freq.png    modality-selection frequency
#      figures/<design>_<ds>_first.png   first-pick distribution
#
#  Shared by every design: val-fit (§2), class-balanced reward + |A|-normalised cost (§5),
#  availability in the state (§10), MAX_DEPTH = M (§1).
# ═══════════════════════════════════════════════════════════════════════════════
set -o pipefail   # NOT -u: conda's activate-gcc script references an unbound SYS_SYSROOT

DS="${1:?usage: ./runs.sh <dataset> <gpu>}"
GPU="${2:?usage: ./runs.sh <dataset> <gpu>}"
FOLDS="${FOLDS:-0,1,2}"
SEEDS="${SEEDS:-0,1,2}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # shared box: avoid fragmentation OOM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-maestro_energy}"
cd "$(dirname "$0")"

DESIGNS=(baselines dqn_qprior)
DEPTH=$(python -c "import environment as e; print(e.depth_of('$DS'))")

echo "═══════════════════════════════════════════════════════════════════"
echo " $DS   GPU=$GPU   depth=$DEPTH (=M)   folds=$FOLDS   seeds=$SEEDS   fit=val"
echo " started $(date)"
echo "═══════════════════════════════════════════════════════════════════"

# ── Stage 1: cache ────────────────────────────────────────────────────────────
if python -c "import environment as e; e.get_cache('val','$DS',0)" >/dev/null 2>&1; then
  echo "[stage1] depth-$DEPTH cache present — skipping build"
else
  echo "[stage1] building depth-$DEPTH cache ($(date +%H:%M:%S)) ..."
  python rebuild_caches.py "$DS" || { echo "[FATAL] cache build failed"; exit 1; }
fi

# ── Stage 2: designs ──────────────────────────────────────────────────────────
FAILED=()
for D in "${DESIGNS[@]}"; do
  echo ""
  echo "─── $D  ($(date +%H:%M:%S)) ──────────────────────────────────────"
  if python run_design.py --design "$D" --dataset "$DS" --folds "$FOLDS" --seeds "$SEEDS"; then
    echo "[done] $D"
  else
    echo "[FAILED] $D — continuing"; FAILED+=("$D")
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " $DS finished $(date)"
[ ${#FAILED[@]} -eq 0 ] && echo " all designs OK" || echo " FAILED: ${FAILED[*]}"
echo " results: $(ls -1 results/*_${DS}_*.csv 2>/dev/null | wc -l) csv   figures: $(ls -1 figures/*_${DS}_*.png 2>/dev/null | wc -l) png"
echo "═══════════════════════════════════════════════════════════════════"
