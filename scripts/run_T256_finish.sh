#!/usr/bin/env bash
# Finish the remaining T=256 folds (decalign + dymo_proto, daliahar+wesad) by
# running each missing fold on its OWN GPU in parallel (faster per fold ->
# better chance to complete before the recurring external kill; each fold is
# independent + saves results_fold{N}.json on completion, so progress persists).
# decalign first (the slow bottleneck). Usage: bash run_T256_finish.sh
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"
RESD="$HERE/results_3p3/T256"; RESW="$HERE/results_3p3/T256_wesad"
LOG="$HERE/results_3p3/T256/logs"; mkdir -p "$LOG"

fold_done() { compgen -G "$1"/*/results_fold"$2".json >/dev/null 2>&1; }

JOBS=()
for f in 0 1 2; do
  fold_done "$RESD/decalign" "$f" || JOBS+=("decalign|daliahar|$f")
  fold_done "$RESW/decalign" "$f" || JOBS+=("decalign|wesad|$f")
done
for f in 0 1 2; do
  fold_done "$RESD/dymo_proto" "$f" || JOBS+=("dymo_proto|daliahar|$f")
  fold_done "$RESW/dymo_proto" "$f" || JOBS+=("dymo_proto|wesad|$f")
done
echo "remaining folds: ${#JOBS[@]} -> ${JOBS[*]}"

run_one() {  # $1=method $2=ds $3=fold $4=gpu
  local m="$1" ds="$2" f="$3" g="$4" res
  [ "$ds" = daliahar ] && res="$RESD" || res="$RESW"
  local lg="$LOG/finish_${m}_${ds}_f${f}.log"
  echo "[$(date +%H:%M:%S)] START $m/$ds f$f on cuda:$g"
  if [ "$m" = decalign ]; then
    local bs="--batch_size 128"; [ "$ds" = wesad ] && bs="--batch_size 32"  # bumped (probed safe alone on a GPU)
    ( cd "$TRAIN" && python "main_decalign_${ds}_dyna.py" --transform sax --word_length 1 $bs \
        --fold "$f" --num_epochs 100 --cuda_pick "cuda:$g" \
        --results_dir "$res/decalign" --exp_name decalign_T256 ) >> "$lg" 2>&1
  else
    ( cd "$TRAIN" && python main_dymo_proto.py --dataset "$ds" --word_length 1 \
        --fold "$f" --num_epochs 100 --cuda_pick "cuda:$g" \
        --results_dir "$res/dymo_proto" --exp_name dymo_proto_T256 ) >> "$lg" 2>&1
  fi
  echo "[$(date +%H:%M:%S)] DONE  $m/$ds f$f"
}

# GPU-pinned workers + flock queue: each GPU runs at most ONE job at a time
# (avoids the slot%%4 co-location that OOM'd two 30GB jobs onto one GPU).
QF="$(mktemp)"; printf '%s\n' "${JOBS[@]}" > "$QF"
QL="${QF}.lock"
pop() { exec 9>"$QL"; flock 9; local line; line="$(head -1 "$QF")"; tail -n +2 "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"; flock -u 9; echo "$line"; }
for g in 0 1 2 3; do
  ( while :; do job="$(pop)"; [ -z "$job" ] && break
      IFS='|' read -r m ds f <<< "$job"; run_one "$m" "$ds" "$f" "$g"; done ) &
done
wait; rm -f "$QF" "$QF.lock"
echo "== T256 FINISH (decalign+dymo_proto) COMPLETE $(date) =="
