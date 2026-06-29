#!/usr/bin/env bash
# Wait until the daliahar T=256 campaign frees the GPUs (no daliahar word_length=1
# training procs), then run the WESAD T=256 campaign on all 4 GPUs.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[wesadT256-waiter] $(date) waiting for daliahar T=256 to finish..."
while [ "$(ps -eo args | awk '/word_length 1/ && (/_daliahar/ || /--dataset daliahar/) && $0 !~ /awk/' | wc -l)" -gt 0 ]; do
  sleep 60
done
echo "[wesadT256-waiter] $(date) daliahar T=256 done. GPUs:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GPUS="0 1 2 3" bash "$HERE/run_wesad_T256.sh"
echo "[wesadT256-waiter] $(date) WESAD T=256 finished."
