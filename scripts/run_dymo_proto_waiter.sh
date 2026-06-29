#!/usr/bin/env bash
# Wait until decalign/shaspec AND qmf/pdf campaigns finish, GPU-smoke the faithful
# DyMo (incl. the costly wesad M=10 per-subset Gaussian + greedy), then run the
# full dymo_proto campaign on 4 GPUs.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TRAIN="$HERE/train"

echo "[dymoproto-waiter] $(date) waiting for decalign/shaspec/qmfpdf to finish..."
while [ "$(ps -eo args | awk '/main_decalign_|main_shaspec_|main_qmfpdf_/ && $0 !~ /awk/' | wc -l)" -gt 0 ]; do
  sleep 60
done
echo "[dymoproto-waiter] $(date) prior campaigns done. GPUs:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# GPU-smoke risky paths (1 epoch): daliahar(sax), pamap2(sax_noisy,T512), wesad(M=10 offline+greedy), iemocap(d128)
SM=/tmp/dymoproto_gpu_smoke; rm -rf "$SM"; mkdir -p "$SM"; ok=1
for ds in daliahar pamap2 wesad iemocap; do
  extra=""; [ "$ds" = "iemocap" ] && extra="--split_mode cross_subject"
  echo "[dymoproto-waiter] smoke $ds"
  ( cd "$TRAIN" && python main_dymo_proto.py --dataset "$ds" $extra --fold 0 --cuda_pick cuda:0 \
      --num_epochs 1 --results_dir "$SM/$ds" --exp_name smoke ) > "$SM/$ds.log" 2>&1 || ok=0
  grep -iE "select_acc|avg_selected|test_acc|RuntimeError|OutOfMemory|Traceback" "$SM/$ds.log" | tail -3
done
if [ "$ok" -ne 1 ]; then echo "[dymoproto-waiter] SMOKE FAILED -- not launching. See $SM/*.log"; exit 1; fi
echo "[dymoproto-waiter] smoke passed -> launching full dymo_proto campaign"
GPUS="0 1 2 3" bash "$HERE/run_clean_dymo_proto.sh"
echo "[dymoproto-waiter] $(date) dymo_proto campaign finished."
