#!/usr/bin/env bash
# Wait until the DecAlign/ShaSpec campaign finishes (no main_decalign/main_shaspec
# training procs), smoke-test the risky QMF/PDF data paths, then launch the full
# QMF+PDF campaign on all 4 GPUs.
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh
conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/train"

# ---- 1. wait for decalign/shaspec to finish (ps+awk; awk-line excluded to avoid self-match) ----
echo "[waiter] $(date) waiting for DecAlign/ShaSpec to finish..."
while [ "$(ps -eo args | awk '/main_decalign_|main_shaspec_/ && $0 !~ /awk/' | wc -l)" -gt 0 ]; do
  sleep 60
done
echo "[waiter] $(date) DecAlign/ShaSpec done. GPUs:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ---- 2. smoke the risky paths: qmf+pdf daliahar, qmf pamap2 (input_length), qmf iemocap (d128) ----
SM=/tmp/smoke_qmfpdf; rm -rf "$SM"; mkdir -p "$SM"; ok=1
smoke() {  # $1=ds $2=fusion $3=extra
  echo "[waiter] smoke $2/$1"
  ( cd "$TRAIN" && python "main_qmfpdf_${1}_late_fusion.py" --fusion_method "$2" $3 \
      --fold 0 --cuda_pick cuda:0 --num_epochs 1 --results_dir "$SM/${2}_${1}" --exp_name smoke ) \
      > "$SM/${2}_${1}.log" 2>&1 || ok=0
  grep -iE "Parameters|End of Epoch|Val Acc|RuntimeError|size of tensor|OutOfMemory|Traceback" "$SM/${2}_${1}.log" | tail -3
}
smoke daliahar qmf ""
smoke daliahar pdf ""
smoke pamap2   qmf ""
smoke iemocap  qmf "--split_mode cross_subject"
if [ "$ok" -ne 1 ]; then
  echo "[waiter] SMOKE FAILED -- NOT launching full campaign. Inspect $SM/*.log"; exit 1
fi
echo "[waiter] smoke passed -> launching full QMF/PDF campaign on 4 GPUs"

# ---- 3. full campaign ----
GPUS="0 1 2 3" bash "$HERE/run_clean_qmf_pdf.sh"
echo "[waiter] $(date) QMF/PDF campaign finished."
