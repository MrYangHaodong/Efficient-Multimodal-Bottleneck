#!/usr/bin/env bash
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RES="$HERE/results_3p3/dynmm_pamap2"
all_experts(){ for br in 1mod partial full; do for f in 0 1 2; do
  compgen -G "$RES/dynmm_experts/$br"/*/results_fold$f.json >/dev/null 2>&1 || return 1; done; done; }
echo "[gate] waiting for 9 experts..."
until all_experts; do sleep 60; done
echo "[gate] experts ready $(date); training gate"
for f in 0 1 2; do
  ( cd "$HERE/train" && python main_dynmm_seqA_gate.py --dataset pamap2 --fold $f --cuda_pick cuda:$f \
      --gate_epochs 40 --reg 0.1 --batch_size 64 \
      --results_dir "$RES/dynmm_seqA" --exp_name dynmm_seqA_T512sn ) > "$RES/logs/gate_f$f.log" 2>&1 &
done
wait
echo "== DynMM pamap2 T512sn gate DONE $(date) =="
