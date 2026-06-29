#!/usr/bin/env bash
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
cd "$(dirname "${BASH_SOURCE[0]}")/train"
RES=/files1/haodong/MAESTRO_ttn_robustness_old/clean_models/results_3p3
DEC() { python main_decalign_wesad_dyna.py --fold "$1" --cuda_pick "cuda:$2" --num_epochs 100 \
         --results_dir "$RES/decalign_wesad" --exp_name decalign_aligned; }
SHA() { python main_shaspec_wesad_dyna.py --fold "$1" --cuda_pick "cuda:$2" --num_epochs 100 \
         --results_dir "$RES/shaspec_wesad" --exp_name shaspec_aligned; }
L=$RES/logs
( DEC 1 0 ) >> "$L/decalign_wesad.log" 2>&1 &
( DEC 2 1 ) >> "$L/decalign_wesad.log" 2>&1 &
( SHA 0 2 ; SHA 2 2 ) >> "$L/shaspec_wesad.log" 2>&1 &
( SHA 1 3 ) >> "$L/shaspec_wesad.log" 2>&1 &
wait
echo "### wesad folds done -> aggregating $(date)"
python main_decalign_wesad_dyna.py --aggregate --results_dir "$RES/decalign_wesad" --exp_name decalign_aligned >> "$L/decalign_wesad.log" 2>&1
python main_shaspec_wesad_dyna.py  --aggregate --results_dir "$RES/shaspec_wesad"  --exp_name shaspec_aligned  >> "$L/shaspec_wesad.log" 2>&1
echo "### WESAD PARALLEL COMPLETE $(date)"
