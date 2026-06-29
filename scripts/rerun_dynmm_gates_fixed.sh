#!/usr/bin/env bash
set -u
source /home/egg8711/miniconda3/etc/profile.d/conda.sh; conda activate maestro
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
LOG=/files1/haodong/MAESTRO_ttn_robustness_old/clean_models/results_3p3/dynmm_gate_refix.log
: > "$LOG"
# dataset|resroot(rel ../results_3p3)|exp_name|nfolds|word_length
run_gate(){ local ds="$1" res="$2" ex="$3" nf="$4" wl="$5" f g
  for ((f=0; f<nf; f++)); do
    g=$((f % 4))
    ( python main_dynmm_seqA_gate.py --dataset "$ds" --fold "$f" --cuda_pick "cuda:$g" \
        --word_length "$wl" --gate_epochs 40 --reg 0.1 --batch_size 64 \
        --results_dir "$res" --exp_name "$ex" ) >> "$LOG" 2>&1 &
  done; wait
  echo "== gate done: $ds ==" >> "$LOG"
}
run_gate daliahar ../results_3p3/T256/dynmm_seqA        dynmm_seqA_T256   3 1
run_gate wesad    ../results_3p3/T256_wesad/dynmm_seqA  dynmm_seqA_T256   3 1
run_gate pamap2   ../results_3p3/dynmm_pamap2/dynmm_seqA dynmm_seqA_T512sn 3 1
run_gate dsads    ../results_3p3/dynmm_dsads/dynmm_seqA  dynmm_seqA_dsads  4 1
echo "ALL GATES DONE $(date)" >> "$LOG"
