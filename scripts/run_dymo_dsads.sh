#!/bin/bash
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
RES=../results_3p3/dymo_proto_dsads
LOG=$RES/logs
for f in 0 1 2 3; do
  conda run -n maestro python main_dymo_proto.py --dataset dsads --fold "$f" --word_length 1 \
    --num_epochs 100 --cuda_pick "cuda:$f" \
    --results_dir "$RES" --exp_name dymo_proto >> "$LOG/dsads_f${f}.log" 2>&1 &
done
wait
cd /files1/haodong/MAESTRO_ttn_robustness_old/clean_models/train
conda run -n maestro python main_dymo_proto.py --dataset dsads --aggregate \
  --results_dir "$RES" --exp_name dymo_proto >> "$LOG/aggregate.log" 2>&1
echo "ALL DONE" >> "$LOG/aggregate.log"
