#!/usr/bin/env bash
# Train V6 fusion-only (grad-norm asymmetric modality dropout) on IEMOCAP, 3 splits → aggregate.
# Usage:
#   bash run_iemocap_fusion_only_dyna.sh            # train all 3 splits in parallel
#   bash run_iemocap_fusion_only_dyna.sh --aggregate  # re-run aggregate only

set -euo pipefail
cd "$(dirname "$0")"

# ── configurable ──────────────────────────────────────────────────────────────
EXP_NAME="iemocap_fusion_only_dyna"
RESULTS_DIR="../model_chkpt/IEMOCAP/multimodal_model"
GPUS=(1 2)           # one GPU per split; adjust as needed

NUM_EPOCHS=100
BATCH_SIZE=16
NUM_WORKERS=4

D_MODEL=384
N_BOTTLENECKS=8
NUM_LAYERS=4
NUM_LAYERS_PER_MODAL=2

MAX_MOD_DROP=0.4
PROFILE_UPDATE_FREQ=5

LR=1e-4
LR_SCHEDULE=cosine
CLIP_GRAD=1.0
LOSS=ce
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="main_v6_iemocap_fusion_only_dyna.py"
CONDA_ENV="maestro"

run_split() {
    local S=$1
    local GPU=${GPUS[$S]}
    echo "[split $S] starting on cuda:${GPU}"
    conda run -n "${CONDA_ENV}" python "${SCRIPT}" \
        --split_id="${S}" \
        --cuda_pick="cuda:${GPU}" \
        --exp_name="${EXP_NAME}_s${S}" \
        --results_dir="${RESULTS_DIR}" \
        --num_epochs="${NUM_EPOCHS}" \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --d_model="${D_MODEL}" \
        --n_bottlenecks="${N_BOTTLENECKS}" \
        --num_layers="${NUM_LAYERS}" \
        --num_layers_per_modal="${NUM_LAYERS_PER_MODAL}" \
        --max_modality_drop="${MAX_MOD_DROP}" \
        --profile_update_freq="${PROFILE_UPDATE_FREQ}" \
        --lr="${LR}" \
        --lr_schedule="${LR_SCHEDULE}" \
        --clip_grad="${CLIP_GRAD}" \
        --loss="${LOSS}" \
        2>&1 | tee "../model_chkpt/IEMOCAP/multimodal_model/log_${EXP_NAME}_s${S}.txt"
    echo "[split $S] done"
}

if [[ "${1:-}" == "--aggregate" ]]; then
    echo "=== Aggregating results ==="
    conda run -n "${CONDA_ENV}" python "${SCRIPT}" \
        --aggregate \
        --exp_name="${EXP_NAME}" \
        --results_dir="${RESULTS_DIR}"
    exit 0
fi

mkdir -p "${RESULTS_DIR}"

# Launch all 3 splits in parallel
for S in 0 1 2; do
    run_split "${S}" &
done

echo "Waiting for all splits to finish..."
wait
echo "All splits done."

echo "=== Aggregating results ==="
conda run -n "${CONDA_ENV}" python "${SCRIPT}" \
    --aggregate \
    --exp_name="${EXP_NAME}" \
    --results_dir="${RESULTS_DIR}"
