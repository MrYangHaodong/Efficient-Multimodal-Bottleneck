#!/usr/bin/env python3
"""Submit current Section-3.2 seqA training jobs to Volt H100 workers."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f730bb1630cb603f6905f48142f831d15745cf91"
IMAGE = (
    "nvcr.io/nvidia/pytorch:26.07-py3@sha256:"
    "2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
)
S3_DATA_ROOT = "s3://volt-dev-user-shared-data/FUSELab/payal/SPARQ/datasets"
S3_CHECKPOINT_ROOT = "s3://volt-dev-user-shared-data/FUSELab/payal/SPARQ/checkpoints"

# These are the runnable current-branch recipes whose datasets are present in the
# audited SPARQ S3 prefix.  All explicitly enable the Section-3.2 ingredients:
# learned bottleneck gate, random order, uniform prefix DS/KL, and no EMA.
DATASETS = {
    "iemocap": {
        "s3_name": "IEMOCAP",
        "s3_objects": 60087,
        "s3_bytes": 44970649927,
        "seed": 239,
        "args": [
            "--batch_size", "32", "--lr", "3e-5", "--max_seq_len", "128",
            "--time_compression_ratio", "4", "--split_mode", "cross_subject",
            "--d_model", "128", "--nhead", "8", "--base_factor", "5",
            "--num_layers", "3", "--num_layers_per_modal", "3",
            "--internal_dim", "128", "--n_bottlenecks", "8",
            "--downsample_min_len", "64", "--n_fusion_distill", "2",
            "--use_sparse_attn", "False", "--head_mode", "gap",
        ],
    },
    "eav": {
        "s3_name": "EAV",
        "data_subdir": "EAV",
        "s3_objects": 32875,
        "s3_bytes": 54921024588,
        "seed": 239,
        "args": [
            "--batch_size", "32", "--lr", "1e-4", "--max_seq_len", "128",
            "--time_compression_ratio", "4", "--split_mode", "cross_subject",
            "--d_model", "128", "--nhead", "8", "--base_factor", "10",
            "--num_layers", "4", "--num_layers_per_modal", "2",
            "--internal_dim", "128", "--n_bottlenecks", "8",
            "--n_fusion_distill", "2", "--use_sparse_attn", "True",
            "--head_mode", "gap",
        ],
    },
    "dsads": {
        "s3_name": "DSADS",
        "s3_objects": 16,
        "s3_bytes": 26298948772,
        "seed": 169,
        "args": [
            "--batch_size", "64", "--lr", "1e-4", "--word_length", "1",
            "--transform", "sax", "--d_model", "64", "--nhead", "8",
            "--base_factor", "10", "--num_layers", "3",
            "--num_layers_per_modal", "3", "--n_bottlenecks", "8",
            "--n_fusion_distill", "2", "--use_sparse_attn", "False",
            "--dropout_signal", "uniform", "--head_mode", "gap",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--project", default="arm-aidp-research")
    parser.add_argument("--cluster", default="aws-2")
    parser.add_argument("--machine-type", default="p5.48xlarge")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--wandb-entity", default="arm-aair-ben")
    parser.add_argument("--wandb-project", default="adaptive-modality-acquisition")
    parser.add_argument("--run-date", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--work-root", type=Path,
                        default=Path("/private/tmp/sparq-manuscript-training"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def copy_code(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_path: str, names: list[str]) -> set[str]:
        excluded = {
            ".git", ".volt-jobs", "__pycache__", ".DS_Store", "results",
            "results_3p3", "model_chkpt", "logs", "runs", "wandb",
        }
        return excluded.intersection(names)

    shutil.copytree(REPO_ROOT, destination, ignore=ignore)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in parts)


def render_config(args: argparse.Namespace, dataset: str, fold: int) -> str:
    spec = DATASETS[dataset]
    campaign = args.campaign or f"sparq-manuscript-{SOURCE_COMMIT[:7]}-h100-{args.run_date}"
    run_name = f"{campaign}-{dataset}-f{fold}-s{spec['seed']}"
    job_name = re.sub(r"[^a-z0-9-]", "-", run_name)[:63]
    data_uri = f"{S3_DATA_ROOT}/{spec['s3_name']}/"
    checkpoint_uri = f"{S3_CHECKPOINT_ROOT}/{campaign}/{dataset}/fold{fold}/"
    data_root = f"/volt/data/{spec['s3_name']}"
    if spec.get("data_subdir"):
        data_root += f"/{spec['data_subdir']}"
    train_args = [
        "python", "training/seqA.py",
        "--dataset", dataset,
        "--fold", str(fold),
        "--num_epochs", str(args.epochs),
        "--seed_num", str(spec["seed"]),
        "--data_root", data_root,
        "--results_dir", "/volt/artifacts/checkpoints",
        "--exp_name", "manuscript_repro",
        "--fusion_mode", "sequential",
        "--bottleneck_agg_mode", "gate",
        "--seq_random_order",
        "--prefix_supervision",
        "--prefix_ds_weight", "1.0",
        "--prefix_kd_weight", "1.0",
        "--max_modality_drop", "0.4",
        "--dropout", "0.1",
        "--num_workers", "4",
        "--cuda_pick", "cuda:0",
    ] + spec["args"]

    return f"""project: {args.project}
name: {job_name}
cluster: {args.cluster}
image: {args.image}
timeout: 96h
resources:
  machine_type: {args.machine_type}
  num_gpus: 1
  disk_gb: 512
env:
  PYTHONUNBUFFERED: "1"
  SOURCE_COMMIT: "{SOURCE_COMMIT}"
  DATASET_S3_URI: "{data_uri}"
  CHECKPOINT_S3_URI: "{checkpoint_uri}"
  WANDB_ENTITY: "{args.wandb_entity}"
  WANDB_PROJECT: "{args.wandb_project}"
  WANDB_RUN_GROUP: "{campaign}"
  WANDB_RUN_NAME: "{run_name}"
  WANDB_DIR: "/volt/artifacts/wandb"
  WANDB_CACHE_DIR: "/volt/cache/wandb"
  WANDB_INIT_TIMEOUT: "120"
credential_grants:
  - ref: volt-dev-s3-aidp-research
    provider: aws
    expose:
      mode: aws-default-chain
secrets:
  - ref: wandb-api-key-paymoh01
    expose:
      as_env: WANDB_API_KEY
command:
  - bash
  - -lc
  - |
    set -euo pipefail
    cd /volt/code
    mkdir -p "/volt/data/{spec['s3_name']}" /volt/artifacts/checkpoints \
      /volt/artifacts/wandb /volt/cache/wandb

    save_checkpoints() {{
      find /volt/artifacts/checkpoints -type f -print0 2>/dev/null \
        | sort -z | xargs -0 -r sha256sum > /volt/artifacts/SHA256SUMS
      python volt/s3_transfer.py upload-directory \
        /volt/artifacts/checkpoints "$CHECKPOINT_S3_URI" || true
    }}
    trap save_checkpoints EXIT

    python -m pip install --no-cache-dir \
      'boto3>=1.35' 'wandb>=0.19,<1' 'scikit-learn>=1.5' \
      'thop>=0.1.1' 'tensorboard>=2.17' 'ipython>=8' \
      'pandas>=2' 'scipy>=1.13' 'matplotlib>=3.9' 'tqdm>=4.66'
    python - <<'PY'
    import boto3, torch, wandb, sklearn, thop
    assert torch.cuda.is_available(), "CUDA is unavailable"
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "wandb:", wandb.__version__)
    print("AWS credential source validated:", bool(boto3.Session().get_credentials()))
    PY

    python volt/s3_transfer.py download-prefix \
      "$DATASET_S3_URI" "/volt/data/{spec['s3_name']}" \
      --expected-objects {spec['s3_objects']} --expected-bytes {spec['s3_bytes']}
    test -n "$(find "/volt/data/{spec['s3_name']}" -type f -print -quit)"

    {shell_join(train_args)} 2>&1 | tee /volt/artifacts/train.log
"""


def main() -> int:
    args = parse_args()
    if any(fold < 0 for fold in args.folds):
        raise SystemExit("folds must be non-negative")
    args.work_root.mkdir(parents=True, exist_ok=True)
    code_dir = args.work_root / "code"
    copy_code(code_dir)

    for dataset in args.datasets:
        for fold in args.folds:
            config_path = args.work_root / f"{dataset}-fold{fold}.yaml"
            config_path.write_text(render_config(args, dataset, fold))
            command = ["volt", "job", "submit", str(config_path), "--tar", str(code_dir)]
            print("+", " ".join(shlex.quote(x) for x in command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
