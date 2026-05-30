#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_DIR="configs/rebuild/cross_pattern_1500_500_mse_regression"
LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/cross_pattern_1500_500_no_rpm_lateconcat_mse_regression_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/cross_pattern_1500_500_no_rpm_lateconcat_mse_regression_queue.done"
BASE_PORT="${CROSSPAT_MSE_REG_BASE_PORT:-30911}"

RUN_NAMES=(
  "crosspat_train345_test1_no_rpm_lateconcat_mse_reg_lr1e5_ep35"
  "crosspat_train134_test5_no_rpm_lateconcat_mse_reg_lr1e5_ep35"
  "crosspat_train135_test4_no_rpm_lateconcat_mse_reg_lr1e5_ep35"
)
CONFIGS=(
  "${CONFIG_DIR}/${RUN_NAMES[0]}.yaml"
  "${CONFIG_DIR}/${RUN_NAMES[1]}.yaml"
  "${CONFIG_DIR}/${RUN_NAMES[2]}.yaml"
)

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}"

generate_configs() {
  python3 - <<'PY'
from pathlib import Path

import yaml

config_dir = Path("configs/rebuild/cross_pattern_1500_500_mse_regression")
output_root = Path("outputs/rebuild_reproduction")

splits = [
    ("train345_test1", "dataset/RealArchive/real_20rpm_increment_train345", "dataset/RealArchive/real_20rpm_increment_test1"),
    ("train134_test5", "dataset/RealArchive/real_20rpm_increment_train134", "dataset/RealArchive/real_20rpm_increment_test5"),
    ("train135_test4", "dataset/RealArchive/real_20rpm_increment_train135", "dataset/RealArchive/real_20rpm_increment_test4"),
]

for split_name, train_root, test_root in splits:
    run_name = f"crosspat_{split_name}_no_rpm_lateconcat_mse_reg_lr1e5_ep35"
    cfg = {
        "project": "allnewviscnet",
        "entity": "jongwonsohn-seoul-national-university",
        "name": run_name,
        "version": "v0",
        "train_settings": {
            "num_workers": 0,
            "seed": 1205,
            "classification": False,
            "gmm_bool": False,
            "watch_bool": False,
            "train_bool": True,
            "test_bool": True,
            "attn_bool": False,
            "val_test_bool": True,
            "sanity_check_bool": False,
        },
        "dataset": {
            "train": {
                "train_root": train_root,
                "use_all_samples": True,
                "frame_num": 10,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": "VideoDatasetReal",
                    "batch_size": 8,
                    "aug_bool": False,
                    "test_size": 1.0e-6,
                    "random_state": 37,
                },
            },
            "test": {
                "test_root": test_root,
                "frame_num": 10,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": "VideoDatasetReal",
                    "batch_size": 8,
                    "aug_bool": False,
                    "test_size": 1.0e-6,
                    "random_state": 37,
                },
            },
            "preprocess": {"scaler": "interscaler", "descaler": "interdescaler"},
        },
        "model": {
            "transformer_bool": True,
            "transformer": {"encoder": "VivitEmbed", "class": 10, "num_frames": 50, "image_size": 224},
            "embeddings": {
                "rpm_bool": False,
                "pat_bool": True,
                "pat_mode": "late_concat",
                "pattern_gate_init": 1.0,
            },
            "cnn": {
                "encoder": "Resnet34LSTMEMBED",
                "cnn_train": True,
                "cnn": "resnet34",
                "lstm_size": 256,
                "lstm_layers": 5,
                "output_size": 3,
                "drop_rate": 0.0,
                "embedding_size": 512,
                "embed_weight": 0.0,
            },
            "gmm": {"gmm_num": 3},
        },
        "training": {
            "curr_bool": False,
            "curr_ckpt": f"{run_name}.pth",
            "checkpoint_name": f"{run_name}.pth",
            "num_epochs": 35,
            "real_test_monitor": {
                "enabled": True,
                "dataset": "test",
                "interval_epochs": 1,
                "log_confusion_matrix": False,
            },
            "checkpoint_selection": {"metric": "val_loss"},
            "log_regression_mape": False,
            "loss": "MSE",
            "label_smoothing": 0.0,
            "optimizer": {
                "optim_class": "AdamW",
                "scheduler_class": "CosineAnnealingLR",
                "schedule_policy": "warmup_hold_cosine",
                "warmup_epochs": 1,
                "warmup_start_factor": 0.5,
                "lr_hold_epochs": 6,
                "lr": 1.0e-5,
                "eta_min": 1.0e-10,
                "weight_decay": 0.01,
                "patience": 10,
            },
            "acceptance": {
                "split": split_name,
                "train_root": train_root,
                "test_root": test_root,
                "note": (
                    "Simple no-RPM cross-pattern regression. Fixed 50 frames, late_concat "
                    "pattern conditioning with pattern_gate_init=1.0, no training augmentation, "
                    "MSE on normalized kinematic viscosity, val_loss checkpointing."
                ),
            },
        },
        "misc_dir": {
            "ckpt_root": "outputs/rebuild_reproduction/checkpoints",
            "output_root": str(output_root / run_name),
            "video_subdir": "videos",
            "para_subdir": "parameters",
            "norm_subdir": "parametersNorm",
        },
    }
    target = config_dir / f"{run_name}.yaml"
    target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(target)
PY
}

write_summary() {
  local status="$1"
  local exit_status="${2:-0}"
  python3 - "${SUMMARY_PATH}" "${status}" "${exit_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, exit_status = sys.argv[1:4]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "status": status,
        "exit_status": int(exit_status),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "wandb_project": "allnewviscnet",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
        "checkpoint_selection": "val_loss",
        "method": (
            "Cross-pattern no-RPM fixed-50 late_concat simple regression with MSE "
            "on normalized kinematic viscosity."
        ),
        "runs": payload.get("runs", []),
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

record_run() {
  local run_name="$1"
  local config="$2"
  local status="$3"
  local exit_status="${4:-0}"
  python3 - "${SUMMARY_PATH}" "${run_name}" "${config}" "${status}" "${exit_status}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, run_name, config, status, exit_status = sys.argv[1:6]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
output_root = Path("outputs/rebuild_reproduction") / run_name
error_plot = output_root / "error_plots" / f"{run_name}.png"
error_csv = output_root / "error_plots" / f"{run_name}_sorted_target.csv"
entry = {
    "run_name": run_name,
    "config": config,
    "status": status,
    "exit_status": int(exit_status),
    "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
    "train_log": f"outputs/rebuild_reproduction/logs/{run_name}.log",
    "output_root": str(output_root),
}
if error_plot.exists():
    entry["final_test_error_plot"] = str(error_plot)
if error_csv.exists():
    entry["final_test_error_csv"] = str(error_csv)
runs = [run for run in payload.get("runs", []) if run.get("run_name") != run_name]
runs.append(entry)
payload["runs"] = runs
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

fail_with_summary() {
  local status="$1"
  local exit_status="${2:-1}"
  write_summary "${status}" "${exit_status}"
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
  exit "${exit_status}"
}

if [ "${1:-}" = "--generate-only" ]; then
  generate_configs
  exit 0
fi

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required for W&B-backed MSE regression training." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="allnewviscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

rm -f "${MARKER_PATH}"
generate_configs

for root in \
  dataset/RealArchive/real_20rpm_increment_train345 \
  dataset/RealArchive/real_20rpm_increment_test1 \
  dataset/RealArchive/real_20rpm_increment_train134 \
  dataset/RealArchive/real_20rpm_increment_test5 \
  dataset/RealArchive/real_20rpm_increment_train135 \
  dataset/RealArchive/real_20rpm_increment_test4
do
  if [ ! -d "${root}/videos" ] || [ ! -d "${root}/parametersNorm" ] || [ ! -d "${root}/backgrounds" ]; then
    echo "Missing cross-pattern dataset layout under ${root}" >&2
    fail_with_summary "blocked_missing_dataset" 1
  fi
done

python3 scripts/verify_no_rpm_policy.py "${CONFIGS[@]}"

for run_name in "${RUN_NAMES[@]}"; do
  checkpoint="outputs/rebuild_reproduction/checkpoints/${run_name}.pth"
  if [ -f "${checkpoint}" ]; then
    echo "Refusing to overwrite existing checkpoint: ${checkpoint}" >&2
    fail_with_summary "blocked_existing_checkpoint" 1
  fi
done

write_summary "running" 0

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  log_path="${LOG_DIR}/${run_name}.log"
  port=$((BASE_PORT + idx))
  record_run "${run_name}" "${config}" "running" 0

  set +e
  PYTHONPATH=src torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --master_port="${port}" \
    --node_rank=0 \
    src/main.py \
    -c "${config}" \
    > "${log_path}" 2>&1
  train_status=$?
  set -e

  if [ "${train_status}" -ne 0 ]; then
    record_run "${run_name}" "${config}" "failed" "${train_status}"
    fail_with_summary "failed_${run_name}" "${train_status}"
  fi

  if [ ! -f "outputs/rebuild_reproduction/checkpoints/${run_name}.pth" ]; then
    record_run "${run_name}" "${config}" "failed_missing_checkpoint" 1
    fail_with_summary "failed_missing_checkpoint_after_${run_name}" 1
  fi

  record_run "${run_name}" "${config}" "completed" 0
done

write_summary "completed" 0
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
