#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateconcat_lr3e5_gate001_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateconcat_lr3e5_gate005_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateconcat_lr5e5_gate001_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateconcat_lr5e5_gate005_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateresidual_lr3e5_gate001_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateresidual_lr3e5_gate005_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateresidual_lr5e5_gate001_ep70.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateresidual_lr5e5_gate005_ep70.yaml"
)
RUN_NAMES=(
  "repro_realonly_993_50f_no_rpm_pattern_lateconcat_lr3e5_gate001_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateconcat_lr3e5_gate005_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateconcat_lr5e5_gate001_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateconcat_lr5e5_gate005_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateresidual_lr3e5_gate001_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateresidual_lr3e5_gate005_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateresidual_lr5e5_gate001_ep70"
  "repro_realonly_993_50f_no_rpm_pattern_lateresidual_lr5e5_gate005_ep70"
)

LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/realonly_50f_no_rpm_pattern_hparam_grid_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/realonly_50f_no_rpm_pattern_hparam_grid_queue.done"
BASE_PORT="${REALONLY_50F_PATTERN_GRID_BASE_PORT:-30601}"

mkdir -p "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}" "configs/rebuild/retries"

generate_configs() {
  python3 - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

config_dir = Path("configs/rebuild/retries")
output_root = Path("outputs/rebuild_reproduction")
bases = {
    "late_concat": yaml.safe_load(
        (config_dir / "realonly_993_50f_no_rpm_pattern_lateconcat_valloss_win30sched_ep50.yaml").read_text()
    ),
    "late_residual": yaml.safe_load(
        (config_dir / "realonly_993_50f_no_rpm_pattern_lateresidual_valloss_win30sched_ep50.yaml").read_text()
    ),
}
grid = [
    ("late_concat", "lr3e5_gate001", 3.0e-05, 0.01),
    ("late_concat", "lr3e5_gate005", 3.0e-05, 0.05),
    ("late_concat", "lr5e5_gate001", 5.0e-05, 0.01),
    ("late_concat", "lr5e5_gate005", 5.0e-05, 0.05),
    ("late_residual", "lr3e5_gate001", 3.0e-05, 0.01),
    ("late_residual", "lr3e5_gate005", 3.0e-05, 0.05),
    ("late_residual", "lr5e5_gate001", 5.0e-05, 0.01),
    ("late_residual", "lr5e5_gate005", 5.0e-05, 0.05),
]

for mode, tag, lr, gate in grid:
    suffix = "lateconcat" if mode == "late_concat" else "lateresidual"
    run_name = f"repro_realonly_993_50f_no_rpm_pattern_{suffix}_{tag}_ep70"
    cfg = deepcopy(bases[mode])
    cfg["project"] = "allnewviscnet"
    cfg["name"] = run_name
    cfg["model"]["embeddings"]["rpm_bool"] = False
    cfg["model"]["embeddings"]["pat_bool"] = True
    cfg["model"]["embeddings"]["pat_mode"] = mode
    cfg["model"]["embeddings"]["pattern_gate_init"] = gate
    cfg["training"]["curr_bool"] = False
    cfg["training"]["curr_ckpt"] = f"{run_name}.pth"
    cfg["training"]["checkpoint_name"] = f"{run_name}.pth"
    cfg["training"]["num_epochs"] = 70
    cfg["training"]["checkpoint_selection"] = {"metric": "val_loss"}
    cfg["training"]["optimizer"].update(
        {
            "optim_class": "AdamW",
            "scheduler_class": "CosineAnnealingLR",
            "schedule_policy": "warmup_hold_cosine",
            "warmup_epochs": 2,
            "warmup_start_factor": 0.25,
            "lr_hold_epochs": 15,
            "lr": lr,
            "eta_min": 3.0e-07,
            "weight_decay": 0.01,
            "patience": 18,
        }
    )
    cfg["training"].setdefault("real_test_monitor", {})
    cfg["training"]["real_test_monitor"].update(
        {"enabled": True, "dataset": "test", "interval_epochs": 1, "log_confusion_matrix": True}
    )
    cfg["training"]["acceptance"] = {
        "baseline_run": "repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70",
        "ablation": f"no_rpm_{mode}_pattern_fixed50_hparam_grid",
        "target_accuracy": 0.80,
        "allow_epoch_override": True,
        "note": (
            "Hyperparameter-only grid run for real-only fixed-50-frame no-RPM pattern conditioning. "
            "No data, architecture, or model-code changes; checkpointing and early stopping use val_loss only."
        ),
    }
    cfg["misc_dir"]["output_root"] = str(output_root / run_name)
    target = config_dir / f"realonly_993_50f_no_rpm_pattern_{suffix}_{tag}_ep70.yaml"
    target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(target)
PY
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
  echo "WANDB_API_KEY is required for W&B-backed real-only pattern hyperparameter grid training." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="allnewviscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

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
        "acceptance_gate": {"metric": "real_test_accuracy", "threshold": 0.80},
        "grid": {
            "pat_mode": ["late_concat", "late_residual"],
            "lr": [3.0e-05, 5.0e-05],
            "pattern_gate_init": [0.01, 0.05],
            "weight_decay": [0.01],
            "num_epochs": 70,
            "warmup_epochs": 2,
            "warmup_start_factor": 0.25,
            "lr_hold_epochs": 15,
            "eta_min": 3.0e-07,
            "patience": 18,
        },
        "method": (
            "Analyzer-driven hyperparameter-only grid for real-only fixed-50-frame "
            "no-RPM pattern conditioning. No data, architecture, or model-code changes."
        ),
    }
)
payload.setdefault("runs", [])
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
checkpoint = f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth"
output_root = Path("outputs/rebuild_reproduction") / run_name
metrics_path = output_root / "confusion_matrix" / f"{run_name}_metrics.json"
entry = {
    "run_name": run_name,
    "config": config,
    "status": status,
    "exit_status": int(exit_status),
    "checkpoint": checkpoint,
    "train_log": f"outputs/rebuild_reproduction/logs/{run_name}.log",
    "output_root": str(output_root),
}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    entry["final_test_metrics"] = {
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(metrics_path.parent / f"{run_name}.png"),
        "accuracy": metrics.get("accuracy"),
    }
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

rm -f "${MARKER_PATH}"
generate_configs

for path in \
  dataset/RealArchive/train_993_wo_pat2/backgrounds/1.png \
  dataset/RealArchive/train_993_wo_pat2/backgrounds/2.png \
  dataset/RealArchive/train_993_wo_pat2/backgrounds/3.png \
  dataset/RealArchive/train_993_wo_pat2/backgrounds/4.png \
  dataset/RealArchive/test_1000_wo_pat2/backgrounds/1.png \
  dataset/RealArchive/test_1000_wo_pat2/backgrounds/2.png \
  dataset/RealArchive/test_1000_wo_pat2/backgrounds/3.png \
  dataset/RealArchive/test_1000_wo_pat2/backgrounds/4.png
do
  if [ ! -f "${path}" ]; then
    echo "Missing pattern background: ${path}" >&2
    fail_with_summary "blocked_missing_background" 1
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
