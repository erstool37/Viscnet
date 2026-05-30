#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required for W&B-backed real-only late-pattern training." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="re-rebuild-viscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

CONFIGS=(
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateconcat_ep45.yaml"
  "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_lateresidual_ep45.yaml"
)
RUN_NAMES=(
  "repro_realonly_993_50f_no_rpm_pattern_lateconcat_ep45"
  "repro_realonly_993_50f_no_rpm_pattern_lateresidual_ep45"
)

LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/realonly_50f_no_rpm_late_pattern_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/realonly_50f_no_rpm_late_pattern_queue.done"
BASE_PORT="${REALONLY_50F_LATE_PATTERN_BASE_PORT:-30541}"

mkdir -p "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

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
        "wandb_project": "re-rebuild-viscnet",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
        "method": (
            "Two real-only fixed-50-frame no-RPM pattern-head runs: late_concat and "
            "late_residual. Both avoid early token pattern injection and avoid temporal window shifting."
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
