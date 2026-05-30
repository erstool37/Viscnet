#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required for W&B-backed real-only pattern training." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="re-rebuild-viscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

CONFIG="configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_embed_ep45.yaml"
RUN_NAME="repro_realonly_993_50f_no_rpm_pattern_embed_ep45"
CHECKPOINT="outputs/rebuild_reproduction/checkpoints/${RUN_NAME}.pth"
OUTPUT_ROOT="outputs/rebuild_reproduction/${RUN_NAME}"
LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/${RUN_NAME}_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/${RUN_NAME}.done"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
TRAIN_PORT="${REALONLY_50F_PATTERN_TRAIN_PORT:-30531}"

mkdir -p "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  local exit_status="${2:-0}"
  python3 - "${SUMMARY_PATH}" "${status}" "${exit_status}" "${CONFIG}" "${CHECKPOINT}" "${TRAIN_LOG}" "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, exit_status, config, checkpoint, train_log, output_root = sys.argv[1:8]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "status": status,
        "exit_status": int(exit_status),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "checkpoint": checkpoint,
        "train_log": train_log,
        "output_root": output_root,
        "wandb_project": "re-rebuild-viscnet",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
        "method": "real-only fixed-50-frame training; no RPM; pattern image conditioning with pat_mode=embedding; no temporal window shifting.",
    }
)
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

if [ -f "${CHECKPOINT}" ]; then
  echo "Refusing to overwrite existing checkpoint: ${CHECKPOINT}" >&2
  fail_with_summary "blocked_existing_checkpoint" 1
fi

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

python3 scripts/verify_no_rpm_policy.py "${CONFIG}"

write_summary "running_realonly_50f_pattern_training" 0

set +e
PYTHONPATH=src torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --master_port="${TRAIN_PORT}" \
  --node_rank=0 \
  src/main.py \
  -c "${CONFIG}" \
  > "${TRAIN_LOG}" 2>&1
train_status=$?
set -e

if [ "${train_status}" -ne 0 ]; then
  fail_with_summary "failed_realonly_50f_pattern_training" "${train_status}"
fi

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Training completed but checkpoint is missing: ${CHECKPOINT}" >&2
  fail_with_summary "failed_missing_checkpoint_after_training" 1
fi

python3 - "${SUMMARY_PATH}" "${OUTPUT_ROOT}" "${RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
run_name = sys.argv[3]
payload = json.loads(summary_path.read_text())
metrics_path = output_root / "confusion_matrix" / f"{run_name}_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    payload["final_test_metrics"] = {
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(metrics_path.parent / f"{run_name}.png"),
        "accuracy": metrics.get("accuracy"),
    }
summary_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

write_summary "completed" 0
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
