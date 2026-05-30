#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required for W&B-backed real-only training." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="re-rebuild-viscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

CONFIG="configs/rebuild/retries/realonly_993_window30x21_no_rpm_ep50.yaml"
RUN_NAME="repro_realonly_993_window30x21_no_rpm_ep50"
CHECKPOINT="outputs/rebuild_reproduction/checkpoints/${RUN_NAME}.pth"
TRAIN_OUTPUT_ROOT="outputs/rebuild_reproduction/${RUN_NAME}"
TRAIN_FIXED_METRICS="${TRAIN_OUTPUT_ROOT}/confusion_matrix/${RUN_NAME}_metrics.json"
VARWIN_ROOT="outputs/rebuild_reproduction/${RUN_NAME}_variable_window_diagnostic"
VARWIN_RUN_NAME="${RUN_NAME}_varwin_realtest"
VARWIN_SUMMARY="${VARWIN_ROOT}/${VARWIN_RUN_NAME}/summary.json"
LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/realonly_window30x21_no_rpm_train_then_varwin"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/realonly_window30x21_no_rpm_train_then_varwin.done"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
VARWIN_LOG="${LOG_DIR}/${VARWIN_RUN_NAME}.log"
TRAIN_PORT="${REALONLY_WINDOW30_NO_RPM_TRAIN_PORT:-30511}"
VARWIN_PORT="${REALONLY_WINDOW30_NO_RPM_VARWIN_PORT:-30512}"

mkdir -p "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  local exit_status="${2:-0}"
  python3 - "${SUMMARY_PATH}" "${status}" "${exit_status}" "${CONFIG}" "${CHECKPOINT}" "${TRAIN_LOG}" "${VARWIN_SUMMARY}" "${VARWIN_LOG}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, exit_status, config, checkpoint, train_log, varwin_summary, varwin_log = sys.argv[1:9]
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
        "variable_window_summary": varwin_summary,
        "variable_window_log": varwin_log,
        "wandb_project": "re-rebuild-viscnet",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
        "method": "real-only no-RPM training on 21 contiguous 30-frame windows per training video; no synthetic training or synthetic checkpoint resume.",
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

python3 scripts/build_real_window_dataset.py \
  --source-manifest configs/rebuild/manifests/real_train_993.json \
  --output-root outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1 \
  --window-size 30 \
  --windows-per-video 21

if [ ! -e "outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1/backgrounds" ]; then
  echo "Derived real-window dataset backgrounds are missing." >&2
  fail_with_summary "blocked_missing_derived_backgrounds" 1
fi

python3 scripts/verify_no_rpm_policy.py "${CONFIG}"

write_summary "running_realonly_training" 0

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
  fail_with_summary "failed_realonly_training" "${train_status}"
fi

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Training completed but checkpoint is missing: ${CHECKPOINT}" >&2
  fail_with_summary "failed_missing_checkpoint_after_training" 1
fi

write_summary "running_variable_window_realtest_diagnostic" 0

set +e
PYTHONPATH=src torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --master_port="${VARWIN_PORT}" \
  --node_rank=0 \
  scripts/run_variable_window_realtest_diagnostic.py \
  --config "${CONFIG}" \
  --checkpoint "$(basename "${CHECKPOINT}")" \
  --output-root "${VARWIN_ROOT}" \
  --run-name "${VARWIN_RUN_NAME}" \
  --batch-size 16 \
  --window-size 30 \
  --source-frame-count 50 \
  --fixed-first30-metrics "${TRAIN_FIXED_METRICS}" \
  > "${VARWIN_LOG}" 2>&1
varwin_status=$?
set -e

if [ "${varwin_status}" -ne 0 ]; then
  fail_with_summary "failed_variable_window_realtest_diagnostic" "${varwin_status}"
fi

python3 - "${SUMMARY_PATH}" "${VARWIN_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, varwin_summary_path = map(Path, sys.argv[1:3])
summary = json.loads(summary_path.read_text())
varwin = json.loads(varwin_summary_path.read_text())
summary["variable_window_result"] = {
    "runtime_zeroed_rpm_confirmed": varwin["runtime_zeroed_rpm_confirmed"],
    "configured_num_frames": varwin["configured_num_frames"],
    "window_starts": varwin["window_starts"],
    "fixed_first30_metrics": varwin.get("fixed_first30_metrics"),
    "fixed_start0_window": varwin["fixed_start0_window"],
    "variable_window_all_starts_per_window": varwin["variable_window_all_starts_per_window"],
    "variable_window_mean_logits_per_video": varwin["variable_window_mean_logits_per_video"],
    "distribution_delta_vs_fixed_first30": varwin.get("distribution_delta_vs_fixed_first30"),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY

write_summary "completed" 0
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
