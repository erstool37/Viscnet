#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required for allnewViscnet W&B-backed training provenance." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="allnewViscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

RUN_NAME="allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70"
EVAL_NAME="allnew_synth_no_rpm_augv2noise_window30_realtest_frozen_eval"
TRAIN_CONFIG="configs/rebuild/retries/${RUN_NAME}.yaml"
EVAL_CONFIG="configs/rebuild/retries/${EVAL_NAME}.yaml"
LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/allnew_no_rpm_augv2noise_window30_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/allnew_no_rpm_augv2noise_window30_queue.done"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
EVAL_LOG="${LOG_DIR}/${EVAL_NAME}.log"
CHECKPOINT_PATH="outputs/rebuild_reproduction/checkpoints/${RUN_NAME}.pth"
PORT_BASE="${ALLNEW_WINDOW30_NOISE_PORT_BASE:-30470}"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  local active_stage="${2:-}"
  python3 - "${SUMMARY_PATH}" "${status}" "${active_stage}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, active_stage = sys.argv[1:4]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
payload["status"] = status
payload["active_stage"] = active_stage
payload["updated_at"] = datetime.now(timezone.utc).isoformat()
payload["wandb_project"] = "allnewViscnet"
payload["run_name"] = "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70"
payload["config"] = (
    "configs/rebuild/retries/"
    "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70.yaml"
)
payload["checkpoint"] = (
    "outputs/rebuild_reproduction/checkpoints/"
    "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70.pth"
)
payload["diagnostic_gate"] = {
    "min_used_classes": 8,
    "max_predicted_class_share": 0.35,
    "max_zero_predicted_classes": 2,
}
payload["selection_rule"] = {
    "primary": "real_test_distribution_score",
    "tie_breaker": "real_test_loss",
    "accuracy_role": "diagnostic only",
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

append_run() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local status="$4"
  local exit_status="$5"
  local log_path="$6"
  python3 - "${SUMMARY_PATH}" "${run_name}" "${config}" "${stage}" "${status}" "${exit_status}" "${log_path}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, stage, status, exit_status, log_path = sys.argv[1:8]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
payload.setdefault("runs", []).append(
    {
        "run_name": run_name,
        "config": config,
        "stage": stage,
        "status": status,
        "exit_status": int(exit_status),
        "log": log_path,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

mark_failed() {
  local active_stage="$1"
  write_summary "failed" "${active_stage}"
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
}

run_torchrun() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local port="$4"
  local log_path="$5"
  write_summary "running" "${stage}"
  set +e
  PYTHONPATH=src torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --master_port="${port}" \
    --node_rank=0 \
    src/main.py \
    -c "${config}" \
    > "${log_path}" 2>&1
  local status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    append_run "${run_name}" "${config}" "${stage}" "failed" "${status}" "${log_path}"
    mark_failed "${stage}"
    exit "${status}"
  fi
  append_run "${run_name}" "${config}" "${stage}" "finished" "${status}" "${log_path}"
}

run_distribution_check() {
  local run_name="$1"
  local metrics="outputs/rebuild_reproduction/${run_name}/confusion_matrix/${run_name}_metrics.json"
  local output_dir="outputs/rebuild_reproduction/${run_name}/distribution"
  local output="${output_dir}/${run_name}_distribution.json"
  mkdir -p "${output_dir}"
  set +e
  python3 scripts/check_confusion_distribution.py \
    --metrics "${metrics}" \
    --output "${output}"
  local status=$?
  set -e
  python3 - "${SUMMARY_PATH}" "${run_name}" "${metrics}" "${output}" "${status}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, run_name, metrics, output, status = sys.argv[1:6]
path = Path(summary_path)
payload = json.loads(path.read_text())
payload["final_distribution_check"] = {
    "run_name": run_name,
    "metrics": metrics,
    "summary": output,
    "exit_status": int(status),
    "note": "Exit status 1 means diagnostic gate was not met; the training/eval provenance is still preserved.",
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

if [ -f "${CHECKPOINT_PATH}" ]; then
  echo "Refusing to reuse existing checkpoint: ${CHECKPOINT_PATH}" >&2
  echo "Move or remove it before launching a from-scratch run." >&2
  write_summary "blocked_existing_checkpoint" "preflight"
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
  exit 1
fi

python3 scripts/verify_no_rpm_policy.py "${TRAIN_CONFIG}" "${EVAL_CONFIG}"

write_summary "starting" "preflight"
run_torchrun "${RUN_NAME}" "${TRAIN_CONFIG}" "synthetic_pretrain" "${PORT_BASE}" "${TRAIN_LOG}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Expected checkpoint missing after training: ${CHECKPOINT_PATH}" >&2
  write_summary "failed_missing_checkpoint" "synthetic_pretrain"
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
  exit 1
fi

run_torchrun "${EVAL_NAME}" "${EVAL_CONFIG}" "frozen_realtest_eval" "$((PORT_BASE + 1))" "${EVAL_LOG}"
run_distribution_check "${EVAL_NAME}"

write_summary "finished" ""
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
