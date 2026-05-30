#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="allnewViscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

DIAG_CONFIG="configs/rebuild/retries/realonly_993_window30x21_ep50_no_rpm_runtime_varwin_eval.yaml"
CHECKPOINT="outputs/rebuild_reproduction/checkpoints/repro_realonly_993_window30x21_ep50.pth"
DIAG_RUN_NAME="realonly_993_window30x21_ep50_no_rpm_runtime_varwin_eval"
DIAG_OUTPUT_ROOT="outputs/rebuild_reproduction/realonly_window30_no_rpm_runtime_variable_window_diagnostic"
DIAG_SUMMARY="${DIAG_OUTPUT_ROOT}/${DIAG_RUN_NAME}/summary.json"
LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/realonly_window30_no_rpm_runtime_varwin_diag_then_resume"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
WORKFLOW_SUMMARY="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/realonly_window30_no_rpm_runtime_varwin_diag_then_resume.done"
DIAG_LOG="${LOG_DIR}/${DIAG_RUN_NAME}.log"
DIAG_PORT="${REALONLY_WINDOW30_VARWIN_DIAG_PORT:-30493}"
RESUME_SESSION="allnew_no_rpm_augv2noise_window30_resume_after_realonly_varwin"
RESUME_SCRIPT="/root/Viscnet/scripts/run_allnew_no_rpm_augv2noise_window30_resume_after_realonly_varwin_queue.sh"

mkdir -p "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_workflow_summary() {
  local status="$1"
  local exit_status="${2:-0}"
  python3 - "${WORKFLOW_SUMMARY}" "${status}" "${exit_status}" "${DIAG_SUMMARY}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, exit_status, diag_summary = sys.argv[1:5]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "status": status,
        "exit_status": int(exit_status),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_summary": diag_summary,
        "diagnostic_log": "outputs/rebuild_reproduction/logs/realonly_993_window30x21_ep50_no_rpm_runtime_varwin_eval.log",
        "checkpoint": "outputs/rebuild_reproduction/checkpoints/repro_realonly_993_window30x21_ep50.pth",
        "config": "configs/rebuild/retries/realonly_993_window30x21_ep50_no_rpm_runtime_varwin_eval.yaml",
        "resume_session": null,
        "resume_script": null,
        "note": "Real-only 30-frame checkpoint evaluated with no-RPM runtime. Source checkpoint was RPM-trained; no real-only 30-frame no-RPM-trained checkpoint exists locally.",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

if pgrep -f "torchrun .*src/main.py -c configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag" >/dev/null; then
  echo "Training still appears active; pause it before running this diagnostic." >&2
  write_workflow_summary "blocked_training_still_active" 1
  cp "${WORKFLOW_SUMMARY}" "${MARKER_PATH}"
  exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint missing: ${CHECKPOINT}" >&2
  write_workflow_summary "blocked_missing_checkpoint" 1
  cp "${WORKFLOW_SUMMARY}" "${MARKER_PATH}"
  exit 1
fi

write_workflow_summary "running_realonly_variable_window_diagnostic" 0

set +e
PYTHONPATH=src torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --master_port="${DIAG_PORT}" \
  --node_rank=0 \
  scripts/run_variable_window_realtest_diagnostic.py \
  --config "${DIAG_CONFIG}" \
  --checkpoint "$(basename "${CHECKPOINT}")" \
  --output-root "${DIAG_OUTPUT_ROOT}" \
  --run-name "${DIAG_RUN_NAME}" \
  --batch-size 16 \
  --window-size 30 \
  --source-frame-count 50 \
  > "${DIAG_LOG}" 2>&1
status=$?
set -e

if [ "${status}" -ne 0 ]; then
  write_workflow_summary "failed_realonly_variable_window_diagnostic" "${status}"
  cp "${WORKFLOW_SUMMARY}" "${MARKER_PATH}"
  exit "${status}"
fi

python3 - "${WORKFLOW_SUMMARY}" "${DIAG_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

workflow_path, diag_path = map(Path, sys.argv[1:3])
workflow = json.loads(workflow_path.read_text())
diag = json.loads(diag_path.read_text())
workflow["diagnostic_result"] = {
    "config": diag["config"],
    "checkpoint": diag["checkpoint"],
    "runtime_zeroed_rpm_confirmed": diag["runtime_zeroed_rpm_confirmed"],
    "configured_num_frames": diag["configured_num_frames"],
    "window_starts": diag["window_starts"],
    "fixed_start0_window": diag["fixed_start0_window"],
    "variable_window_all_starts_per_window": diag["variable_window_all_starts_per_window"],
    "variable_window_mean_logits_per_video": diag["variable_window_mean_logits_per_video"],
    "distribution_delta_vs_fixed_first30": diag.get("distribution_delta_vs_fixed_first30"),
}
workflow_path.write_text(json.dumps(workflow, indent=2) + "\n")
PY

write_workflow_summary "diagnostic_finished_no_training_resumed" 0

cp "${WORKFLOW_SUMMARY}" "${MARKER_PATH}"
