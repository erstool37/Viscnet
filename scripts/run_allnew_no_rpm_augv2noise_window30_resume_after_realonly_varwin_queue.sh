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

RUN_NAME="allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_realonly_varwin_ep61"
CONFIG="configs/rebuild/retries/${RUN_NAME}.yaml"
LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/allnew_no_rpm_augv2noise_window30_resume_after_realonly_varwin_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/allnew_no_rpm_augv2noise_window30_resume_after_realonly_varwin_queue.done"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
SOURCE_CHECKPOINT="outputs/rebuild_reproduction/checkpoints/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_varwin_ep62.pth"
OUTPUT_CHECKPOINT="outputs/rebuild_reproduction/checkpoints/${RUN_NAME}.pth"
PORT="${ALLNEW_WINDOW30_RESUME_REALONLY_VARWIN_PORT:-30494}"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
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
payload = {
    "status": status,
    "exit_status": int(exit_status),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "wandb_project": "allnewViscnet",
    "run_name": "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_realonly_varwin_ep61",
    "config": "configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_realonly_varwin_ep61.yaml",
    "source_checkpoint": "outputs/rebuild_reproduction/checkpoints/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_varwin_ep62.pth",
    "output_checkpoint": "outputs/rebuild_reproduction/checkpoints/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_realonly_varwin_ep61.pth",
    "log": "outputs/rebuild_reproduction/logs/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_resume_after_realonly_varwin_ep61.log",
    "resume_mode": "checkpoint_restart_not_exact_process_resume",
    "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
}
Path(summary_path).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

if [ ! -f "${SOURCE_CHECKPOINT}" ]; then
  echo "Source checkpoint missing: ${SOURCE_CHECKPOINT}" >&2
  write_summary "blocked_missing_source_checkpoint" 1
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
  exit 1
fi

if [ -e "${OUTPUT_CHECKPOINT}" ]; then
  echo "Refusing to overwrite existing checkpoint: ${OUTPUT_CHECKPOINT}" >&2
  write_summary "blocked_existing_output_checkpoint" 1
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
  exit 1
fi

python3 scripts/verify_no_rpm_policy.py "${CONFIG}"

write_summary "running" 0
set +e
PYTHONPATH=src torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --master_port="${PORT}" \
  --node_rank=0 \
  src/main.py \
  -c "${CONFIG}" \
  > "${LOG_PATH}" 2>&1
status=$?
set -e

if [ "${status}" -eq 0 ]; then
  write_summary "finished" "${status}"
else
  write_summary "failed" "${status}"
fi
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
exit "${status}"
