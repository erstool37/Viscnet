#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/rebuild/retries/realonly_993_window30x21_no_rpm_mse_ft_from_cls_lr5e6_ep35.yaml"
RUN_NAME="realonly_993_window30x21_no_rpm_mse_ft_from_cls_lr5e6_ep35"
OUTPUT_ROOT="outputs/rebuild_reproduction/${RUN_NAME}"
LOG_DIR="outputs/rebuild_reproduction/logs"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
MARKER_PATH="${MARKER_DIR}/${RUN_NAME}.done"
STATUS_PATH="${MARKER_DIR}/${RUN_NAME}.status.json"
ERROR_CSV="${OUTPUT_ROOT}/error_plots/${RUN_NAME}_sorted_target.csv"
METRICS_JSON="${OUTPUT_ROOT}/regression_error_summary.json"
METRICS_MD="${OUTPUT_ROOT}/regression_error_summary.md"

mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${OUTPUT_ROOT}"
rm -f "${MARKER_PATH}" "${STATUS_PATH}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export WANDB_PROJECT="allnewViscnet"
export WANDB_MODE="online"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set after loading .env; refusing to launch without W&B logging." >&2
  exit 1
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "started_at=$(date -Iseconds)" | tee "${LOG_PATH}"
echo "config=${CONFIG}" | tee -a "${LOG_PATH}"
echo "wandb_project=${WANDB_PROJECT}" | tee -a "${LOG_PATH}"
echo "wandb_mode=${WANDB_MODE}" | tee -a "${LOG_PATH}"
echo "cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG}" | tee -a "${LOG_PATH}"

set +e
torchrun --standalone --nproc_per_node=8 src/main.py -c "${CONFIG}" 2>&1 | tee -a "${LOG_PATH}"
train_status=${PIPESTATUS[0]}
set -e

summary_status=1
if [[ "${train_status}" -eq 0 && -f "${ERROR_CSV}" ]]; then
  python3 scripts/summarize_regression_errors.py \
    --csv "${ERROR_CSV}" \
    --output-json "${METRICS_JSON}" \
    --output-md "${METRICS_MD}" 2>&1 | tee -a "${LOG_PATH}"
  summary_status=${PIPESTATUS[0]}
fi

finished_at="$(date -Iseconds)"
python3 - <<PY
import json
from pathlib import Path
payload = {
    "run_name": "${RUN_NAME}",
    "config": "${CONFIG}",
    "log": "${LOG_PATH}",
    "output_root": "${OUTPUT_ROOT}",
    "checkpoint": "outputs/rebuild_reproduction/checkpoints/${RUN_NAME}.pth",
    "error_csv": "${ERROR_CSV}",
    "metrics_json": "${METRICS_JSON}",
    "metrics_md": "${METRICS_MD}",
    "wandb_project": "${WANDB_PROJECT}",
    "wandb_mode": "${WANDB_MODE}",
    "train_status": int("${train_status}"),
    "summary_status": int("${summary_status}"),
    "finished_at": "${finished_at}",
}
Path("${STATUS_PATH}").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
if payload["train_status"] == 0 and payload["summary_status"] == 0:
    Path("${MARKER_PATH}").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
PY

if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi
exit "${summary_status}"
