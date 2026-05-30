#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CLASS_CONFIG="configs/rebuild/cross_pattern_pattern_norm/crosspat_train345_test1_no_rpm_patternnorm_xattn_cls_ep35.yaml"
REG_CONFIG="configs/rebuild/cross_pattern_pattern_norm/crosspat_train345_test1_no_rpm_patternnorm_xattn_mse_ft_from_cls_lr5e6_ep25.yaml"
CLASS_RUN="crosspat_train345_test1_no_rpm_patternnorm_xattn_cls_ep35"
REG_RUN="crosspat_train345_test1_no_rpm_patternnorm_xattn_mse_ft_from_cls_lr5e6_ep25"
CLASS_CKPT="outputs/rebuild_reproduction/checkpoints/${CLASS_RUN}.pth"
REG_CKPT="outputs/rebuild_reproduction/checkpoints/${REG_RUN}.pth"
LOG_DIR="outputs/rebuild_reproduction/logs"
WORKFLOW_DIR="outputs/rebuild_reproduction/patternnorm_classification_then_regression"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${WORKFLOW_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/patternnorm_classification_then_regression.done"

mkdir -p "${LOG_DIR}" "${WORKFLOW_DIR}" "${MARKER_DIR}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required for this W&B-backed run." >&2
  exit 2
fi

export WANDB_PROJECT="allnewviscnet"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

write_summary() {
  local status="$1"
  local stage="${2:-}"
  local exit_status="${3:-0}"
  python3 - "${SUMMARY_PATH}" "${status}" "${stage}" "${exit_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, stage, exit_status = sys.argv[1:5]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "status": status,
        "stage": stage,
        "exit_status": int(exit_status),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "wandb_project": "allnewviscnet",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
        "classification_config": "configs/rebuild/cross_pattern_pattern_norm/crosspat_train345_test1_no_rpm_patternnorm_xattn_cls_ep35.yaml",
        "classification_checkpoint": "outputs/rebuild_reproduction/checkpoints/crosspat_train345_test1_no_rpm_patternnorm_xattn_cls_ep35.pth",
        "regression_config": "configs/rebuild/cross_pattern_pattern_norm/crosspat_train345_test1_no_rpm_patternnorm_xattn_mse_ft_from_cls_lr5e6_ep25.yaml",
        "regression_checkpoint": "outputs/rebuild_reproduction/checkpoints/crosspat_train345_test1_no_rpm_patternnorm_xattn_mse_ft_from_cls_lr5e6_ep25.pth",
        "regression_mae_artifact": "outputs/rebuild_reproduction/crosspat_train345_test1_no_rpm_patternnorm_xattn_mse_ft_from_cls_lr5e6_ep25/error_plots/crosspat_train345_test1_no_rpm_patternnorm_xattn_mse_ft_from_cls_lr5e6_ep25.png",
        "runs": payload.get("runs", []),
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

record_run() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local status="$4"
  local exit_status="${5:-0}"
  python3 - "${SUMMARY_PATH}" "${run_name}" "${config}" "${stage}" "${status}" "${exit_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, stage, status, exit_status = sys.argv[1:7]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
entry = {
    "run_name": run_name,
    "config": config,
    "stage": stage,
    "status": status,
    "exit_status": int(exit_status),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
    "log": f"outputs/rebuild_reproduction/logs/{run_name}.log",
}
payload.setdefault("runs", [])
payload["runs"] = [item for item in payload["runs"] if item.get("run_name") != run_name]
payload["runs"].append(entry)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_stage() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local log_path="${LOG_DIR}/${run_name}.log"
  write_summary "running" "${stage}" 0
  record_run "${run_name}" "${config}" "${stage}" "running" 0
  echo "[$(date -Iseconds)] starting ${stage}: ${run_name}" | tee "${log_path}"
  if torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" src/main.py -c "${config}" >>"${log_path}" 2>&1; then
    record_run "${run_name}" "${config}" "${stage}" "completed" 0
  else
    local exit_status=$?
    record_run "${run_name}" "${config}" "${stage}" "failed" "${exit_status}"
    write_summary "failed" "${stage}" "${exit_status}"
    exit "${exit_status}"
  fi
}

rm -f "${MARKER_PATH}"
write_summary "starting" "preflight" 0
python3 scripts/verify_no_rpm_policy.py "${CLASS_CONFIG}" "${REG_CONFIG}"

if [[ -f "${CLASS_CKPT}" ]]; then
  record_run "${CLASS_RUN}" "${CLASS_CONFIG}" "classification" "skipped_existing_checkpoint" 0
else
  run_stage "${CLASS_RUN}" "${CLASS_CONFIG}" "classification"
fi

if [[ ! -f "${CLASS_CKPT}" ]]; then
  write_summary "failed" "classification_checkpoint_missing" 3
  echo "Missing classification checkpoint after classification stage: ${CLASS_CKPT}" >&2
  exit 3
fi

run_stage "${REG_RUN}" "${REG_CONFIG}" "regression_finetune"

if [[ ! -f "${REG_CKPT}" ]]; then
  write_summary "failed" "regression_checkpoint_missing" 4
  echo "Missing regression checkpoint after regression stage: ${REG_CKPT}" >&2
  exit 4
fi

write_summary "completed" "done" 0
touch "${MARKER_PATH}"
echo "completed: ${SUMMARY_PATH}"
