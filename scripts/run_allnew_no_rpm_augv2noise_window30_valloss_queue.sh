#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required for allnewviscnet W&B-backed training provenance." >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="allnewviscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

RUN_NAME="allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_valloss_ep70"
EVAL_NAME="allnew_synth_no_rpm_augv2noise_window30_valloss_realtest_frozen_eval"
TRAIN_CONFIG="configs/rebuild/retries/${RUN_NAME}.yaml"
EVAL_CONFIG="configs/rebuild/retries/${EVAL_NAME}.yaml"
LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/allnew_no_rpm_augv2noise_window30_valloss_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/allnew_no_rpm_augv2noise_window30_valloss_queue.done"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
EVAL_LOG="${LOG_DIR}/${EVAL_NAME}.log"
CHECKPOINT_PATH="outputs/rebuild_reproduction/checkpoints/${RUN_NAME}.pth"
PORT_BASE="${ALLNEW_WINDOW30_NOISE_VALLOSS_PORT_BASE:-30970}"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  local active_stage="${2:-}"
  local exit_status="${3:-0}"
  python3 - "${SUMMARY_PATH}" "${status}" "${active_stage}" "${exit_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, active_stage, exit_status = sys.argv[1:5]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
payload.update(
    {
        "status": status,
        "active_stage": active_stage,
        "exit_status": int(exit_status),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "wandb_project": "allnewviscnet",
        "wandb_metric_policy": ["train_loss", "val_loss", "test_loss"],
        "checkpoint_selection": "val_loss",
        "run_name": "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_valloss_ep70",
        "config": (
            "configs/rebuild/retries/"
            "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_valloss_ep70.yaml"
        ),
        "checkpoint": (
            "outputs/rebuild_reproduction/checkpoints/"
            "allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_valloss_ep70.pth"
        ),
        "method": (
            "Synthetic no-RPM sph35000 pretrain from scratch with 30-frame random temporal windows, "
            "augv2_noise video-consistent augmentation, per-epoch real-test diagnostic loss/confusion, "
            "and val_loss checkpointing."
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
entry = {
    "run_name": run_name,
    "config": config,
    "stage": stage,
    "status": status,
    "exit_status": int(exit_status),
    "log": log_path,
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
if stage == "synthetic_pretrain":
    entry["checkpoint"] = f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth"
else:
    metrics = Path("outputs/rebuild_reproduction") / run_name / "confusion_matrix" / f"{run_name}_metrics.json"
    if metrics.exists():
        entry["metrics"] = str(metrics)
        entry["confusion_matrix"] = str(metrics.parent / f"{run_name}.png")
runs = [run for run in payload.get("runs", []) if not (run.get("run_name") == run_name and run.get("stage") == stage)]
runs.append(entry)
payload["runs"] = runs
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

finish_with_status() {
  local status="$1"
  local active_stage="${2:-}"
  local exit_status="${3:-0}"
  write_summary "${status}" "${active_stage}" "${exit_status}"
  cp "${SUMMARY_PATH}" "${MARKER_PATH}"
  exit "${exit_status}"
}

run_torchrun() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local port="$4"
  local log_path="$5"
  write_summary "running" "${stage}" 0
  record_run "${run_name}" "${config}" "${stage}" "running" 0 "${log_path}"
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
    record_run "${run_name}" "${config}" "${stage}" "failed" "${status}" "${log_path}"
    finish_with_status "failed" "${stage}" "${status}"
  fi
  record_run "${run_name}" "${config}" "${stage}" "finished" "${status}" "${log_path}"
}

run_distribution_diagnostic() {
  local run_name="$1"
  local metrics="outputs/rebuild_reproduction/${run_name}/confusion_matrix/${run_name}_metrics.json"
  local output_dir="outputs/rebuild_reproduction/${run_name}/distribution"
  local output="${output_dir}/${run_name}_distribution.json"
  mkdir -p "${output_dir}"
  set +e
  python3 scripts/check_confusion_distribution.py --metrics "${metrics}" --output "${output}"
  local status=$?
  set -e
  python3 - "${SUMMARY_PATH}" "${run_name}" "${metrics}" "${output}" "${status}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, run_name, metrics, output, status = sys.argv[1:6]
path = Path(summary_path)
payload = json.loads(path.read_text())
payload["final_distribution_diagnostic"] = {
    "run_name": run_name,
    "metrics": metrics,
    "summary": output,
    "exit_status": int(status),
    "note": "Diagnostic only; not used for checkpoint selection or queue pass/fail.",
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

for root in \
  dataset/CFDArchive/sph_35000 \
  dataset/RealArchive/test_1000_wo_pat2
do
  if [ ! -d "${root}/videos" ] || [ ! -d "${root}/parametersNorm" ] || [ ! -d "${root}/backgrounds" ]; then
    echo "Missing dataset layout under ${root}" >&2
    finish_with_status "blocked_missing_dataset" "preflight" 1
  fi
done

if [ -f "${CHECKPOINT_PATH}" ]; then
  echo "Refusing to overwrite existing checkpoint: ${CHECKPOINT_PATH}" >&2
  finish_with_status "blocked_existing_checkpoint" "preflight" 1
fi

python3 scripts/verify_no_rpm_policy.py "${TRAIN_CONFIG}" "${EVAL_CONFIG}"

write_summary "starting" "preflight" 0
run_torchrun "${RUN_NAME}" "${TRAIN_CONFIG}" "synthetic_pretrain" "${PORT_BASE}" "${TRAIN_LOG}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Expected checkpoint missing after training: ${CHECKPOINT_PATH}" >&2
  finish_with_status "failed_missing_checkpoint" "synthetic_pretrain" 1
fi

run_torchrun "${EVAL_NAME}" "${EVAL_CONFIG}" "frozen_realtest_eval" "$((PORT_BASE + 1))" "${EVAL_LOG}"
run_distribution_diagnostic "${EVAL_NAME}"

finish_with_status "finished" "" 0
