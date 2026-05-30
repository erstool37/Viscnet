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

LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/allnew_no_rpm_aug_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/allnew_no_rpm_aug_queue.done"
PORT_BASE="${ALLNEW_NO_RPM_AUG_PORT_BASE:-30310}"

AUGV1_SYNTH_CONFIG="configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80.yaml"
AUGV2_SYNTH_CONFIG="configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv2_ep80.yaml"
AUGV1_EVAL_CONFIG="configs/rebuild/retries/allnew_synth_no_rpm_augv1_realtest_frozen_eval.yaml"
AUGV2_EVAL_CONFIG="configs/rebuild/retries/allnew_synth_no_rpm_augv2_realtest_frozen_eval.yaml"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  local active_run="${2:-}"
  python3 - "${SUMMARY_PATH}" "${status}" "${active_run}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, active_run = sys.argv[1:4]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
payload["status"] = status
payload["active_run"] = active_run
payload["updated_at"] = datetime.now(timezone.utc).isoformat()
payload["wandb_project"] = "allnewViscnet"
payload["policy"] = "allnew no-RPM synthetic augmentation diagnostics"
payload["selection_rule"] = {
    "well_distributed": {
        "min_used_classes": 8,
        "max_predicted_class_share": 0.35,
        "max_zero_predicted_classes": 2,
    },
    "tie_breaker": "higher frozen real-test accuracy when class-distribution diagnostics are similar",
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
  python3 - "${SUMMARY_PATH}" "${run_name}" "${config}" "${stage}" "${status}" "${exit_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, stage, status, exit_status = sys.argv[1:7]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
payload.setdefault("runs", []).append(
    {
        "run_name": run_name,
        "config": config,
        "stage": stage,
        "status": status,
        "exit_status": int(exit_status),
        "log": f"outputs/rebuild_reproduction/logs/{run_name}.log",
        "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
        "metrics": f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json",
        "distribution": f"outputs/rebuild_reproduction/{run_name}/distribution/{run_name}_distribution.json",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

skip_if_checkpoint_exists() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local checkpoint="outputs/rebuild_reproduction/checkpoints/${run_name}.pth"
  if [ ! -f "${checkpoint}" ]; then
    return 1
  fi
  echo "Skipping ${run_name}; checkpoint already exists at ${checkpoint}."
  append_run "${run_name}" "${config}" "${stage}" "skipped_existing_checkpoint" 0
  return 0
}

run_torchrun() {
  local run_name="$1"
  local config="$2"
  local stage="$3"
  local port="$4"
  local log_path="${LOG_DIR}/${run_name}.log"
  write_summary "running" "${run_name}"
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
  if [ "${status}" -eq 0 ]; then
    append_run "${run_name}" "${config}" "${stage}" "finished" "${status}"
  else
    append_run "${run_name}" "${config}" "${stage}" "failed" "${status}"
    write_summary "failed" "${run_name}"
    cp "${SUMMARY_PATH}" "${MARKER_PATH}"
    exit "${status}"
  fi
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
  python3 - "${SUMMARY_PATH}" "${run_name}" "${output}" "${status}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, run_name, output, status = sys.argv[1:5]
path = Path(summary_path)
payload = json.loads(path.read_text())
payload.setdefault("distribution_checks", {})[run_name] = {
    "summary": output,
    "exit_status": int(status),
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

select_transfer_candidate() {
  python3 - "${SUMMARY_PATH}" <<'PY'
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
runs = [
    "allnew_synth_no_rpm_augv1_realtest_frozen_eval",
    "allnew_synth_no_rpm_augv2_realtest_frozen_eval",
]
summaries = {}
for run in runs:
    path = Path(f"outputs/rebuild_reproduction/{run}/distribution/{run}_distribution.json")
    if not path.exists():
        raise SystemExit(f"missing frozen eval distribution summary: {path}")
    summaries[run] = json.loads(path.read_text())

def score(item):
    run, summary = item
    return (
        1 if summary["well_distributed"] else 0,
        int(summary["predicted_classes_used"]),
        -float(summary["max_predicted_class_share"]),
        float(summary.get("accuracy") or 0.0),
    )

selected, selected_summary = max(summaries.items(), key=score)
payload["transfer_candidate"] = {
    "run_name": selected,
    "source_checkpoint": (
        "allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80.pth"
        if "augv1" in selected
        else "allnew_synthetic_pretrain_sph35000_no_rpm_augv2_ep80.pth"
    ),
    "reason": "selected by frozen real-test class-distribution diagnostic; accuracy is only a tie-breaker",
    "distribution": selected_summary,
}
summary_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

python3 scripts/verify_no_rpm_policy.py \
  "${AUGV1_SYNTH_CONFIG}" \
  "${AUGV2_SYNTH_CONFIG}" \
  "${AUGV1_EVAL_CONFIG}" \
  "${AUGV2_EVAL_CONFIG}"

write_summary "starting" ""

if ! skip_if_checkpoint_exists \
  "allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80" \
  "${AUGV1_SYNTH_CONFIG}" \
  "synthetic_pretrain"; then
  run_torchrun \
    "allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80" \
    "${AUGV1_SYNTH_CONFIG}" \
    "synthetic_pretrain" \
    "${PORT_BASE}"
fi

run_torchrun \
  "allnew_synthetic_pretrain_sph35000_no_rpm_augv2_ep80" \
  "${AUGV2_SYNTH_CONFIG}" \
  "synthetic_pretrain" \
  "$((PORT_BASE + 1))"

run_torchrun \
  "allnew_synth_no_rpm_augv1_realtest_frozen_eval" \
  "${AUGV1_EVAL_CONFIG}" \
  "frozen_realtest_eval" \
  "$((PORT_BASE + 2))"
run_distribution_check "allnew_synth_no_rpm_augv1_realtest_frozen_eval"

run_torchrun \
  "allnew_synth_no_rpm_augv2_realtest_frozen_eval" \
  "${AUGV2_EVAL_CONFIG}" \
  "frozen_realtest_eval" \
  "$((PORT_BASE + 3))"
run_distribution_check "allnew_synth_no_rpm_augv2_realtest_frozen_eval"

select_transfer_candidate
write_summary "finished" ""
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
