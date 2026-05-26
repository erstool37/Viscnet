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
export WANDB_PROJECT="re-rebuild-viscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/no_rpm_transfer_policy_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/no_rpm_transfer_policy_queue.done"
PORT_BASE="${NO_RPM_POLICY_PORT_BASE:-30210}"
NPROC="${NPROC_PER_NODE:-4}"
SYNTH_CONFIG="configs/rebuild/retries/synthetic_pretrain_sph35000_no_rpm_ep50.yaml"
TRANSFER_CONFIG="configs/rebuild/retries/transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70.yaml"
DUAL_CONFIG="configs/rebuild/retries/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70.yaml"

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
payload["policy"] = "no RPM embeddings for every queued run"
payload["realonly_comparator"] = {
    "run_name": "repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70",
    "metrics": "outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70/confusion_matrix/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70_metrics.json",
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

append_run() {
  local run_name="$1"
  local config="$2"
  local status="$3"
  local exit_status="$4"
  python3 - "${SUMMARY_PATH}" "${run_name}" "${config}" "${status}" "${exit_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, status, exit_status = sys.argv[1:6]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
payload.setdefault("runs", []).append(
    {
        "run_name": run_name,
        "config": config,
        "status": status,
        "exit_status": int(exit_status),
        "log": f"outputs/rebuild_reproduction/logs/{run_name}.log",
        "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
        "metrics": f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json",
        "reliability": f"outputs/rebuild_reproduction/{run_name}/reliability_plots/{run_name}_metrics.json",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_training() {
  local run_name="$1"
  local config="$2"
  local port="$3"
  local log_path="${LOG_DIR}/${run_name}.log"
  write_summary "running" "${run_name}"
  set +e
  PYTHONPATH=src torchrun \
    --nproc_per_node="${NPROC}" \
    --nnodes=1 \
    --master_port="${port}" \
    --node_rank=0 \
    src/main.py \
    -c "${config}" \
    > "${log_path}" 2>&1
  local status=$?
  set -e
  if [ "${status}" -eq 0 ]; then
    append_run "${run_name}" "${config}" "finished" "${status}"
  else
    append_run "${run_name}" "${config}" "failed" "${status}"
    write_summary "failed" "${run_name}"
    cp "${SUMMARY_PATH}" "${MARKER_PATH}"
    exit "${status}"
  fi
}

python3 scripts/verify_no_rpm_policy.py "${SYNTH_CONFIG}" "${TRANSFER_CONFIG}" "${DUAL_CONFIG}"
write_summary "starting" ""

run_training \
  "repro_synthetic_pretrain_sph35000_no_rpm_ep50" \
  "${SYNTH_CONFIG}" \
  "${PORT_BASE}"

run_training \
  "repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70" \
  "${TRANSFER_CONFIG}" \
  "$((PORT_BASE + 1))"

run_training \
  "dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70" \
  "${DUAL_CONFIG}" \
  "$((PORT_BASE + 2))"

python3 scripts/plot_dual_pattern_confusion.py \
  --metrics "outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/confusion_matrix/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70_metrics.json" \
  --output-dir "outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/paper_confusion" \
  --title "Dual-pattern no-RPM synthetic-transfer"

write_summary "finished" ""
cp "${SUMMARY_PATH}" "${MARKER_PATH}"
