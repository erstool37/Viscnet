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

CONFIG="configs/rebuild/retries/realonly_993_batch8_normal_no_rpm_lr1e5_ep90.yaml"
RUN_NAME="repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90"
WAIT_SESSION="${NO_RPM_FOLLOWUP_WAIT_SESSION:-no_rpm_transfer_policy_queue}"
LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/no_rpm_realonly_followup_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/no_rpm_realonly_followup_queue.done"
PORT="${NO_RPM_REALONLY_FOLLOWUP_PORT:-30230}"
NPROC="${NPROC_PER_NODE:-4}"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  python3 - "${SUMMARY_PATH}" "${RUN_NAME}" "${CONFIG}" "${status}" "${LOG_PATH}" "${WAIT_SESSION}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, status, log_path, wait_session = sys.argv[1:7]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "run_name": run_name,
        "config": config,
        "status": status,
        "wait_session": wait_session,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log": log_path,
        "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
        "metrics": f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json",
        "reliability": f"outputs/rebuild_reproduction/{run_name}/reliability_plots/{run_name}_metrics.json",
        "policy": "no RPM embeddings",
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

python3 scripts/verify_no_rpm_policy.py "${CONFIG}"
write_summary "waiting_for_${WAIT_SESSION}"
while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 60
done

write_summary "running"
set +e
PYTHONPATH=src torchrun \
  --nproc_per_node="${NPROC}" \
  --nnodes=1 \
  --master_port="${PORT}" \
  --node_rank=0 \
  src/main.py \
  -c "${CONFIG}" \
  > "${LOG_PATH}" 2>&1
STATUS=$?
set -e

if [ "${STATUS}" -eq 0 ]; then
  write_summary "finished"
else
  write_summary "failed"
fi

python3 - "${SUMMARY_PATH}" "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, status = sys.argv[1:3]
path = Path(summary_path)
payload = json.loads(path.read_text())
payload["exit_status"] = int(status)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

cp "${SUMMARY_PATH}" "${MARKER_PATH}"
exit "${STATUS}"
