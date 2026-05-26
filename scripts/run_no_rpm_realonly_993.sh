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

CONFIG="configs/rebuild/retries/realonly_993_batch8_normal_no_rpm_lr3e5_ep70.yaml"
RUN_NAME="repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70"
LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/no_rpm_realonly_993"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/no_rpm_realonly_993.done"
PORT="${NO_RPM_REALONLY_PORT:-30170}"
NPROC="${NPROC_PER_NODE:-4}"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

python3 - "${SUMMARY_PATH}" "${CONFIG}" "${RUN_NAME}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, config, run_name = sys.argv[1:4]
payload = {
    "run_name": run_name,
    "config": config,
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "log": f"outputs/rebuild_reproduction/logs/{run_name}.log",
}
Path(summary_path).write_text(json.dumps(payload, indent=2) + "\n")
PY

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

python3 - "${SUMMARY_PATH}" "${STATUS}" "${RUN_NAME}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, status, run_name = sys.argv[1:4]
summary_path = Path(summary_path)
payload = json.loads(summary_path.read_text())
status_int = int(status)
payload["exit_status"] = status_int
payload["status"] = "finished" if status_int == 0 else "failed"
payload["finished_at"] = datetime.now(timezone.utc).isoformat()
payload["checkpoint"] = f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth"
payload["metrics"] = f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json"
payload["reliability"] = f"outputs/rebuild_reproduction/{run_name}/reliability_plots/{run_name}_metrics.json"
summary_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

cp "${SUMMARY_PATH}" "${MARKER_PATH}"
exit "${STATUS}"
