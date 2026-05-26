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

CONFIG="configs/rebuild/retries/dual_pattern_realonly_v2_450_ep70.yaml"
RUN_NAME="dual_pattern_realonly_v2_450_ep70"
WAIT_SESSION="${DUAL_PATTERN_WAIT_SESSION:-no_rpm_realonly_993}"
LOG_DIR="outputs/rebuild_reproduction/logs"
SUMMARY_DIR="outputs/rebuild_reproduction/dual_pattern_queue"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
SUMMARY_PATH="${SUMMARY_DIR}/summary.json"
MARKER_PATH="${MARKER_DIR}/dual_pattern_queue.done"
PORT="${DUAL_PATTERN_PORT:-30180}"
NPROC="${NPROC_PER_NODE:-4}"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${MARKER_DIR}"
rm -f "${MARKER_PATH}"

write_summary() {
  local status="$1"
  python3 - "${SUMMARY_PATH}" "${RUN_NAME}" "${CONFIG}" "${status}" "${LOG_PATH}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, status, log_path = sys.argv[1:6]
path = Path(summary_path)
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "run_name": run_name,
        "config": config,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log": log_path,
        "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
        "metrics": f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json",
        "paper_confusion_dir": f"outputs/rebuild_reproduction/{run_name}/paper_confusion",
    }
)
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

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
  METRICS="outputs/rebuild_reproduction/${RUN_NAME}/confusion_matrix/${RUN_NAME}_metrics.json"
  PAPER_DIR="outputs/rebuild_reproduction/${RUN_NAME}/paper_confusion"
  python3 scripts/plot_dual_pattern_confusion.py \
    --metrics "${METRICS}" \
    --output-dir "${PAPER_DIR}" \
    --title "Dual-pattern real-only V2-450"
  write_summary "finished"
else
  write_summary "failed"
fi

cp "${SUMMARY_PATH}" "${MARKER_PATH}"
exit "${STATUS}"
