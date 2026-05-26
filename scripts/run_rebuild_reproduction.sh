#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29513}"

mkdir -p outputs/rebuild_reproduction/logs

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

signal_rebuild_done() {
  if [ -n "${REBUILD_DONE_SIGNAL:-}" ]; then
    tmux wait-for -S "${REBUILD_DONE_SIGNAL}" || true
  fi
}

run_post_analysis() {
  local label="$1"
  python scripts/post_rebuild_training.py --label "${label}"
}

trap signal_rebuild_done EXIT

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set; wandb.init may run unauthenticated or offline depending on W&B settings." >&2
fi

if [ -n "${REBUILD_CONFIGS:-}" ]; then
  read -r -a configs <<< "${REBUILD_CONFIGS}"
else
  configs=(
    configs/rebuild/realonly_993.yaml
    configs/rebuild/transfer_993.yaml
  )
fi

for config in "${configs[@]}"; do
  if [ -n "${REBUILD_START_CONFIG:-}" ]; then
    if [ "${config}" != "${REBUILD_START_CONFIG}" ]; then
      continue
    fi
    unset REBUILD_START_CONFIG
  fi
  if [ -n "${REBUILD_STOP_BEFORE_CONFIG:-}" ]; then
    if [ "${config}" = "${REBUILD_STOP_BEFORE_CONFIG}" ]; then
      echo "=== $(date -Iseconds) :: stopping before ${config} ==="
      break
    fi
  fi
  run_id="$(
    python - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["checkpoint_name"]).stem)
PY
  )"
  log_path="outputs/rebuild_reproduction/logs/${run_id}.log"
  echo "=== $(date -Iseconds) :: ${config} ===" | tee "${log_path}"
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --master_port="${MASTER_PORT}" \
    --node_rank=0 \
    src/main.py -c "${config}" 2>&1 | tee -a "${log_path}"
  if [ "${REBUILD_MIDRUN_ANALYSIS:-0}" = "1" ]; then
    run_post_analysis "midrun_${run_id}"
  fi
  if [ -n "${REBUILD_STOP_AFTER_CONFIG:-}" ]; then
    if [ "${config}" = "${REBUILD_STOP_AFTER_CONFIG}" ]; then
      echo "=== $(date -Iseconds) :: stopping after ${config} ==="
      break
    fi
  fi
done

if [ "${REBUILD_FINAL_ANALYSIS:-0}" = "1" ]; then
  final_args=(--label "${REBUILD_FINAL_LABEL:-rebuild_final}")
  if [ "${REBUILD_NOTIFY_WANDB:-0}" = "1" ]; then
    final_args+=(--wandb-alert)
  fi
  python scripts/post_rebuild_training.py "${final_args[@]}"
fi
