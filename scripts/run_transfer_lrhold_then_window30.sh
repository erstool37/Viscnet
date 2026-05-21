#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

signal_done() {
  if [ -n "${REBUILD_DONE_SIGNAL:-}" ]; then
    tmux wait-for -S "${REBUILD_DONE_SIGNAL}" || true
  fi
}

trap signal_done EXIT

transfer_config="configs/rebuild/retries/transfer_993_microbatch_lrhold_ep70_min87.yaml"
window_config="configs/rebuild/retries/realonly_993_window30x21_ep50.yaml"
transfer_run="repro_transfer_993_microbatch_lrhold_ep70_min87"

REBUILD_CONFIGS="${transfer_config}" \
REBUILD_INCLUDE_RETRIES=1 \
REBUILD_FINAL_ANALYSIS=1 \
REBUILD_FINAL_LABEL=transfer_lrhold_ep70_min87_before_window30 \
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-1}" \
REBUILD_DONE_SIGNAL= \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
MASTER_PORT="${TRANSFER_MASTER_PORT:-29524}" \
bash scripts/run_rebuild_reproduction.sh

python scripts/require_rebuild_accuracy.py --run-name "${transfer_run}" --min-accuracy 0.8701

python scripts/build_real_window_dataset.py

REBUILD_CONFIGS="${window_config}" \
REBUILD_INCLUDE_RETRIES=1 \
REBUILD_FINAL_ANALYSIS=1 \
REBUILD_FINAL_LABEL=window30x21_ep50_after_transfer_lrhold \
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-1}" \
REBUILD_DONE_SIGNAL= \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
MASTER_PORT="${WINDOW_MASTER_PORT:-29525}" \
bash scripts/run_rebuild_reproduction.sh
