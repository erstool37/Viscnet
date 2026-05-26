#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

mkdir -p outputs/rebuild_reproduction/logs outputs/rebuild_reproduction/window21_clip_eval_queue

SUMMARY_JSON="outputs/rebuild_reproduction/window21_clip_eval_queue/summary.json"
START_PORT="${START_PORT:-30120}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WINDOW_BATCH_SIZE="${WINDOW21_CLIP_BATCH_SIZE:-32}"
export WANDB_PROJECT="re-rebuild-viscnet"
export WANDB_ENTITY="${WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

python3 - "${SUMMARY_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "runs": []}, indent=2) + "\n")
PY

record_run() {
  local run_name="$1"
  local config="$2"
  local status="$3"
  local output_dir="$4"
  local wandb_json="${5:-}"
  python3 - "${SUMMARY_JSON}" "${run_name}" "${config}" "${status}" "${output_dir}" "${wandb_json}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, run_name, config, status, output_dir, wandb_json = sys.argv[1:7]
summary_path = Path(summary_path)
summary = json.loads(summary_path.read_text())
row = {
    "run_name": run_name,
    "config": config,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "output_dir": output_dir,
    "metrics": f"{output_dir}/window21_clip_test_inference_metrics.json",
    "predictions": f"{output_dir}/window21_clip_test_inference_predictions.json",
}
if wandb_json:
    row["wandb"] = json.loads(wandb_json)
summary["runs"] = [item for item in summary.get("runs", []) if item.get("run_name") != run_name]
summary["runs"].append(row)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY
}

log_wandb() {
  local label="$1"
  local config="$2"
  local output_dir="$3"
  local metrics="${output_dir}/window21_clip_test_inference_metrics.json"
  local predictions="${output_dir}/window21_clip_test_inference_predictions.json"
  local wandb_json
  wandb_json="$(python3 scripts/log_eval_metrics_to_wandb.py \
    --run-name "${label}" \
    --config "${config}" \
    --output-dir "${output_dir}" \
    --metrics "${metrics}" \
    --predictions "${predictions}")"
  record_run "${label}" "${config}" "wandb_logged" "${output_dir}" "${wandb_json}"
}

run_eval() {
  local label="$1"
  local config="$2"
  local output_dir="$3"
  local port="$4"
  local metrics="${output_dir}/window21_clip_test_inference_metrics.json"
  if [ -f "${metrics}" ]; then
    record_run "${label}" "${config}" "skipped_existing_metrics" "${output_dir}"
    log_wandb "${label}" "${config}" "${output_dir}"
    return 0
  fi
  mkdir -p "${output_dir}"
  record_run "${label}" "${config}" "running" "${output_dir}"
  PYTHONPATH=src torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${port}" \
    scripts/window21_test_inference.py \
    --config "${config}" \
    --output-dir "${output_dir}" \
    --prediction-mode clip \
    --window-batch-size "${WINDOW_BATCH_SIZE}" \
    > "outputs/rebuild_reproduction/logs/${label}.log" 2>&1
  record_run "${label}" "${config}" "done" "${output_dir}"
  log_wandb "${label}" "${config}" "${output_dir}"
}

run_eval \
  "repro_realonly_993_window30x21_ep50_window21_clip" \
  "configs/rebuild/retries/realonly_993_window30x21_ep50.yaml" \
  "outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_clip_test_inference" \
  "${START_PORT}"

run_eval \
  "repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93_window21_clip" \
  "configs/rebuild/retries/transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.yaml" \
  "outputs/rebuild_reproduction/repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93/window21_clip_test_inference" \
  "$((START_PORT + 1))"

python3 - "${SUMMARY_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text())
summary["finished_at"] = datetime.now(timezone.utc).isoformat()
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY
