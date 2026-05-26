#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export WANDB_PROJECT="${ALL_QUEUE_WANDB_PROJECT:-re-rebuild-viscnet}"
export WANDB_ENTITY="${ALL_QUEUE_WANDB_ENTITY:-jongwonsohn-seoul-national-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

OUTPUT_ROOT="${MOD_FRAME_OUTPUT_ROOT:-outputs/rebuild_reproduction/mod_frame_test_validation}"
LOG_ROOT="${MOD_FRAME_LOG_ROOT:-outputs/rebuild_reproduction/logs}"
SUMMARY_JSON="${OUTPUT_ROOT}/summary.json"
START_PORT="${START_PORT:-29920}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

configs=(
  "outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_realonly_993_window10_stride5_phase5_ep50.yaml"
  "outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45.yaml"
  "outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55.yaml"
)

checkpoint_for() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["misc_dir"]["ckpt_root"]) / cfg["training"]["checkpoint_name"])
PY
}

run_name_for() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["checkpoint_name"]).stem)
PY
}

curr_checkpoint_for() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["misc_dir"]["ckpt_root"]) / cfg["training"].get("curr_ckpt", ""))
PY
}

record_summary() {
  python3 - "${SUMMARY_JSON}" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
run_name, config, status, output = sys.argv[2:6]
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
else:
    summary = {"runs": []}
row = {
    "run_name": run_name,
    "config": config,
    "status": status,
    "output": output,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
summary["runs"] = [item for item in summary.get("runs", []) if item.get("run_name") != run_name]
summary["runs"].append(row)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY
}

port="${START_PORT}"
for config in "${configs[@]}"; do
  run_name="$(run_name_for "${config}")"
  checkpoint="$(checkpoint_for "${config}")"
  if [ ! -f "${checkpoint}" ]; then
    curr_checkpoint="$(curr_checkpoint_for "${config}")"
    if grep -q "curr_bool: true" "${config}" && [ ! -f "${curr_checkpoint}" ]; then
      echo "Missing current checkpoint ${curr_checkpoint}; refusing to rebuild synthetic data for ${run_name}" >&2
      record_summary "${run_name}" "${config}" "blocked_missing_curr_checkpoint_no_synth_rebuild" "${OUTPUT_ROOT}/${run_name}/summary.json"
      exit 1
    fi
    echo "=== $(date -Iseconds) :: training missing real mod-frame checkpoint ${run_name} ==="
    record_summary "${run_name}" "${config}" "training" "${OUTPUT_ROOT}/${run_name}/summary.json"
    REBUILD_CONFIGS="${config}" \
    REBUILD_INCLUDE_RETRIES=1 \
    REBUILD_FINAL_ANALYSIS=0 \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    MASTER_PORT="${port}" \
    bash scripts/run_rebuild_reproduction.sh
    port=$((port + 1))
  fi

  echo "=== $(date -Iseconds) :: modulo-frame real-test inference ${run_name} ==="
  record_summary "${run_name}" "${config}" "evaluating" "${OUTPUT_ROOT}/${run_name}/summary.json"
  PYTHONPATH=src python3 scripts/mod_frame_test_inference.py \
    --config "${config}" \
    --output-root "${OUTPUT_ROOT}" \
    --sample-stride 5 \
    --phase-offsets all \
    --materialize-clips \
    > "${LOG_ROOT}/mod_frame_test_${run_name}.log" 2>&1
  record_summary "${run_name}" "${config}" "done" "${OUTPUT_ROOT}/${run_name}/summary.json"
done

python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    data = json.loads(path.read_text())
    rows.append({
        "run_name": data["run_name"],
        "source_accuracy": data["source_accuracy"],
        "clip_accuracy": data["clip_accuracy"],
        "source_count": data["source_count"],
        "clip_count": data["clip_count"],
        "summary": str(path),
    })
(root / "results.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
lines = ["# Mod-Frame Test Validation", "", "| Run | Source Acc | Clip Acc | Sources | Clips |", "| --- | ---: | ---: | ---: | ---: |"]
for row in rows:
    lines.append(
        f"| `{row['run_name']}` | {row['source_accuracy']:.4f} | {row['clip_accuracy']:.4f} | "
        f"{row['source_count']} | {row['clip_count']} |"
    )
(root / "results.md").write_text("\n".join(lines) + "\n")
print(root / "results.json")
print(root / "results.md")
PY
