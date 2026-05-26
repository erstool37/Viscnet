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

QUEUE_ROOT="${PATTERN_LOPO_QUEUE_ROOT:-outputs/rebuild_reproduction/pattern_lopo_queue}"
SUMMARY_JSON="${QUEUE_ROOT}/summary.json"
LOG_ROOT="${PATTERN_LOPO_LOG_ROOT:-outputs/rebuild_reproduction/logs}"
START_PORT="${START_PORT:-29940}"
BATCH_CANDIDATES="${PATTERN_LOPO_BATCH_CANDIDATES:-8 4}"
SYNTHETIC30_CKPT="outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_window30_batch8_ep50.pth"

mkdir -p "${QUEUE_ROOT}" "${LOG_ROOT}"

if [ ! -f "${SYNTHETIC30_CKPT}" ]; then
  echo "Missing ${SYNTHETIC30_CKPT}; refusing to rebuild synthetic data in pattern LOPO queue." >&2
  exit 1
fi

python3 scripts/build_pattern_lopo_configs.py > "${QUEUE_ROOT}/build_pattern_lopo_configs.log"

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
  python3 - "${SUMMARY_JSON}" "${run_name}" "${config}" "${status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
summary_path, run_name, config, status = sys.argv[1:5]
summary = json.loads(Path(summary_path).read_text())
row = {
    "run_name": run_name,
    "config": config,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
    "metrics": f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json",
}
summary["runs"] = [item for item in summary.get("runs", []) if item.get("run_name") != run_name]
summary["runs"].append(row)
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
PY
}

make_batch_config() {
  local config="$1"
  local batch="$2"
  local dst="${QUEUE_ROOT}/$(basename "${config}")"
  python3 - "${config}" "${dst}" "${batch}" <<'PY'
from pathlib import Path
import sys
import yaml
src, dst, batch = sys.argv[1:4]
cfg = yaml.safe_load(Path(src).read_text())
cfg["project"] = __import__("os").environ.get("WANDB_PROJECT", cfg.get("project"))
cfg["entity"] = __import__("os").environ.get("WANDB_ENTITY", cfg.get("entity"))
cfg["dataset"]["train"]["dataloader"]["batch_size"] = int(batch)
cfg["dataset"]["test"]["dataloader"]["batch_size"] = int(batch)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${dst}"
}

run_config_with_fallback() {
  local config="$1"
  local port="$2"
  local run_name
  run_name="$(python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["checkpoint_name"]).stem)
PY
)"
  local checkpoint="outputs/rebuild_reproduction/checkpoints/${run_name}.pth"
  local metric="outputs/rebuild_reproduction/${run_name}/confusion_matrix/${run_name}_metrics.json"
  if [ -f "${checkpoint}" ] && [ -f "${metric}" ]; then
    record_run "${run_name}" "${config}" "skipped_existing_metrics"
    return 0
  fi
  for batch in ${BATCH_CANDIDATES}; do
    local batch_config
    batch_config="$(make_batch_config "${config}" "${batch}")"
    echo "=== $(date -Iseconds) :: pattern LOPO ${run_name} batch=${batch} ==="
    record_run "${run_name}" "${config}" "running_batch${batch}"
    set +e
    REBUILD_CONFIGS="${batch_config}" \
    REBUILD_INCLUDE_RETRIES=1 \
    REBUILD_FINAL_ANALYSIS=0 \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    MASTER_PORT="${port}" \
    bash scripts/run_rebuild_reproduction.sh
    status="$?"
    set -e
    if [ "${status}" = "0" ]; then
      record_run "${run_name}" "${config}" "done_batch${batch}"
      return 0
    fi
    if [ -f "${LOG_ROOT}/${run_name}.log" ] && grep -Eiq "out of memory|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|DefaultCPUAllocator" "${LOG_ROOT}/${run_name}.log"; then
      record_run "${run_name}" "${config}" "oom_batch${batch}"
      rm -f "${checkpoint}"
      continue
    fi
    record_run "${run_name}" "${config}" "failed_batch${batch}"
    return "${status}"
  done
  record_run "${run_name}" "${config}" "failed_all_batches"
  return 1
}

configs=(
  configs/rebuild/pattern_lopo/pattern_lopo_realonly_train_not1_test1.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_transfer_synth30_train_not1_test1.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_realonly_train_not2_test2.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_transfer_synth30_train_not2_test2.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_realonly_train_not3_test3.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_transfer_synth30_train_not3_test3.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_realonly_train_not4_test4.yaml
  configs/rebuild/pattern_lopo/pattern_lopo_transfer_synth30_train_not4_test4.yaml
)

port="${START_PORT}"
for config in "${configs[@]}"; do
  run_config_with_fallback "${config}" "${port}"
  port=$((port + 1))
done

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
