#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rebuild_reproduction/cross_pattern_validation}"
COMMON_ARGS=(
  --dataset-root dataset/RealArchive/dualpatterndataset_V2_450
  --output-root "${OUTPUT_ROOT}"
  --device cpu
  --batch-size 1
)

run_if_missing() {
  local name="$1"
  shift
  if [ -f "${OUTPUT_ROOT}/${name}/metrics.json" ]; then
    echo "=== $(date -Iseconds) :: skip existing CPU cross-pattern metrics ${name} ==="
    return 0
  fi
  echo "=== $(date -Iseconds) :: CPU cross-pattern ${name} ==="
  python3 scripts/cross_pattern_validation.py --name "${name}" "$@" "${COMMON_ARGS[@]}"
}

run_if_missing \
  repro_realonly_993_window30x21_ep50 \
  --config outputs/rebuild_reproduction/rerebuild_configs/realonly_993_window30x21_ep50.yaml

run_if_missing \
  repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93 \
  --config outputs/rebuild_reproduction/transfer_min93_queue/configs/transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.yaml

run_if_missing \
  rawfps_realonly_existing993_1000_legacy10fps_224px \
  --config configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_224px.yaml

python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    data = json.loads(metrics_path.read_text())
    rows.append({
        "run_name": metrics_path.parent.name,
        "accuracy": data["accuracy"],
        "sample_count": data["sample_count"],
        "by_render_family": data["by_render_family"],
        "metrics": str(metrics_path),
        "report": str(metrics_path.parent / "report.md"),
    })
(root / "summary_prefill.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
print(root / "summary_prefill.json")
PY
