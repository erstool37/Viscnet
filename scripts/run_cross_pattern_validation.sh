#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_FOR_QUEUE="${WAIT_FOR_QUEUE:-1}"
DEVICE="${CROSS_PATTERN_DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rebuild_reproduction/cross_pattern_validation}"
DATASET_ROOT="${DATASET_ROOT:-dataset/RealArchive/dualpatterndataset_V2_450}"
MAX_SAMPLES_PER_RENDER="${MAX_SAMPLES_PER_RENDER:-}"

if [ "${WAIT_FOR_QUEUE}" = "1" ] && tmux has-session -t all_pending_training_queue 2>/dev/null; then
  echo "=== $(date -Iseconds) :: waiting for all_pending_training_queue_done ==="
  tmux wait-for all_pending_training_queue_done
fi

args=(
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
)
if [ -n "${MAX_SAMPLES_PER_RENDER}" ]; then
  args+=(--max-samples-per-render "${MAX_SAMPLES_PER_RENDER}")
fi

run_if_missing() {
  local name="$1"
  shift
  if [ -f "${OUTPUT_ROOT}/${name}/metrics.json" ]; then
    echo "=== $(date -Iseconds) :: skip existing cross-pattern metrics ${name} ==="
    return 0
  fi
  python3 scripts/cross_pattern_validation.py --name "${name}" "$@" "${args[@]}"
}

run_if_missing \
  repro_realonly_993_window30x21_ep50 \
  --config outputs/rebuild_reproduction/rerebuild_configs/realonly_993_window30x21_ep50.yaml

run_if_missing \
  repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93 \
  --config outputs/rebuild_reproduction/transfer_min93_queue/configs/transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.yaml

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_legacy10fps_224px.pth ]; then
  run_if_missing \
    rawfps_realonly_existing993_1000_legacy10fps_224px \
    --config configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_224px.yaml
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_nativefps_224px.pth ]; then
  run_if_missing \
    rawfps_realonly_existing993_1000_nativefps_224px \
    --config configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_nativefps_224px.yaml
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_ratio1500_500_legacy10fps_224px.pth ]; then
  run_if_missing \
    rawfps_realonly_ratio1500_500_legacy10fps_224px \
    --config configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_legacy10fps_224px.yaml
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_ratio1500_500_nativefps_224px.pth ]; then
  run_if_missing \
    rawfps_realonly_ratio1500_500_nativefps_224px \
    --config configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_nativefps_224px.yaml
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_legacy10fps_336px.pth ]; then
  run_if_missing \
    rawfps_realonly_existing993_1000_legacy10fps_336px \
    --config configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_336px.yaml \
    --image-size 336
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_nativefps_336px.pth ]; then
  run_if_missing \
    rawfps_realonly_existing993_1000_nativefps_336px \
    --config configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_nativefps_336px.yaml \
    --image-size 336
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_ratio1500_500_legacy10fps_336px.pth ]; then
  run_if_missing \
    rawfps_realonly_ratio1500_500_legacy10fps_336px \
    --config configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_legacy10fps_336px.yaml \
    --image-size 336
fi

if [ -f outputs/rebuild_reproduction/checkpoints/rawfps_realonly_ratio1500_500_nativefps_336px.pth ]; then
  run_if_missing \
    rawfps_realonly_ratio1500_500_nativefps_336px \
    --config configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_nativefps_336px.yaml \
    --image-size 336
fi

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
(root / "summary.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
lines = ["# Cross-Pattern Validation Summary", "", "| Run | Accuracy | Samples | AB | CD | EF |", "|---|---:|---:|---:|---:|---:|"]
for row in rows:
    fam = row["by_render_family"]
    lines.append(
        f"| {row['run_name']} | {row['accuracy']:.4f} | {row['sample_count']} | "
        f"{fam.get('AB', {}).get('accuracy', 0):.4f} | {fam.get('CD', {}).get('accuracy', 0):.4f} | {fam.get('EF', {}).get('accuracy', 0):.4f} |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(root / "summary.json")
print(root / "summary.md")
PY
