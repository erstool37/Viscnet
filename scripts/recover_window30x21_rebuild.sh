#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WINDOW_CONFIG="${WINDOW_CONFIG:-configs/rebuild/retries/realonly_993_window30x21_ep50.yaml}"
WINDOW_DATASET="${WINDOW_DATASET:-outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1}"
REREBUILD_CONFIG_ROOT="${REREBUILD_CONFIG_ROOT:-outputs/rebuild_reproduction/rerebuild_configs}"
FINAL_LABEL="${FINAL_LABEL:-window30x21_recovery}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29525}"
INFER_MASTER_PORT="${INFER_MASTER_PORT:-29630}"
REQUIRE_WANDB="${REQUIRE_WANDB:-1}"
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-1}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export WANDB_PROJECT="${RECOVERY_WANDB_PROJECT:-re-rebuild-viscnet}"
export WANDB_ENTITY="${RECOVERY_WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

required_paths=(
  "${WINDOW_CONFIG}"
  "configs/rebuild/manifests/real_train_993.json"
  "dataset/RealArchive/train_993_wo_pat2/videos"
  "dataset/RealArchive/test_1000_wo_pat2/videos"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "Required path is missing: ${path}" >&2
    exit 1
  fi
done

if [ "${REQUIRE_WANDB}" = "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Create .env or export WANDB_API_KEY before launch." >&2
  echo "Set REQUIRE_WANDB=0 only for a deliberate non-W&B dry recovery." >&2
  exit 1
fi

python3 scripts/build_real_window_dataset.py --output-root "${WINDOW_DATASET}"

mkdir -p "${REREBUILD_CONFIG_ROOT}"
WINDOW_RUN_CONFIG="${REREBUILD_CONFIG_ROOT}/$(basename "${WINDOW_CONFIG}")"
python3 - "${WINDOW_CONFIG}" "${WINDOW_RUN_CONFIG}" "${WANDB_PROJECT}" "${WANDB_ENTITY}" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst, project, entity = sys.argv[1:5]
cfg = yaml.safe_load(Path(src).read_text())
cfg["project"] = project
cfg["entity"] = entity
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

REBUILD_CONFIGS="${WINDOW_RUN_CONFIG}" \
REBUILD_INCLUDE_RETRIES=1 \
REBUILD_FINAL_ANALYSIS=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${TRAIN_MASTER_PORT}" \
bash scripts/run_rebuild_reproduction.sh

MASTER_PORT="${INFER_MASTER_PORT}" \
PYTHONPATH=src \
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${INFER_MASTER_PORT}" \
  scripts/window21_test_inference.py \
  --config "${WINDOW_RUN_CONFIG}"

final_args=(--label "${FINAL_LABEL}")
if [ "${REBUILD_NOTIFY_WANDB}" = "1" ]; then
  final_args+=(--wandb-alert)
fi

REBUILD_CHECK_CONFIGS="${WINDOW_RUN_CONFIG}" python3 scripts/post_rebuild_training.py "${final_args[@]}"
