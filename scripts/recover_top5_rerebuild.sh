#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REREBUILD_CONFIG_ROOT="${REREBUILD_CONFIG_ROOT:-outputs/rebuild_reproduction/rerebuild_configs}"
WINDOW_DATASET="${WINDOW_DATASET:-outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1}"
FINAL_LABEL="${FINAL_LABEL:-four_run_rerebuild}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29526}"
INFER_MASTER_PORT="${INFER_MASTER_PORT:-29630}"
REQUIRE_WANDB="${REQUIRE_WANDB:-1}"
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-1}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export WANDB_PROJECT="${REREBUILD_WANDB_PROJECT:-re-rebuild-viscnet}"
export WANDB_ENTITY="${REREBUILD_WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

source_configs=(
  "configs/rebuild/synthetic_pretrain.yaml"
  "configs/rebuild/retries/transfer_993_batch8_normal_lr3e5_ep70.yaml"
  "configs/rebuild/retries/realonly_993_batch8_normal_lr3e5_ep70.yaml"
  "configs/rebuild/retries/realonly_993_window30x21_ep50.yaml"
)

pre_window_configs=(
  "configs/rebuild/synthetic_pretrain.yaml"
  "configs/rebuild/retries/transfer_993_batch8_normal_lr3e5_ep70.yaml"
  "configs/rebuild/retries/realonly_993_batch8_normal_lr3e5_ep70.yaml"
)

window_source_config="configs/rebuild/retries/realonly_993_window30x21_ep50.yaml"

required_paths=(
  "configs/rebuild/manifests/real_train_993.json"
  "dataset/CFDArchive/sph_35000/videos"
  "dataset/RealArchive/train_993_wo_pat2/videos"
  "dataset/RealArchive/test_1000_wo_pat2/videos"
)

for path in "${source_configs[@]}" "${required_paths[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "Required path is missing: ${path}" >&2
    exit 1
  fi
done

if [ "${REQUIRE_WANDB}" = "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Create .env or export WANDB_API_KEY before launch." >&2
  exit 1
fi

mkdir -p "${REREBUILD_CONFIG_ROOT}"
run_configs=()
for config in "${source_configs[@]}"; do
  out="${REREBUILD_CONFIG_ROOT}/$(basename "${config}")"
  python3 - "${config}" "${out}" "${WANDB_PROJECT}" "${WANDB_ENTITY}" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst, project, entity = sys.argv[1:5]
cfg = yaml.safe_load(Path(src).read_text())
cfg["project"] = project
cfg["entity"] = entity
cfg["dataset"]["train"]["dataloader"]["batch_size"] = 8
cfg["dataset"]["test"]["dataloader"]["batch_size"] = 8
cfg.get("training", {}).pop("update_density", None)
cfg.setdefault("training", {}).setdefault("acceptance", {})["allow_batch_size_override"] = True
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  run_configs+=("${out}")
done

config_list="${run_configs[*]}"
pre_window_run_configs=()
for config in "${pre_window_configs[@]}"; do
  pre_window_run_configs+=("${REREBUILD_CONFIG_ROOT}/$(basename "${config}")")
done
pre_window_config_list="${pre_window_run_configs[*]}"
window_config="${REREBUILD_CONFIG_ROOT}/$(basename "${window_source_config}")"

REBUILD_CONFIGS="${pre_window_config_list}" \
REBUILD_INCLUDE_RETRIES=1 \
REBUILD_FINAL_ANALYSIS=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${TRAIN_MASTER_PORT}" \
bash scripts/run_rebuild_reproduction.sh

python3 scripts/build_real_window_dataset.py --output-root "${WINDOW_DATASET}"

REBUILD_CONFIGS="${window_config}" \
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
  --config "${window_config}"

final_args=(--label "${FINAL_LABEL}")
if [ "${REBUILD_NOTIFY_WANDB}" = "1" ]; then
  final_args+=(--wandb-alert)
fi

REBUILD_CHECK_CONFIGS="${config_list}" python3 scripts/post_rebuild_training.py "${final_args[@]}"
