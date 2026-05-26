#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

REREBUILD_CONFIG_ROOT="${REREBUILD_CONFIG_ROOT:-outputs/rebuild_reproduction/rerebuild_configs}"
WINDOW_DATASET="${WINDOW_DATASET:-outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1}"
FINAL_LABEL="${FINAL_LABEL:-remaining_batch8_normal}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29533}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-29534}"
INFER_MASTER_PORT="${INFER_MASTER_PORT:-29633}"
REQUIRE_WANDB="${REQUIRE_WANDB:-1}"
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-1}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export WANDB_PROJECT="${REREBUILD_WANDB_PROJECT:-re-rebuild-viscnet}"
export WANDB_ENTITY="${REREBUILD_WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

REALONLY_CONFIG="configs/rebuild/retries/realonly_993_batch8_normal_lr3e5_ep70.yaml"
TRANSFER_CONFIG="configs/rebuild/retries/transfer_993_batch8_normal_lr3e5_ep70.yaml"
WINDOW_SOURCE_CONFIG="configs/rebuild/retries/realonly_993_window30x21_ep50.yaml"

REALONLY_CKPT="outputs/rebuild_reproduction/checkpoints/repro_realonly_993_batch8_normal_lr3e5_ep70.pth"
SYNTHETIC_CKPT="outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000.pth"

required_paths=(
  "${REALONLY_CONFIG}"
  "${TRANSFER_CONFIG}"
  "${WINDOW_SOURCE_CONFIG}"
  "${REALONLY_CKPT}"
  "${SYNTHETIC_CKPT}"
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
  exit 1
fi

mkdir -p "${REREBUILD_CONFIG_ROOT}" outputs/rebuild_reproduction/logs

REALONLY_RUN_CONFIG="${REREBUILD_CONFIG_ROOT}/$(basename "${REALONLY_CONFIG}")"
TRANSFER_RUN_CONFIG="${REREBUILD_CONFIG_ROOT}/$(basename "${TRANSFER_CONFIG}")"
WINDOW_RUN_CONFIG="${REREBUILD_CONFIG_ROOT}/$(basename "${WINDOW_SOURCE_CONFIG}")"
REALONLY_EVAL_CONFIG="${REREBUILD_CONFIG_ROOT}/eval_$(basename "${REALONLY_CONFIG}")"
TRANSFER_EVAL_CONFIG="${REREBUILD_CONFIG_ROOT}/eval_$(basename "${TRANSFER_CONFIG}")"

python3 - \
  "${REALONLY_CONFIG}" "${REALONLY_RUN_CONFIG}" \
  "${TRANSFER_CONFIG}" "${TRANSFER_RUN_CONFIG}" \
  "${WINDOW_SOURCE_CONFIG}" "${WINDOW_RUN_CONFIG}" \
  "${REALONLY_EVAL_CONFIG}" "${TRANSFER_EVAL_CONFIG}" \
  "${WANDB_PROJECT}" "${WANDB_ENTITY}" <<'PY'
from pathlib import Path
import sys
import yaml

(
    realonly_src,
    realonly_dst,
    transfer_src,
    transfer_dst,
    window_src,
    window_dst,
    realonly_eval_dst,
    transfer_eval_dst,
    project,
    entity,
) = sys.argv[1:11]


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["project"] = project
    cfg["entity"] = entity
    cfg["dataset"]["train"]["dataloader"]["batch_size"] = 8
    cfg["dataset"]["test"]["dataloader"]["batch_size"] = 8
    cfg.get("training", {}).pop("update_density", None)
    cfg.setdefault("training", {}).setdefault("acceptance", {})["allow_batch_size_override"] = True
    return cfg


def write_config(path: str, cfg: dict) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


realonly = load_config(realonly_src)
transfer = load_config(transfer_src)
window = load_config(window_src)

# Avoid the distributed final-test gather path during multi-GPU training.
for cfg in [realonly, transfer, window]:
    cfg["train_settings"]["test_bool"] = False

write_config(realonly_dst, realonly)
write_config(transfer_dst, transfer)
write_config(window_dst, window)

for cfg, dst in [(realonly, realonly_eval_dst), (transfer, transfer_eval_dst)]:
    eval_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    eval_cfg["train_settings"]["train_bool"] = False
    eval_cfg["train_settings"]["test_bool"] = True
    eval_cfg["train_settings"]["val_test_bool"] = False
    eval_cfg["training"]["curr_bool"] = True
    eval_cfg["training"]["curr_ckpt"] = eval_cfg["training"]["checkpoint_name"]
    write_config(dst, eval_cfg)
PY

echo "=== $(date -Iseconds) :: eval realonly batch8 normal ==="
PYTHONPATH=src torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --master_port="${EVAL_MASTER_PORT}" \
  --node_rank=0 \
  src/main.py -c "${REALONLY_EVAL_CONFIG}" \
  > outputs/rebuild_reproduction/logs/eval_repro_realonly_993_batch8_normal_lr3e5_ep70.log 2>&1

echo "=== $(date -Iseconds) :: train transfer batch8 normal ==="
REBUILD_CONFIGS="${TRANSFER_RUN_CONFIG}" \
REBUILD_INCLUDE_RETRIES=1 \
REBUILD_FINAL_ANALYSIS=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${TRAIN_MASTER_PORT}" \
bash scripts/run_rebuild_reproduction.sh

echo "=== $(date -Iseconds) :: eval transfer batch8 normal ==="
PYTHONPATH=src torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --master_port="$((EVAL_MASTER_PORT + 1))" \
  --node_rank=0 \
  src/main.py -c "${TRANSFER_EVAL_CONFIG}" \
  > outputs/rebuild_reproduction/logs/eval_repro_transfer_993_batch8_normal_lr3e5_ep70.log 2>&1

echo "=== $(date -Iseconds) :: rebuild window dataset ==="
python3 scripts/build_real_window_dataset.py --output-root "${WINDOW_DATASET}" --force

echo "=== $(date -Iseconds) :: train window30x21 ==="
REBUILD_CONFIGS="${WINDOW_RUN_CONFIG}" \
REBUILD_INCLUDE_RETRIES=1 \
REBUILD_FINAL_ANALYSIS=0 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="$((TRAIN_MASTER_PORT + 1))" \
bash scripts/run_rebuild_reproduction.sh

echo "=== $(date -Iseconds) :: window21 inference ==="
PYTHONPATH=src torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${INFER_MASTER_PORT}" \
  scripts/window21_test_inference.py \
  --config "${WINDOW_RUN_CONFIG}"

final_args=(--label "${FINAL_LABEL}")
if [ "${REBUILD_NOTIFY_WANDB}" = "1" ]; then
  final_args+=(--wandb-alert)
fi

echo "=== $(date -Iseconds) :: post rebuild training ==="
REBUILD_CHECK_CONFIGS="${REALONLY_RUN_CONFIG} ${TRANSFER_RUN_CONFIG} ${WINDOW_RUN_CONFIG}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
python3 scripts/post_rebuild_training.py "${final_args[@]}"
