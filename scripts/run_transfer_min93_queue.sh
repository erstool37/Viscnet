#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TARGET_ACCURACY="${TARGET_ACCURACY:-0.93}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29541}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-29561}"
INFER_MASTER_PORT="${INFER_MASTER_PORT:-29661}"
QUEUE_ROOT="${QUEUE_ROOT:-outputs/rebuild_reproduction/transfer_min93_queue}"
CONFIG_ROOT="${CONFIG_ROOT:-${QUEUE_ROOT}/configs}"
SUMMARY_JSON="${SUMMARY_JSON:-${QUEUE_ROOT}/summary.json}"
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-1}"
REQUIRE_WANDB="${REQUIRE_WANDB:-1}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export WANDB_PROJECT="${TRANSFER_MIN93_WANDB_PROJECT:-re-rebuild-viscnet}"
export WANDB_ENTITY="${TRANSFER_MIN93_WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

standard_config="configs/rebuild/retries/transfer_993_batch8_normal_lr1e5_ep90_min93.yaml"
synthetic30_config="configs/rebuild/retries/synthetic_pretrain_window30_batch8_ep50.yaml"
window_configs=(
  "configs/rebuild/retries/transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.yaml"
  "configs/rebuild/retries/transfer_993_window30x21_from_synth30_lr5e6_ep55_min93.yaml"
)
synthetic30_ckpt="outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_window30_batch8_ep50.pth"
WINDOW_DATASET="${WINDOW_DATASET:-outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1}"
WINDOW_SIZE="${WINDOW_SIZE:-30}"
WINDOWS_PER_VIDEO="${WINDOWS_PER_VIDEO:-21}"
WINDOW_SAMPLE_STRIDE="${WINDOW_SAMPLE_STRIDE:-1}"
WINDOW_PHASE_OFFSETS="${WINDOW_PHASE_OFFSETS:-0}"
REBUILD_WINDOW_DATASET="${REBUILD_WINDOW_DATASET:-0}"
WINDOW_ALLOW_PARTIAL_PHASE_OFFSETS="${WINDOW_ALLOW_PARTIAL_PHASE_OFFSETS:-0}"
window_manifest="${WINDOW_DATASET}/manifest.json"

required_paths=(
  "${standard_config}"
  "${synthetic30_config}"
  "${window_configs[@]}"
  "outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000.pth"
  "dataset/CFDArchive/sph_35000/videos"
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
  echo "WANDB_API_KEY is not set. Refusing to launch W&B-backed transfer queue." >&2
  exit 1
fi

mkdir -p "${CONFIG_ROOT}" "${QUEUE_ROOT}" outputs/rebuild_reproduction/logs
printf '{"target_accuracy": %.6f, "results": []}\n' "${TARGET_ACCURACY}" > "${SUMMARY_JSON}"

prepare_config() {
  local src="$1"
  local dst="${CONFIG_ROOT}/$(basename "${src}")"
  python3 - "${src}" "${dst}" "${WANDB_PROJECT}" "${WANDB_ENTITY}" "${WINDOW_DATASET}" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst, project, entity, window_dataset = sys.argv[1:6]
cfg = yaml.safe_load(Path(src).read_text())
cfg["project"] = project
cfg["entity"] = entity
cfg["train_settings"]["test_bool"] = False
cfg["dataset"]["train"]["dataloader"]["batch_size"] = 8
cfg["dataset"]["test"]["dataloader"]["batch_size"] = 8
train_cfg = cfg.get("dataset", {}).get("train", {})
if train_cfg.get("manifest") and "real_train_993_windows30_stride1" in str(train_cfg.get("manifest")):
    train_cfg["train_root"] = window_dataset
    train_cfg["manifest"] = str(Path(window_dataset) / "manifest.json")
cfg.get("training", {}).pop("update_density", None)
cfg.setdefault("training", {}).setdefault("acceptance", {})["allow_batch_size_override"] = True
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${dst}"
}

make_eval_config() {
  local src="$1"
  local dst="${CONFIG_ROOT}/eval_$(basename "${src}")"
  python3 - "${src}" "${dst}" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst = sys.argv[1:3]
cfg = yaml.safe_load(Path(src).read_text())
cfg["train_settings"]["train_bool"] = False
cfg["train_settings"]["test_bool"] = True
cfg["train_settings"]["val_test_bool"] = False
cfg["training"]["curr_bool"] = True
cfg["training"]["curr_ckpt"] = cfg["training"]["checkpoint_name"]
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${dst}"
}

record_result() {
  local label="$1"
  local kind="$2"
  local config="$3"
  local metric_path="$4"
  python3 - "${SUMMARY_JSON}" "${label}" "${kind}" "${config}" "${metric_path}" "${TARGET_ACCURACY}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, label, kind, config, metric_path, target = sys.argv[1:7]
summary = json.loads(Path(summary_path).read_text())
metrics = json.loads(Path(metric_path).read_text())
accuracy = float(metrics["accuracy"])
summary["results"].append(
    {
        "label": label,
        "kind": kind,
        "config": config,
        "metric_path": metric_path,
        "accuracy": accuracy,
        "target_accuracy": float(target),
        "passed_target": accuracy >= float(target),
    }
)
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
print(f"{label} {accuracy:.4f}")
PY
}

meets_target() {
  local metric_path="$1"
  python3 - "${metric_path}" "${TARGET_ACCURACY}" <<'PY'
import json
import sys
from pathlib import Path

accuracy = float(json.loads(Path(sys.argv[1]).read_text())["accuracy"])
raise SystemExit(0 if accuracy >= float(sys.argv[2]) else 1)
PY
}

train_config() {
  local config="$1"
  local port="$2"
  REBUILD_CONFIGS="${config}" \
  REBUILD_INCLUDE_RETRIES=1 \
  REBUILD_FINAL_ANALYSIS=0 \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MASTER_PORT="${port}" \
  bash scripts/run_rebuild_reproduction.sh
}

eval_standard() {
  local config="$1"
  local eval_config
  eval_config="$(make_eval_config "${config}")"
  local run_name
  run_name="$(python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["checkpoint_name"]).stem)
PY
)"
  PYTHONPATH=src torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_port="${EVAL_MASTER_PORT}" \
    --node_rank=0 \
    src/main.py -c "${eval_config}" \
    > "outputs/rebuild_reproduction/logs/eval_${run_name}.log" 2>&1
  echo "outputs/rebuild_reproduction/${run_name}/confusion_matrix/${run_name}_metrics.json"
}

run_window_inference() {
  local config="$1"
  local run_name
  run_name="$(python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["checkpoint_name"]).stem)
PY
)"
  PYTHONPATH=src torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${INFER_MASTER_PORT}" \
    scripts/window21_test_inference.py \
    --config "${config}" \
    > "outputs/rebuild_reproduction/logs/window21_${run_name}.log" 2>&1
  python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(str(Path(cfg["inference"]["temporal_window"]["output_dir"]) / "window21_test_inference_metrics.json"))
PY
}

echo "=== $(date -Iseconds) :: transfer min93 queue start ==="

target_reached=0
completed_configs=()
standard_run_config="$(prepare_config "${standard_config}")"
completed_configs+=("${standard_run_config}")
echo "=== $(date -Iseconds) :: train standard transfer low-lr ==="
train_config "${standard_run_config}" "${TRAIN_MASTER_PORT}"
echo "=== $(date -Iseconds) :: eval standard transfer low-lr ==="
standard_metric="$(eval_standard "${standard_run_config}")"
record_result "standard_transfer_lr1e5_ep90" "standard_eval" "${standard_run_config}" "${standard_metric}"
if meets_target "${standard_metric}"; then
  echo "=== target reached by standard transfer ==="
  target_reached=1
fi

if [ "${target_reached}" = "0" ]; then
  if [ "${REBUILD_WINDOW_DATASET}" = "1" ] || [ ! -f "${window_manifest}" ]; then
    echo "=== $(date -Iseconds) :: rebuild window dataset ==="
    window_build_args=(
      --output-root "${WINDOW_DATASET}"
      --window-size "${WINDOW_SIZE}"
      --windows-per-video "${WINDOWS_PER_VIDEO}"
      --sample-stride "${WINDOW_SAMPLE_STRIDE}"
      --phase-offsets "${WINDOW_PHASE_OFFSETS}"
      --force
    )
    if [ "${WINDOW_ALLOW_PARTIAL_PHASE_OFFSETS}" = "1" ]; then
      window_build_args+=(--allow-partial-phase-offsets)
    fi
    python3 scripts/build_real_window_dataset.py "${window_build_args[@]}"
  fi

  synthetic30_run_config="$(prepare_config "${synthetic30_config}")"
  if [ ! -f "${synthetic30_ckpt}" ]; then
    echo "=== $(date -Iseconds) :: train synthetic30 pretrain ==="
    train_config "${synthetic30_run_config}" "$((TRAIN_MASTER_PORT + 1))"
  else
    echo "=== $(date -Iseconds) :: synthetic30 checkpoint already exists ==="
  fi

  idx=0
  for config in "${window_configs[@]}"; do
    idx=$((idx + 1))
    run_config="$(prepare_config "${config}")"
    completed_configs+=("${run_config}")
    echo "=== $(date -Iseconds) :: train window transfer ${idx} ==="
    train_config "${run_config}" "$((TRAIN_MASTER_PORT + 1 + idx))"
    echo "=== $(date -Iseconds) :: window21 eval transfer ${idx} ==="
    metric_path="$(run_window_inference "${run_config}")"
    record_result "window_transfer_${idx}" "window21_eval" "${run_config}" "${metric_path}"
    if meets_target "${metric_path}"; then
      echo "=== target reached by window transfer ${idx} ==="
      target_reached=1
      break
    fi
  done
fi

joined_configs="${completed_configs[*]}"
final_args=(--label "transfer_min93_queue")
if [ "${REBUILD_NOTIFY_WANDB}" = "1" ]; then
  final_args+=(--wandb-alert)
fi
REBUILD_CHECK_CONFIGS="${joined_configs}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
python3 scripts/post_rebuild_training.py "${final_args[@]}"

echo "=== $(date -Iseconds) :: transfer min93 queue done ==="
