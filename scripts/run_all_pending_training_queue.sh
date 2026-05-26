#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export WANDB_PROJECT="${ALL_QUEUE_WANDB_PROJECT:-re-rebuild-viscnet}"
export WANDB_ENTITY="${ALL_QUEUE_WANDB_ENTITY:-jongwonsohn-seoul-national-university}"

REQUIRE_WANDB="${REQUIRE_WANDB:-1}"
QUEUE_ROOT="${QUEUE_ROOT:-outputs/rebuild_reproduction/all_pending_training_queue}"
CONFIG_ROOT="${CONFIG_ROOT:-${QUEUE_ROOT}/configs}"
PHASE_DATASET="${PHASE_DATASET:-outputs/rebuild_reproduction/derived_datasets/real_train_993_window10_stride5_phase5}"
RAW_FPS_ROOT="${RAW_FPS_ROOT:-outputs/rebuild_reproduction/derived_datasets/raw_fps_benchmark}"
SUMMARY_JSON="${SUMMARY_JSON:-${QUEUE_ROOT}/summary.json}"
START_PORT="${START_PORT:-29701}"
RUN_PHASE_OFFSET="${RUN_PHASE_OFFSET:-1}"
RUN_RAW_FPS="${RUN_RAW_FPS:-1}"
RUN_RAW_FPS_336="${RUN_RAW_FPS_336:-1}"
REBUILD_NOTIFY_WANDB="${REBUILD_NOTIFY_WANDB:-0}"
SKIP_EXISTING_CHECKPOINTS="${SKIP_EXISTING_CHECKPOINTS:-1}"
RUN_EVAL_AFTER_TRAIN="${RUN_EVAL_AFTER_TRAIN:-1}"
EVAL_SKIPPED_CHECKPOINTS="${EVAL_SKIPPED_CHECKPOINTS:-0}"

if [ "${REQUIRE_WANDB}" = "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Refusing to launch the full pending queue." >&2
  exit 1
fi

mkdir -p "${CONFIG_ROOT}" "${QUEUE_ROOT}" outputs/rebuild_reproduction/logs outputs/rebuild_reproduction/session_markers

signal_done() {
  local status="$?"
  python3 - "${SUMMARY_JSON}" "${status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
status = int(sys.argv[2])
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
else:
    summary = {"runs": []}
summary["finished_at"] = datetime.now(timezone.utc).isoformat()
summary["exit_status"] = status
if status == 0:
    summary.pop("active_run", None)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
Path("outputs/rebuild_reproduction/session_markers/all_pending_training_queue.done").write_text(
    f"exit_status={status}\nfinished_at={summary['finished_at']}\nsummary={summary_path}\n"
)
PY
  if [ -n "${ALL_QUEUE_DONE_SIGNAL:-}" ]; then
    tmux wait-for -S "${ALL_QUEUE_DONE_SIGNAL}" || true
  fi
}
trap signal_done EXIT

python3 - "${SUMMARY_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    summary = json.loads(path.read_text())
    summary.setdefault("runs", [])
    summary["resumed_at"] = datetime.now(timezone.utc).isoformat()
    summary.pop("exit_status", None)
    summary.pop("finished_at", None)
else:
    summary = {
        "label": "all_pending_training_queue",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
path.write_text(json.dumps(summary, indent=2) + "\n")
PY

prepare_phase_configs() {
  python3 - "${CONFIG_ROOT}" "${PHASE_DATASET}" <<'PY'
from pathlib import Path
import sys
import os
import yaml

config_root = Path(sys.argv[1])
phase_dataset = sys.argv[2]
config_root.mkdir(parents=True, exist_ok=True)


def load(path):
    return yaml.safe_load(Path(path).read_text())


def dump(name, cfg):
    path = config_root / f"{name}.yaml"
    cfg["project"] = os.environ.get("WANDB_PROJECT", cfg.get("project"))
    cfg["entity"] = os.environ.get("WANDB_ENTITY", cfg.get("entity"))
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(path)
    return path


def set_10_frame_common(cfg):
    cfg["dataset"]["train"]["frame_num"] = 2
    cfg["dataset"]["train"]["time"] = 5
    cfg["dataset"]["test"]["frame_num"] = 2
    cfg["dataset"]["test"]["time"] = 5
    cfg["model"]["transformer"]["num_frames"] = 10
    cfg["train_settings"]["test_bool"] = False
    cfg["train_settings"]["val_test_bool"] = True
    return cfg


def set_phase_train(cfg):
    cfg["dataset"]["train"]["train_root"] = phase_dataset
    cfg["dataset"]["train"]["manifest"] = f"{phase_dataset}/manifest.json"
    cfg["dataset"]["train"]["use_all_samples"] = True
    return cfg


def set_window10_inference(cfg, run_name):
    cfg["inference"] = {
        "temporal_window": {
            "enabled": True,
            "average_logits": True,
            "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
            "test_root": "dataset/RealArchive/test_1000_wo_pat2",
            "baseline_metrics": f"outputs/rebuild_reproduction/{run_name}/confusion_matrix/{run_name}_metrics.json",
            "full_frames": 50,
            "window_size": 10,
            "num_windows": 41,
            "window_batch_size": 32,
            "output_dir": f"outputs/rebuild_reproduction/{run_name}/window41_test_inference",
        }
    }
    return cfg


synth = load("configs/rebuild/retries/synthetic_pretrain_window30_batch8_ep50.yaml")
synth = set_10_frame_common(synth)
synth["name"] = "repro_synthetic_pretrain_window10_stride5_phase5_ep50"
synth["train_settings"]["test_bool"] = False
synth["train_settings"]["val_test_bool"] = False
synth["training"]["checkpoint_name"] = "repro_synthetic_pretrain_window10_stride5_phase5_ep50.pth"
synth["training"]["curr_ckpt"] = synth["training"]["checkpoint_name"]
synth["misc_dir"]["output_root"] = "outputs/rebuild_reproduction/repro_synthetic_pretrain_window10_stride5_phase5_ep50"
dump(synth["name"], synth)

real = load("configs/rebuild/retries/realonly_993_window30x21_ep50.yaml")
real = set_10_frame_common(real)
real = set_phase_train(real)
real["name"] = "repro_realonly_993_window10_stride5_phase5_ep50"
real["training"]["curr_bool"] = False
real["training"]["checkpoint_name"] = "repro_realonly_993_window10_stride5_phase5_ep50.pth"
real["misc_dir"]["output_root"] = "outputs/rebuild_reproduction/repro_realonly_993_window10_stride5_phase5_ep50"
real["training"]["acceptance"]["note"] = "real-only 993 diagnostic using 10-frame clips sampled every 5 source frames across all 5 phase offsets"
real = set_window10_inference(real, real["name"])
dump(real["name"], real)

transfer_templates = [
    ("configs/rebuild/retries/transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.yaml", "lr1e5_ep45"),
    ("configs/rebuild/retries/transfer_993_window30x21_from_synth30_lr5e6_ep55_min93.yaml", "lr5e6_ep55"),
]
for template, suffix in transfer_templates:
    cfg = load(template)
    cfg = set_10_frame_common(cfg)
    cfg = set_phase_train(cfg)
    cfg["name"] = f"repro_transfer_993_window10_stride5_phase5_from_synth10_{suffix}"
    cfg["training"]["curr_bool"] = True
    cfg["training"]["curr_ckpt"] = "repro_synthetic_pretrain_window10_stride5_phase5_ep50.pth"
    cfg["training"]["checkpoint_name"] = f"{cfg['name']}.pth"
    cfg["misc_dir"]["output_root"] = f"outputs/rebuild_reproduction/{cfg['name']}"
    cfg["training"]["acceptance"]["note"] = "transfer diagnostic initialized from matching 10-frame synthetic pretrain and trained on stride-5 phase-offset real clips"
    cfg = set_window10_inference(cfg, cfg["name"])
    dump(cfg["name"], cfg)
PY
}

run_train_config() {
  local config="$1"
  local port="$2"
  local run_name
  LAST_RUN_STATUS="unknown"
  run_name="$(python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["checkpoint_name"]).stem)
PY
)"
  local checkpoint="outputs/rebuild_reproduction/checkpoints/${run_name}.pth"
  if [ "${SKIP_EXISTING_CHECKPOINTS}" = "1" ] && [ -f "${checkpoint}" ]; then
    echo "=== $(date -Iseconds) :: skip existing checkpoint ${checkpoint} ==="
    record_run "${run_name}" "${config}" "skipped_existing_checkpoint"
    LAST_RUN_STATUS="skipped_existing_checkpoint"
    return 0
  fi
  local is_rawfps="0"
  local is_336="0"
  if [[ "${run_name}" == rawfps_* ]]; then
    is_rawfps="1"
  fi
  if [[ "${run_name}" == *336px ]]; then
    is_336="1"
  fi

  local attempts
  if [ "${is_rawfps}" = "1" ]; then
    if [ "${is_336}" = "1" ]; then
      attempts="${RAW_FPS_BATCH_CANDIDATES_336:-16 12 8 4 2}"
    else
      attempts="${RAW_FPS_BATCH_CANDIDATES_224:-32 24 16}"
    fi
  else
    attempts="default"
  fi

  local attempt batch train_config status log_path
  for attempt in ${attempts}; do
    if [ "${attempt}" = "default" ]; then
      train_config="$(make_train_only_config "${config}")"
      batch="default"
    elif [ "${is_336}" = "1" ]; then
      export RAW_FPS_BATCH_336="${attempt}"
      train_config="$(make_train_only_config "${config}")"
      unset RAW_FPS_BATCH_336
      batch="${attempt}"
    else
      export RAW_FPS_BATCH_224="${attempt}"
      train_config="$(make_train_only_config "${config}")"
      unset RAW_FPS_BATCH_224
      batch="${attempt}"
    fi

    echo "=== $(date -Iseconds) :: train ${train_config} batch=${batch} ==="
    record_run "${run_name}" "${config}" "running_batch${batch}"
    set +e
    REBUILD_CONFIGS="${train_config}" \
    REBUILD_INCLUDE_RETRIES=1 \
    REBUILD_FINAL_ANALYSIS=0 \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    MASTER_PORT="${port}" \
    bash scripts/run_rebuild_reproduction.sh
    status="$?"
    set -e
    if [ "${status}" = "0" ]; then
      record_run "${run_name}" "${config}" "trained_batch${batch}"
      LAST_RUN_STATUS="trained"
      return 0
    fi

    log_path="outputs/rebuild_reproduction/logs/${run_name}.log"
    if [ "${is_rawfps}" = "1" ] && [ -f "${log_path}" ] && grep -Eiq "out of memory|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|DefaultCPUAllocator" "${log_path}"; then
      echo "=== $(date -Iseconds) :: batch=${batch} failed with allocation error; trying next batch if available ==="
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

make_train_only_config() {
  local config="$1"
  local train_config="${CONFIG_ROOT}/train_$(basename "${config}")"
  python3 - "${config}" "${train_config}" <<'PY'
from pathlib import Path
import sys
import os
import yaml

src, dst = sys.argv[1:3]
cfg = yaml.safe_load(Path(src).read_text())
cfg["project"] = os.environ.get("WANDB_PROJECT", cfg.get("project"))
cfg["entity"] = os.environ.get("WANDB_ENTITY", cfg.get("entity"))
if str(cfg.get("name", "")).startswith("rawfps_"):
    is_336 = "336px" in str(cfg.get("name", ""))
    train_batch = int(os.environ.get("RAW_FPS_BATCH_336" if is_336 else "RAW_FPS_BATCH_224", "4" if is_336 else "32"))
    eval_batch = int(os.environ.get("RAW_FPS_EVAL_BATCH_336" if is_336 else "RAW_FPS_EVAL_BATCH_224", "8" if is_336 else "64"))
    cfg["dataset"]["train"]["dataloader"]["batch_size"] = train_batch
    cfg["dataset"]["test"]["dataloader"]["batch_size"] = eval_batch
cfg["train_settings"]["train_bool"] = True
cfg["train_settings"]["test_bool"] = False
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${train_config}"
}

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
entry = {
    "run_name": run_name,
    "config": config,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint": f"outputs/rebuild_reproduction/checkpoints/{run_name}.pth",
}
summary["runs"] = [row for row in summary.get("runs", []) if row.get("run_name") != run_name]
summary["runs"].append(entry)
if status == "running":
    summary["active_run"] = entry
elif summary.get("active_run", {}).get("run_name") == run_name:
    summary.pop("active_run", None)
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
PY
}

make_eval_config() {
  local config="$1"
  local eval_config="${CONFIG_ROOT}/eval_$(basename "${config}")"
  python3 - "${config}" "${eval_config}" <<'PY'
from pathlib import Path
import sys
import os
import yaml

src, dst = sys.argv[1:3]
cfg = yaml.safe_load(Path(src).read_text())
cfg["project"] = os.environ.get("WANDB_PROJECT", cfg.get("project"))
cfg["entity"] = os.environ.get("WANDB_ENTITY", cfg.get("entity"))
if str(cfg.get("name", "")).startswith("rawfps_"):
    is_336 = "336px" in str(cfg.get("name", ""))
    train_batch = int(os.environ.get("RAW_FPS_BATCH_336" if is_336 else "RAW_FPS_BATCH_224", "4" if is_336 else "32"))
    eval_batch = int(os.environ.get("RAW_FPS_EVAL_BATCH_336" if is_336 else "RAW_FPS_EVAL_BATCH_224", "8" if is_336 else "64"))
    cfg["dataset"]["train"]["dataloader"]["batch_size"] = train_batch
    cfg["dataset"]["test"]["dataloader"]["batch_size"] = eval_batch
cfg["train_settings"]["train_bool"] = False
cfg["train_settings"]["test_bool"] = True
cfg["train_settings"]["val_test_bool"] = False
cfg["training"]["curr_bool"] = True
cfg["training"]["curr_ckpt"] = cfg["training"]["checkpoint_name"]
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${eval_config}"
}

run_single_rank_eval_if_needed() {
  local config="$1"
  local port="$2"
  if [ "${RUN_EVAL_AFTER_TRAIN}" != "1" ]; then
    return 0
  fi
  if python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
raise SystemExit(0 if cfg.get("inference", {}).get("temporal_window", {}).get("enabled") else 1)
PY
  then
    return 0
  fi
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
  echo "=== $(date -Iseconds) :: single-rank eval ${config} ==="
  PYTHONPATH=src torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_port="${port}" \
    --node_rank=0 \
    src/main.py -c "${eval_config}" \
    > "outputs/rebuild_reproduction/logs/eval_${run_name}.log" 2>&1
}

run_window_inference_if_configured() {
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
  if [[ "${run_name}" == *synthetic_pretrain* ]]; then
    return 0
  fi
  if python3 - "${config}" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
raise SystemExit(0 if cfg.get("inference", {}).get("temporal_window", {}).get("enabled") else 1)
PY
  then
    echo "=== $(date -Iseconds) :: window inference ${config} ==="
    PYTHONPATH=src torchrun \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${port}" \
      scripts/window21_test_inference.py \
      --config "${config}" \
      > "outputs/rebuild_reproduction/logs/window_${run_name}.log" 2>&1
  fi
}

queue_configs=()
port="${START_PORT}"

if [ "${RUN_PHASE_OFFSET}" = "1" ]; then
  echo "=== $(date -Iseconds) :: build phase-offset dataset ==="
  if [ ! -f "${PHASE_DATASET}/manifest.json" ]; then
    python3 scripts/build_real_window_dataset.py \
      --output-root "${PHASE_DATASET}" \
      --window-size 10 \
      --windows-per-video 1 \
      --sample-stride 5 \
      --phase-offsets all \
      --force
  else
    echo "=== $(date -Iseconds) :: phase-offset dataset already exists ==="
  fi
  prepare_phase_configs
  phase_configs=(
    "${CONFIG_ROOT}/repro_synthetic_pretrain_window10_stride5_phase5_ep50.yaml"
    "${CONFIG_ROOT}/repro_realonly_993_window10_stride5_phase5_ep50.yaml"
    "${CONFIG_ROOT}/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45.yaml"
    "${CONFIG_ROOT}/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55.yaml"
  )
  for config in "${phase_configs[@]}"; do
    queue_configs+=("${config}")
    run_train_config "${config}" "${port}"
    port=$((port + 1))
    if [ "${LAST_RUN_STATUS}" != "skipped_existing_checkpoint" ] || [ "${EVAL_SKIPPED_CHECKPOINTS}" = "1" ]; then
      run_single_rank_eval_if_needed "${config}" "${port}"
      port=$((port + 1))
      run_window_inference_if_configured "${config}" "${port}"
      port=$((port + 1))
    fi
  done
fi

if [ "${RUN_RAW_FPS}" = "1" ]; then
  echo "=== $(date -Iseconds) :: materialize raw-FPS 224 dataset ==="
  if [ ! -f "${RAW_FPS_ROOT}/build_report_image224.json" ]; then
    python3 scripts/build_raw_fps_window_dataset.py \
      --materialize-clips \
      --write-configs \
      --image-size 224 \
      --output-root "${RAW_FPS_ROOT}"
  else
    echo "=== $(date -Iseconds) :: raw-FPS 224 dataset already exists ==="
  fi
  raw224_configs=(
    configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_224px.yaml
    configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_nativefps_224px.yaml
    configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_legacy10fps_224px.yaml
    configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_nativefps_224px.yaml
  )
  for config in "${raw224_configs[@]}"; do
    queue_configs+=("${config}")
    run_train_config "${config}" "${port}"
    port=$((port + 1))
    if [ "${LAST_RUN_STATUS}" != "skipped_existing_checkpoint" ] || [ "${EVAL_SKIPPED_CHECKPOINTS}" = "1" ]; then
      run_single_rank_eval_if_needed "${config}" "${port}"
      port=$((port + 1))
    fi
  done
fi

if [ "${RUN_RAW_FPS_336}" = "1" ]; then
  echo "=== $(date -Iseconds) :: materialize raw-FPS 336 dataset ==="
  if [ ! -f "${RAW_FPS_ROOT}/build_report_image336.json" ]; then
    python3 scripts/build_raw_fps_window_dataset.py \
      --materialize-clips \
      --write-configs \
      --image-size 336 \
      --output-root "${RAW_FPS_ROOT}"
  else
    echo "=== $(date -Iseconds) :: raw-FPS 336 dataset already exists ==="
  fi
  raw336_configs=(
    configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_336px.yaml
    configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_nativefps_336px.yaml
    configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_legacy10fps_336px.yaml
    configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_nativefps_336px.yaml
  )
  for config in "${raw336_configs[@]}"; do
    queue_configs+=("${config}")
    run_train_config "${config}" "${port}"
    port=$((port + 1))
    if [ "${LAST_RUN_STATUS}" != "skipped_existing_checkpoint" ] || [ "${EVAL_SKIPPED_CHECKPOINTS}" = "1" ]; then
      run_single_rank_eval_if_needed "${config}" "${port}"
      port=$((port + 1))
    fi
  done
fi

joined_configs="${queue_configs[*]}"
final_args=(--label "all_pending_training_queue")
if [ "${REBUILD_NOTIFY_WANDB}" = "1" ]; then
  final_args+=(--wandb-alert)
fi
REBUILD_CHECK_CONFIGS="${joined_configs}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
python3 scripts/post_rebuild_training.py "${final_args[@]}"

echo "=== $(date -Iseconds) :: all pending training queue done ==="
