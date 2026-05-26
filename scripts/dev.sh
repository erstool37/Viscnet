#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set; wandb.init may run unauthenticated or offline depending on W&B settings." >&2
fi

### Normalizing dataset
# python3 src/utils/preprocess.py -c configs/config.yaml -m synthetic
# python3 src/utils/preprocess.py -c configs/config.yaml -m real # Considers the "test_root" dataset to be real data
# python3 src/utils/preprocess_ODDEVEN.py bash -c configs/config.yaml

### Training
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config2.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config3.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config4.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config5.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config6.yaml
