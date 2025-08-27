export CUBLAS_WORKSPACE_CONFIG=:4096:8
wandb login 270cceb2081f0f17845e654a8e70d0f052c924d8

## Normalizing dataset
# python3 src/utils/preprocess.py -c configs/config.yaml -m synthetic
# python3 src/utils/preprocess.py -c configs/config.yaml -m real

## Training
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=1 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config.yaml
torchrun --nproc_per_node=1 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config2.yaml