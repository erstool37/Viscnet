export CUBLAS_WORKSPACE_CONFIG=:4096:8
wandb login e4fc630ae5f28ea9dc5453b32b848681d664e9a1
# 270cceb2081f0f17845e654a8e70d0f052c924d8

### Normalizing dataset
# python3 src/utils/preprocess.py -c configs/config.yaml -m synthetic
# python3 src/utils/preprocess.py -c configs/config.yaml -m real # Considers the "test_root" dataset to be real data
# python3 src/utils/preprocess_ODDEVEN.py bash -c configs/config.yaml

### Training
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config2.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config3.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config4.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config5.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --master_port=29513 --node_rank=0 src/main.py -c configs/config6.yaml