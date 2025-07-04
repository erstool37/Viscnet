export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 src/utils/preprocess.py -c configs/config.yaml -m train
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 src/main_rpm.py -c configs/config.yaml 

# nproc_per_node=2 is the number of GPUs per node
# nnodes=1 is the number of nodes on this machine
# node_rank=0 is the rank of this node (0 for the first node)

# python3 src/main_rpm.py -c configs/config.yaml
# python3 src/utils/preprocess.py -c configs/config.yaml -m real
# python3 src/main.py -c configs/config.yaml