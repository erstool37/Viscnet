export CUBLAS_WORKSPACE_CONFIG=:4096:8

wandb login e4fc630ae5f28ea9dc5453b32b848681d664e9a1
# python3 src/utils/preprocess.py -c configs/configTransEmbed.yaml -m real
python3 src/utils/preprocess_real.py -c configs/configTransEmbed.yaml -m train # both normalization is needed
python3 src/utils/preprocess_real.py -c configs/configTransEmbed.yaml -m real
export OMP_NUM_THREADS=1
# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 src/main_rpm.py -c configs/configTrans.yaml
# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 src/main_embed.py -c configs/configTransEmbed.yaml
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 src/main_dummy.py -c configs/configTransEmbed.yaml

# how to use this set to fine tune with
# 1. use this dev.sh
# 2. split the real world impeller 1000 dataset into train and test(with and without render A, F), use copy.py file i gave u
# 3. use main_dummy.py, go to the configTransEmbed.yaml and set the data_root as the realworld train dataset, and set real_root as the test dataset.

# nproc_per_node=2 is the number of GPUs per node
# nnodes=1 is the number of nodes on this machine
# node_rank=0 is the rank of this node (0 for the first node)

# python3 src/main_rpm.py -c configs/config.yaml
# python3 src/utils/preprocess.py -c configs/config.yaml -m real
# python3 src/main.py -c configs/config.yaml