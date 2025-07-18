import os
import torch.distributed as dist

def ddp_setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")
    return rank, world_size, local_rank

def ddp_cleanup():
    dist.destroy_process_group()