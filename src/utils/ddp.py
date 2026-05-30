import os

import numpy as np
import torch
import torch.distributed as dist


def ddp_setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # initialize the process group
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")
    return rank, world_size, local_rank


def ddp_cleanup():
    dist.destroy_process_group()


def broadcast_object_list_for_device(object_list, src=0, device=None):
    if dist.get_world_size() > 1:
        dist.broadcast_object_list(object_list, src=src, device=device)
    return object_list


def gather_lists(arr_list):
    bucket = [None] * dist.get_world_size()
    dist.all_gather_object(bucket, arr_list)
    merged = []
    for part in bucket:
        merged.extend(part)
    return np.concatenate(merged, axis=0)
