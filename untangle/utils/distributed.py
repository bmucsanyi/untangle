"""Distributed training utils."""

import os

import torch


def is_distributed_env():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_distributed_env():
        local_rank = os.environ["LOCAL_RANK"]
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        distributed = True

    if distributed and device != "cpu":
        device = f"{device}:{local_rank}"

    if device.startswith("cuda:"):
        torch.cuda.set_device(device)

    args.device = device
    args.world_size = world_size
    args.rank = global_rank
    args.local_rank = local_rank
    args.distributed = distributed

    device = torch.device(device)

    if args.storage_device == "cuda" and not torch.cuda.is_available():
        msg = "Storage device 'cuda' requested but cuda is not available."
        raise ValueError(msg)

    storage_device = device if args.storage_device == "cuda" else torch.device("cpu")

    return device, storage_device


def distribute_bn(model, world_size):
    # Ensure every node has the same running bn stats
    for bn_name, bn_buf in model.module.named_buffers(recurse=True):
        if ("running_mean" in bn_name) or ("running_var" in bn_name):
            # Average bn stats across whole group
            torch.distributed.all_reduce(bn_buf, op=torch.distributed.ReduceOp.SUM)
            bn_buf /= float(world_size)


def reduce_tensor(tensor, world_size):
    reduced_tensor = tensor.clone()
    torch.distributed.all_reduce(reduced_tensor, op=torch.distributed.ReduceOp.SUM)
    reduced_tensor /= world_size
    return reduced_tensor
