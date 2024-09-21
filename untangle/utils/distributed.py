"""Distributed training utils."""

import os

import torch


def is_distributed_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1

    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist_backend = "nccl"
    dist_url = "env://"

    if is_distributed_env():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars if needed
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["RANK"] = str(global_rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
            )
        else:
            # DDP via torchrun
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
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
