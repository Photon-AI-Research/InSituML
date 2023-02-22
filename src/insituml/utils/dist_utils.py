import os

import torch


def has_master_addr():
    return bool(os.getenv('MASTER_ADDR'))


def is_distributed():
    return torch.distributed.is_available() and has_master_addr()


def get_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


def get_rank():
    if is_distributed():
        return torch.distributed.get_rank()
    else:
        return 0


def get_local_rank():
    if is_distributed():
        return int(os.environ['LOCAL_RANK'])
    else:
        return 0


def is_rank_0():
    return get_rank() == 0


def print0(*args, **kwargs):
    if is_rank_0():
        print(*args, **kwargs)


def maybe_initialize():
    if has_master_addr():
        assert torch.distributed.is_available()
        if torch.cuda.is_available():
            dist_backend = 'nccl'
            torch.cuda.set_device(get_local_rank())
        else:
            dist_backend = 'gloo'
        torch.distributed.init_process_group(dist_backend)


def barrier():
    if is_distributed():
        torch.distributed.barrier()


def all_reduce_avg(value):
    if not is_distributed():
        value_reduced = value.clone()
    # ReduceOp.AVG only supported in NCCL.
    elif torch.distributed.get_backend() == 'nccl':
        value_reduced = value.clone()
        torch.distributed.all_reduce(
            value_reduced, torch.distributed.ReduceOp.AVG)
    else:
        value_reduced = value / get_world_size()
        torch.distributed.all_reduce(
            value_reduced, torch.distributed.ReduceOp.SUM)
    return value_reduced


def all_reduce(value, op=torch.distributed.ReduceOp.SUM):
    value_reduced = value.clone()
    if is_distributed():
        torch.distributed.all_reduce(value_reduced)
    return value_reduced


def get_ddp_devices():
    assert is_distributed(), \
        'querying DDP devices not allowed if not running distributed'
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        device_ids = [local_rank]
        output_device = local_rank
    else:
        device_ids = None
        output_device = None
    return device_ids, output_device


def maybe_ddp_wrap(model):
    if is_distributed():
        device_ids, output_device = get_ddp_devices()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
        )
    return model


def create_distributed_sampler(
        data_set,
        epoch,
        shuffle=True,
        seed=0,
        drop_last=False,
):
    sampler = torch.utils.data.DistributedSampler(
        data_set,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    sampler.set_epoch(epoch)
    return sampler
