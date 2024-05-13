import os
import torch


def has_master_addr():
    '''
    Checks if the MASTER_ADDR environment variable is set.

    Returns:
        bool: True if MASTER_ADDR is set, False otherwise.
    '''
    return bool(os.getenv('MASTER_ADDR'))


def is_distributed():
    '''
    Checks if distributed training is enabled.

    Returns:
        bool: True if distributed training is enabled, False otherwise.
    '''
    return torch.distributed.is_available() and has_master_addr()


def get_world_size():
    '''
    Gets the total number of processes in the distributed group.

    Returns:
        int: The total number of processes.
    '''
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


def get_rank():
    '''
    Gets the rank of the current process in the distributed group.

    Returns:
        int: The rank of the current process.
    '''
    if is_distributed():
        return torch.distributed.get_rank()
    else:
        return 0


def get_local_rank():
    '''
    Gets the local rank of the current process.

    Returns:
        int: The local rank of the current process.
    '''
    if is_distributed():
        return int(os.environ['LOCAL_RANK'])
    else:
        return 0


def is_rank_0():
    '''
    Checks if the current process has rank 0.

    Returns:
        bool: True if the current process has rank 0, False otherwise.
    '''
    return get_rank() == 0


def print0(*args, **kwargs):
    '''
    Prints the provided arguments only from the process with rank 0.

    Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    '''
    if is_rank_0():
        print(*args, **kwargs)


def maybe_initialize():
    '''
    Initializes the distributed process group if MASTER_ADDR is set.
    '''
    if has_master_addr():
        assert torch.distributed.is_available()
        if torch.cuda.is_available():
            dist_backend = 'nccl'
            torch.cuda.set_device(get_local_rank())
        else:
            dist_backend = 'gloo'
        torch.distributed.init_process_group(dist_backend)


def barrier():
    '''
    Synchronizes all processes in the distributed group.
    '''
    if is_distributed():
        torch.distributed.barrier()


def all_reduce_avg(value):
    '''
    Performs an average all-reduce operation across all processes.

    Parameters:
        value (torch.Tensor): The tensor to be reduced.

    Returns:
        torch.Tensor: The reduced tensor.
    '''
    if not is_distributed():
        value_reduced = value.clone()
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
    '''
    Performs an all-reduce operation across all processes.

    Parameters:
        value (torch.Tensor): The tensor to be reduced.
        op (torch.distributed.ReduceOp): The reduction operation (default: SUM).

    Returns:
        torch.Tensor: The reduced tensor.
    '''
    value_reduced = value.clone()
    if is_distributed():
        torch.distributed.all_reduce(value_reduced)
    return value_reduced


def get_ddp_devices():
    '''
    Gets the devices for DistributedDataParallel (DDP) training.

    Returns:
        tuple: A tuple containing the device IDs and the output device.
    '''
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
    '''
    Wraps a model with DistributedDataParallel (DDP) if distributed training is enabled.

    Parameters:
        model (torch.nn.Module): The model to be wrapped.

    Returns:
        torch.nn.parallel.DistributedDataParallel: The DDP-wrapped model.
    '''
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
    '''
    Creates a distributed sampler for the dataset.

    Parameters:
        data_set (torch.utils.data.Dataset): The dataset.
        epoch (int): Current epoch.
        shuffle (bool): Whether to shuffle the data (default: True).
        seed (int): Random seed (default: 0).
        drop_last (bool): Whether to drop the last incomplete batch (default: False).

    Returns:
        torch.utils.data.DistributedSampler: The distributed sampler.
    '''
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
