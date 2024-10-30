import torch.distributed as dist
from torch.distributed._tensor import init_device_mesh

mesh = None

def init_global_device_mesh():
    global mesh
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

def is_initialized():
    global mesh
    return mesh is not None
