# copied from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
from mpi4py import MPI
import os
comm = MPI.COMM_WORLD
rank = comm.rank
world_size = comm.size
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

# this doesn't work, needs a wrapper script to work

import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP



# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'gv028'
    os.environ['MASTER_PORT'] = '12340'

    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    print("1.0 rank", rank)
    
    # create model and move it to GPU with id rank
    model = ToyModel().to(0)
    print("1.1 rank", rank)
    ddp_model = DDP(model, device_ids=[0])
    print("1.2 rank", rank)
    loss_fn = nn.MSELoss()
    print("1.3 rank", rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print("1.4 rank", rank)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(0)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    
    print("run with", n_gpus, "GPUs and WORLD SIZE", world_size)
    print("torch version", torch.__version__)
    
    
    demo_basic(comm.rank, world_size)
    #run_demo(demo_checkpoint, world_size)
    world_size = n_gpus//2
    
# mpirun -n 4 python tools/openpmd-streaming-continual-learning.py --io_config=share/configs/io_config_hemera.py --model_config=share/configs/model_config.py --type_streamer=offline
# mpirun python ddp_torch2_toy.py