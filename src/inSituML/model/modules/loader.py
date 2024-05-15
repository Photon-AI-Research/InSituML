import torch
from torch.utils.data import DataLoader

from . import dist_utils


def get_loader(dataset, batch_size=1):
    if len(dataset) == 0:
        raise ValueError("Loader is empty")
    sampler = dist_utils.create_distributed_sampler(dataset, 0, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print("Total number of points: ", dataset.__len__())
    print("Size of loader: ", len(loader))
    return loader
