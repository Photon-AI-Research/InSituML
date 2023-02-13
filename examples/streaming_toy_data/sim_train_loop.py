#!/usr/bin/env python3

import importlib

from icecream import ic
from torch.utils.data import DataLoader

import data_gen

# When used as %run -i this.py in ipython
importlib.reload(data_gen)

if __name__ == "__main__":
    ds = data_gen.ToyIterDataset(dt=0.3)
    dl = DataLoader(ds, batch_size=2)

    for idx, (x, y) in enumerate(dl):
        ic(idx, ds.time, x, y)

        if idx % 2:
            ds.step()

        if idx == 10:
            break
