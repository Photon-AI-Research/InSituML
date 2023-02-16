#!/usr/bin/env python3

import importlib

from icecream import ic
from torch.utils.data import DataLoader

from insituml.toy_data import generate

# When used as %run -i this.py in ipython
importlib.reload(generate)


if __name__ == "__main__":

    ds = generate.TimeDependentTensorDataset(
        ##*generate.generate_toy8(label_kind="all", npoints=1024),
        *generate.generate_fake_toy(npoints=6),
        dt=0.3,
    )

    train_dl = DataLoader(ds, batch_size=2, shuffle=True)

    # PIC time step
    for i_time in range(2):
        # train some epochs on this time step's data
        for i_epoch in range(2):
            for i_batch, (X_batch, Y_batch) in enumerate(train_dl):
                ic(i_time, i_epoch, i_batch, ds.time, X_batch, Y_batch)
        ds.step()
