#!/usr/bin/env python3

import importlib

from icecream import ic
from torch.utils.data import DataLoader

from insituml.toy_data import generate

# When used as %run -i this.py in ipython
importlib.reload(generate)


def header(msg):
    bar = "=" * 78
    print(f"{bar}\n{msg}\n{bar}")


if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Infinite stream of data. Call step() whenever, even within an "epoch",
    # where length of epoch is X.shape[0].
    # ------------------------------------------------------------------------

    header("cycle=True")

    ds = generate.TimeDependentDataset(dt=0.3, cycle=True)
    dl = DataLoader(ds, batch_size=2)

    for i_batch, (x, y) in enumerate(dl):
        ic(i_batch, ds.time, x, y)

        if i_batch % 2:
            ds.step()

        if i_batch == 10:
            break

    # ------------------------------------------------------------------------
    # Create epochs.
    # ------------------------------------------------------------------------

    header("cycle=False")

    ds = generate.TimeDependentDataset(dt=0.3, cycle=False)
    dl = DataLoader(ds, batch_size=2)

    for i_epoch in range(5):
        for i_batch, (x, y) in enumerate(dl):
            ic(i_epoch, i_batch, ds.time, x, y)
        ds.step()
