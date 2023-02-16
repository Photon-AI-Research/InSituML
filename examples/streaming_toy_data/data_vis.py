#!/usr/bin/env python3

import importlib
import sys

import numpy as np
import torch as T
from matplotlib import pyplot as plt

from insituml.toy_data import generate

# When used as %run -i this.py in ipython
importlib.reload(generate)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        assert len(sys.argv) == 2
        method = sys.argv[1]
    else:
        method = "tdds"

    with T.no_grad():
        nsteps = 5
        npoints = 1024
        X, Y = generate.generate_toy8(
            label_kind="all", npoints=npoints, scale=0.2**2
        )

        if method == "tdds":
            ds = generate.TimeDependentTensorDataset(
                X,
                Y,
                time_x_func=lambda x, t: T.tensor(
                    [x[0] + t, x[1] + T.sin(T.tensor(t)) + 0.3 * t]
                ),
                dt=6.25,
            )

            ps, ls = generate.tdds_arrays(ds, batch_size=npoints, nsteps=nsteps)

        elif method == "td":
            ps, ls = generate.td_arrays(
                X,
                Y,
                time_func_mode="abs",
                time=T.linspace(0, 25, nsteps),
                time_x_func=lambda x, t: T.stack(
                    [x[:, 0] + t, x[:, 1] + T.sin(t) + 0.3 * t]
                ).T,
            )

        else:
            raise ValueError(f"Illegal {method=}")

        ps = ps.numpy().reshape((-1, ps.shape[-1]))
        ls = ls.numpy().reshape((-1, ls.shape[-1]))


        fig, axs = plt.subplots(ncols=2)

        nz_i, nz_j = np.nonzero(ls)
        nz_val = ls[(nz_i, nz_j)]
        color = nz_j + nz_val

        axs[0].scatter(ps[:, 0], ps[:, 1], c=color, s=0.5)
        axs[0].set_aspect("equal")
        axs[1].matshow(ls, aspect="auto")

        plt.show()
