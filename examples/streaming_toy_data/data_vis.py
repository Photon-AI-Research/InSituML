#!/usr/bin/env python3

import importlib

import numpy as np
import torch as T
from matplotlib import pyplot as plt

import data_gen

# When used as %run -i this.py in ipython
importlib.reload(data_gen)

ps, ls = data_gen.generate_td_array(
    pos_lab_func=lambda: data_gen.generate_toy8(
        label_kind="all",
        npoints=1024,
        scale=0.2**2,
    ),
    time_func_mode="abs",
    time=T.linspace(0, 25, 5),
    time_pos_func=lambda x, t: T.stack(
        [x[:, 0] + t, x[:, 1] + T.sin(t) + 0.3 * t]
    ).T,
)

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
