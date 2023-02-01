#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

import data_gen

ps, ls = data_gen.generate_td_array(
    pos_lab_func=lambda: data_gen.generate_toy8(label_kind="all", npoints=1024),
    ntime=20,
    dt=4.0,
)

ps = ps.reshape((-1, ps.shape[-1]))
ls = ls.reshape((-1, ls.shape[-1]))

fig,axs = plt.subplots(ncols=2)

nz_i, nz_j = np.nonzero(ls)
nz_val = ls[(nz_i, nz_j)]
color = nz_j + nz_val

axs[0].scatter(ps[:, 0], ps[:, 1], c=color, s=0.5)
axs[0].set_aspect("equal")
axs[1].matshow(ls, aspect="auto")

plt.show()
