#!/usr/bin/env python3

import importlib
from typing import Sequence, Optional
from multiprocessing import cpu_count

from icecream import ic

from torch.utils.data import DataLoader
import torch as T
from torch import nn

import numpy as np

from matplotlib import pyplot as plt

from FrEIA.framework import (
    InputNode,
    OutputNode,
    Node,
    ReversibleGraphNet,
    ConditionNode,
)
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

from nmbx.convergence import Convergence

from insituml.toy_data import generate

# When used as %run -i this.py in ipython
importlib.reload(generate)

# Adapted from Nico_toy8_examples/train_cINN_distributed_toy8.py
def build_model(ndim_x=2, ndim_y=8, ncc=2, nh=512, nl=1):
    def subnet_fc(c_in, c_out):
        layers = [nn.Linear(c_in, nh), nn.LeakyReLU()]

        for _ in range(nl):
            layers.append(nn.Linear(nh, nh))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(nh, c_out))
        mlp = nn.Sequential(*layers)

        for lin_layer in mlp:
            if isinstance(lin_layer, nn.Linear):
                nn.init.constant_(lin_layer.bias, 0)
                nn.init.xavier_uniform_(lin_layer.weight)

        nn.init.constant_(mlp[-1].bias, 0)
        nn.init.constant_(mlp[-1].weight, 0)
        return mlp

    nodes = [InputNode(ndim_x, name="input")]
    cond = ConditionNode(ndim_y, name="condition")

    for i_cc in range(ncc):
        nodes.append(
            Node(
                nodes[-1],
                GLOWCouplingBlock,
                dict(subnet_constructor=subnet_fc, clamp=2.0),
                name=f"coupling_{i_cc}",
                conditions=cond,
            )
        )
        nodes.append(
            Node(
                nodes[-1],
                PermuteRandom,
                {"seed": i_cc},
                name=f"permute_{i_cc}",
            )
        )

    model = ReversibleGraphNet(
        nodes + [cond, OutputNode(nodes[-1])], verbose=False
    )
    return model


def plot_chunks(
    ax,
    lst: Sequence[Sequence[float]],
    plot_kwds=dict(),
    vlines_kwds=dict(colors="gray", linestyles="--"),
    log_y: bool = False,
):
    line_pos = []
    start = 0
    plot = ax.semilogy if log_y else ax.plot
    for y in lst:
        x = start + np.arange(len(y))
        plot(x, y, **plot_kwds)
        start = x[-1]
        line_pos.append(start)
    ax.vlines(line_pos[:-1], *ax.yaxis.get_data_interval(), **vlines_kwds)


if __name__ == "__main__":

    T.set_num_threads(cpu_count() // 2)

    nsteps = 2
    npoints = 512
    batch_size_div = 2
    batch_size = max(npoints // batch_size_div, 1)
    print_every_epoch = 50
    max_epoch = int(1e4)

    ds = generate.TimeDependentTensorDataset(
        *generate.generate_toy8(label_kind="all", npoints=npoints, seed=123),
        dt=5,
    )

    train_dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = build_model(ndim_x=2, ndim_y=8, ncc=1, nh=512, nl=1)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = T.optim.AdamW(
        trainable_parameters,
        lr=1e-3 / batch_size_div,
        ##betas=(0.8, 0.9),
        ##eps=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )

    loss_hist = []
    model.train()

    # PIC time step
    for i_step in range(nsteps):
        # train some epochs on this time step's data
        i_epoch = -1
        mean_epoch_loss_hist = []
        conv = Convergence(
            mode="fall", wlen=20, datol=5e-4, wait=5, reduction=np.median
        )
        while True:
            i_epoch += 1
            loss_sum = 0.0
            for i_batch, (X_batch, Y_batch) in enumerate(train_dl):
                ##ic(i_step, i_epoch, i_batch, ds.time, X_batch, Y_batch)
                ##ic(i_step, i_epoch, i_batch, ds.time)

                optimizer.zero_grad()
                z, log_j = model(X_batch, c=Y_batch)
                ##rev_X, _ = model(z, c=Y_batch, rev=True)
                loss = (T.mean(z**2.0) - T.mean(log_j)) / 2.0
                nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
                loss_sum += loss.data.item()
                loss.backward()
                optimizer.step()

            mean_epoch_loss = loss_sum / batch_size
            mean_epoch_loss_hist.append(mean_epoch_loss)

            if (i_epoch + 1) % print_every_epoch == 0:
                ic(i_step, i_epoch, mean_epoch_loss)

            # termination (reconstruction loss converged, etc)
            if conv.check(mean_epoch_loss_hist):
                print(f"converged, last {mean_epoch_loss=}")
                break
            elif (i_epoch + 1) == max_epoch:
                break
                print(f"hit max iter, last {mean_epoch_loss=}")

        ds.step()
        loss_hist.append(np.array(mean_epoch_loss_hist))

    ncols = 3
    fig, axs = plt.subplots(ncols=ncols, figsize=(5 * ncols, 5))

    # Reset time, re-create train data for plotting
    ds.time = 0
    Xt_3d, Yt_3d = generate.tdds_arrays(
        ds=ds, nsteps=nsteps, batch_size=npoints
    )
    Xt = Xt_3d.numpy().reshape((-1, Xt_3d.shape[-1]))
    Yt = Yt_3d.numpy().reshape((-1, Yt_3d.shape[-1]))
    color = generate.label2color_toyN(Yt)

    axs[0].scatter(Xt[:, 0], Xt[:, 1], c=color, s=0.5)
    with T.no_grad():
        rev_X, _ = model(T.randn(Xt.shape), c=T.from_numpy(Yt), rev=True)

        axs[1].scatter(rev_X[:, 0], rev_X[:, 1], c=color, s=0.5)

        loss_hist_shifted = [x + np.abs(np.min(x)) + 1 for x in loss_hist]
        plot_chunks(axs[2], loss_hist_shifted, log_y=True)
        ##plot_chunks(axs[2], loss_hist, log_y=False)

    plt.show()
