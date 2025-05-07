#!/usr/bin/env python3

from typing import Sequence

from icecream import ic
from torch.utils.data import DataLoader
import torch as T
from torch import nn
import dill as pickle
from matplotlib import pyplot as plt
import numpy as np

from FrEIA.framework import (
    InputNode,
    OutputNode,
    Node,
    ReversibleGraphNet,
    ConditionNode,
)
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

try:
    from nmbx.convergence import SlopeZero

    conv_control_possible = True
except ImportError:
    conv_control_possible = False
    print("no convergence control possible, we use max_epoch")

from insituml.toy_data import generate, memory


# Adapted from Nico_toy8_examples/train_cINN_distributed_toy8.py
def build_model(
    ndim_x=2, ndim_y=8, n_coupling=2, n_hidden=1, hidden_width=512
):
    """
    Parameters
    ----------
    ndim_x
        Input dims. 2 in case of toy8.
    ndim_y
        Output (= condition) dims. 8 in case of toy8 (8 modes, one-hot label
        vectors of length 8).
    n_coupling
        Number of coupling blocks.
    n_hidden
        Number of hidden layers of width hidden_width in coupling block internal FCNs.
    hidden_width
        Hidden layer width of coupling block internal fully connected nets (FCNs).
    """

    def subnet_fc(c_in, c_out):
        layers = [nn.Linear(c_in, hidden_width), nn.LeakyReLU()]

        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_width, c_out))
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

    for i_cb in range(n_coupling):
        nodes.append(
            Node(
                nodes[-1],
                GLOWCouplingBlock,
                dict(subnet_constructor=subnet_fc, clamp=2.0),
                name=f"coupling_{i_cb}",
                conditions=cond,
            )
        )
        nodes.append(
            Node(
                nodes[-1],
                PermuteRandom,
                {"seed": i_cb},
                name=f"permute_{i_cb}",
            )
        )

    model = ReversibleGraphNet(
        nodes + [cond, OutputNode(nodes[-1])], verbose=False
    )
    return model


def train(use_mem=False):
    """
    Parameters
    ----------
    use_mem
        Use memory mechanism to counteract "forgetting"
    """

    data_file = f"toy_cl_mem_{use_mem}.pk"

    use_cuda = T.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    ic(device)
    ic(use_mem)

    # Manually switch on/off convergence control
    conv_control = True

    if conv_control:
        assert conv_control_possible

    # Production settings
    nsteps = 20
    npoints = 2048
    print_every_epoch = 10
    max_epoch = int(1e4) if conv_control else int(1e3)

    # For testing
    ##nsteps = 3
    ##npoints = 128
    ##print_every_epoch = 1
    ##max_epoch = 3

    batch_size_div = 4
    batch_size = max(npoints // batch_size_div, 1)
    seed = 123
    dt = 0.3
    label_kind = "all"

    ds = generate.TimeDependentTensorDataset(
        *generate.generate_toy8(
            label_kind=label_kind, npoints=npoints, seed=seed
        ),
        dt=dt,
    )

    train_dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = build_model(
        ndim_x=2, ndim_y=8, n_coupling=4, hidden_width=512, n_hidden=1
    )

    if use_cuda:
        model = model.to(device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = T.optim.AdamW(
        trainable_parameters,
        lr=1e-3 / batch_size_div / 2,
        ##betas=(0.8, 0.9),
        ##eps=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )

    loss_hist = []
    model.train()

    if use_mem:
        er_mem = memory.ExperienceReplay(mem_size=npoints)
        n_obs = 0

    # PIC time step
    for i_step in range(nsteps):
        # Train some epochs on this time step's data, until convergence if
        # conv_control is used.
        i_epoch = -1
        mean_epoch_loss_hist = []
        if conv_control:
            conv = SlopeZero(wlen=25, tol=1e-2, wait=20, reduction=np.mean)
        while True:
            i_epoch += 1
            loss_sum = 0.0
            for i_batch, (X_batch, Y_batch) in enumerate(train_dl):
                if use_cuda:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)

                # Don't use memory in first step since there is nothing to
                # remember.
                if use_mem and i_step > 0:
                    X_batch_mem, Y_batch_mem = er_mem.sample(batch_size)
                    if use_cuda:
                        X_batch_mem = X_batch_mem.to(device)
                        Y_batch_mem = Y_batch_mem.to(device)
                    Xb = T.vstack((X_batch, X_batch_mem))
                    Yb = T.vstack((Y_batch, Y_batch_mem))
                else:
                    Xb, Yb = X_batch, Y_batch

                optimizer.zero_grad()
                z, log_j = model(Xb, c=Yb)
                loss = (T.mean(z**2.0) - T.mean(log_j)) / 2.0
                nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
                loss_sum += loss.data.item()
                loss.backward()
                optimizer.step()

                # update mem in every batch as in the ExperienceReplay paper
                if use_mem:
                    er_mem.update_memory(X_batch, Y_batch, n_obs, i_step)
                    n_obs += batch_size

            ### update mem in every epoch with last batch only
            ##if use_mem:
            ##    er_mem.update_memory(X_batch, Y_batch, n_obs, i_step)
            ##    n_obs += batch_size

            mean_epoch_loss = loss_sum / (i_batch + 1)
            mean_epoch_loss_hist.append(mean_epoch_loss)

            if (i_epoch + 1) % print_every_epoch == 0:
                ic(i_step, i_epoch, mean_epoch_loss)

            # termination (reconstruction loss converged, etc)
            if conv_control and conv.check(mean_epoch_loss_hist):
                print(f"converged at {i_epoch=}, last {mean_epoch_loss=}")
                break
            elif (i_epoch + 1) == max_epoch:
                print(f"hit max_epoch, last {mean_epoch_loss=}")
                break

        ds.step()
        loss_hist.append(np.array(mean_epoch_loss_hist))

        ### update mem in PIC step with last batch only
        ##if use_mem:
        ##    er_mem.update_memory(X_batch, Y_batch, n_obs, i_step)
        ##    n_obs += batch_size

        if use_mem:
            ic(er_mem.status())

    # Bring model back to CPU before saving to cache.
    if use_cuda:
        model = model.to("cpu")

    data = dict(
        label_kind=label_kind,
        dt=dt,
        seed=seed,
        model=model,
        nsteps=nsteps,
        npoints=npoints,
        loss_hist=loss_hist,
    )

    pickle.dump(data, open(data_file, "wb"))
    return data_file


def plot_chunks(
    ax,
    lst: Sequence[Sequence[float]],
    plot_kwds=dict(),
    vlines_kwds=dict(colors="gray", linestyles="--", alpha=0.5),
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
    ##ymin, ymax = ax.yaxis.get_data_interval()
    ymin, ymax = ax.get_ylim()
    ax.vlines(line_pos[:-1], ymin, ymax, **vlines_kwds)
    # Fix changed ylim bottom from adding vlines ... sigh.
    ax.set_ylim(bottom=ymin)


def plot(data_file: str = None):
    data = pickle.load(open(data_file, "rb"))

    plt.rcParams["font.size"] = 15

    npoints = data["npoints"]
    nsteps = data["nsteps"]
    model = data["model"]
    loss_hist = data["loss_hist"]

    # Re-create train data for plotting
    ds = generate.TimeDependentTensorDataset(
        *generate.generate_toy8(
            label_kind=data["label_kind"], npoints=npoints, seed=data["seed"]
        ),
        dt=data["dt"],
    )
    # Should be default, but just to make sure
    ds.time = 0

    ncols = 3
    fig, axs = plt.subplots(
        ncols=ncols, figsize=(5 * ncols, 5), tight_layout=True
    )

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
        plot_chunks(axs[2], loss_hist, log_y=False)

    for ax in axs[:2]:
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")

    axs[2].set_xlabel("epoch")
    axs[2].set_ylabel("mean epoch loss")
    axs[2].set_ylim(top=0.0)

    for ext in ["png"]:
        fig_fn = data_file.replace(".pk", f".{ext}")
        fig.savefig(fig_fn)


if __name__ == "__main__":
    data_file = train(use_mem=False)
    plot(data_file)
