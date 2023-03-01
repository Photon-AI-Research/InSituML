#!/usr/bin/env python3
import pandas as pd
import socket
import argparse
from typing import Sequence
from multiprocessing import cpu_count
import time
from icecream import ic

from torch.utils.data import DataLoader
import torch
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

try:
    from nmbx.convergence import SlopeZero

    conv_control_possible = True
except ImportError:
    conv_control_possible = False
    print("no convergence control possible, we use max_epoch")

from insituml.toy_data import generate, memory
import wandb

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

    parser = argparse.ArgumentParser(description='Scaling benchmark of NF training')
    parser.add_argument('--features', dest='features', type=int, default=128)
    parser.add_argument('--coupling_blocks', dest='couplingBlocks', type=int, default=6)
    parser.add_argument('--use_mem', dest='use_mem', action='store_true')
    parser.set_defaults(use_mem=False)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(cpu_count() // 2)

    ### CONSTANTS ###
    
    # Manually switch on/off convergence control
    conv_control = False

    # Use memory mechanism to counteract "forgetting"
    use_mem = args.use_mem
    
    nsteps = 500
    npoints = 2**20 # 1M particles as to be expected from PIC
    batch_size = 512
    print_every_epoch = 50
    max_epoch = int(1e4) if conv_control else int(1e3)
    
    dimension_problem = 9 # 2 or 9
    

    if conv_control:
        assert conv_control_possible


    ## performance benchmark
    #ds = torch.utils.data.TensorDataset(
    #    *generate.generate_toy8(label_kind="all", npoints=npoints, seed=123, device = "cpu")
    #)
    
    

    #train_dl = DataLoader(
    #    ds, batch_size=batch_size, num_workers=16
    #)
    
    

    model = build_model(
        ndim_x=dimension_problem, ndim_y=8, n_coupling=args.couplingBlocks, hidden_width=args.features, n_hidden=1
    )
    
    model = model.to(device)
    
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=2e-4, 
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
    start_time = time.time()
    #with torch.profiler.profile(
    #activities=[
    #    torch.profiler.ProfilerActivity.CPU,
    #    torch.profiler.ProfilerActivity.CUDA,
    #]
    #) as p:
    # train some epochs on this time step's data
    i_epoch = -1
    mean_epoch_loss_hist = []
    
    X_batch, Y_batch = generate.generate_toy8(label_kind="all", npoints=npoints, seed=123, device = device)
    
    if(dimension_problem == 9):
        X_batch = torch.cat([X_batch, X_batch.clone(), X_batch.clone(), X_batch.clone(), X_batch[:,0:1].clone()], dim = 1)
    
    for i_step in range(nsteps):
        i_epoch += 1
        loss_sum = 0.0
        #for (X_batch, Y_batch) in train_dl:

        # Make sure data is copied to correct device
        #X_batch = X_batch.to(device)
        #Y_batch = Y_batch.to(device)

        # Don't use memory in first step since there is nothing to
        # remember.
        if use_mem and i_step > 0:
            X_batch_mem, Y_batch_mem = er_mem.sample(batch_size)
            Xb = torch.vstack((X_batch, X_batch_mem))
            Yb = torch.vstack((Y_batch, Y_batch_mem))
        else:
            Xb, Yb = X_batch, Y_batch

        optimizer.zero_grad()
        z, log_j = model(Xb, c=Yb)
        loss = (torch.mean(z**2.0) - torch.mean(log_j)) / 2.0
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

        mean_epoch_loss = loss_sum / batch_size
        mean_epoch_loss_hist.append(mean_epoch_loss)

        runtime = time.time() - start_time

        if (i_step + 1) % print_every_epoch == 0:
            ic(i_step, mean_epoch_loss, runtime)

        loss_hist.append(np.array(mean_epoch_loss_hist))

        ### update mem in PIC step with last batch only
        ##if use_mem:
        ##    er_mem.update_memory(X_batch, Y_batch, n_obs, i_step)
        ##    n_obs += batch_size

        if use_mem:
            ic(er_mem.status())

    #print("==== CONVERGENCE ====")
    #ic(i_step, mean_epoch_loss, runtime)

    runtime = time.time() - start_time
    device = torch.cuda.get_device_name(0)
    no_coupling_blocks = args.couplingBlocks
    no_features = args.features
    use_mem = args.use_mem
    hostname = socket.gethostname()
    nll = mean_epoch_loss
    

    data = {'host': [hostname], 'device': [device], 'host': [hostname], 'coupling blocks': [no_coupling_blocks], 'features': [no_features], 'CL': [use_mem], 'NLL': [nll] }
    df = pd.DataFrame(data)
    df.to_csv('perf_log.csv', mode='a')
    print(df)

    #print(p.key_averages().table(
    #    sort_by="self_cuda_time_total", row_limit=-1))
