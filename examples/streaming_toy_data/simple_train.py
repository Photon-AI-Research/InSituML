#!/usr/bin/env python3

import importlib

from icecream import ic
from torch.utils.data import DataLoader

import torch as T
from torch import nn

from FrEIA.framework import (
    InputNode,
    OutputNode,
    Node,
    ReversibleGraphNet,
    ConditionNode,
)
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

from insituml.toy_data import generate

# When used as %run -i this.py in ipython
importlib.reload(generate)

# Adapted from Nico_toy8_examples/train_cINN_distributed_toy8.py
def build_model(ndim_x=2, ndim_cond=8, ncc=2, nh=512, nl=1):
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
    cond = ConditionNode(ndim_cond, name="condition")

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


if __name__ == "__main__":
    ds = generate.TimeDependentTensorDataset(
        *generate.generate_toy8(label_kind="all", npoints=1024, seed=123),
        ##*generate.generate_fake_toy(npoints=6),
        dt=1.0,
    )

    train_dl = DataLoader(ds, batch_size=1024, shuffle=True)

    model = build_model(ndim_x=2, ndim_cond=8)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = T.optim.Adam(
        trainable_parameters, lr=1e-3, betas=(0.8, 0.9), eps=1e-6
    )
    model.train()

    # PIC time step
    for i_time in range(5):
        # train some epochs on this time step's data
        for i_epoch in range(20):
            loss_sum = 0
            for i_batch, (X_batch, Y_batch) in enumerate(train_dl):
                ##ic(i_time, i_epoch, i_batch, ds.time, X_batch, Y_batch)

                optimizer.zero_grad()
                z, log_j = model(X_batch, c=Y_batch)
                loss = (T.mean(z**2.0) - T.mean(log_j)) / 2.0
                nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
                loss_sum += loss.data.item()
                loss.backward()
                optimizer.step()

            ic(i_time, i_epoch, loss_sum)
        ds.step()
