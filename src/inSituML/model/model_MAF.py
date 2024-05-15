import sys

sys.path.append("./modules")

import nflows
import numpy as np
import random
import os

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)

from nflows.transforms.standard import AffineTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet

import torch
from torch import nn
from torch import optim
import wandb

from torch.nn import functional as F


class PC_MAF(nn.Module):
    def __init__(
        self,
        dim_condition,
        dim_input,
        num_coupling_layers=1,
        hidden_size=128,
        device="cpu",
        weight_particles=False,
        num_blocks_mat=2,
        activation="relu",
        random_mask=False,
    ):
        """
        Masked autoregressive flows model from https://papers.nips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html
        Args:
            dim_condition(integer): dimensionality of condition
            dim_input(integer): dimensionality of input
            num_coupling_blocks(integer): number of coupling blocks in the model
            hidden_size(integer): number of hidden units per hidden layer in subnetworks
            device: "cpu" or "cuda"

        """

        super().__init__()
        self.device = device
        self.num_coupling_layers = num_coupling_layers
        self.hidden_size = hidden_size
        self.dim_input = dim_input
        self.dim_condition = dim_condition
        self.num_blocks_mat = num_blocks_mat
        self.random_mask = random_mask

        # Activation functions
        activation_functions = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "elu": F.elu,
            "silu": F.silu,
            "leaky_relu": F.leaky_relu,
            "gelu": F.gelu,
        }
        self.activation = activation_functions.get(activation)

        if self.activation is None:
            raise ValueError("Unsupported activation function")

        self.model = self.init_model().to(self.device)
        self.weight_particles = weight_particles

    def init_model(self):
        base_dist = nflows.distributions.normal.StandardNormal(
            shape=[self.dim_input]
        )

        transforms = []
        for _ in range(self.num_coupling_layers):
            transforms.append(ReversePermutation(features=self.dim_input))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.dim_input,
                    hidden_features=self.hidden_size,
                    context_features=self.dim_condition,
                    use_residual_blocks=True,
                    num_blocks=self.num_blocks_mat,
                    activation=self.activation,
                    random_mask=self.random_mask,
                )
            )
        transform = CompositeTransform(transforms)

        return Flow(transform, base_dist).to(self.device)

    def to_device(self, *args):
        """Moves tensors to the same device as the model."""
        return [
            arg.to(self.device) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]

    def forward(self, inputs, context):
        inputs, context = self.to_device(inputs, context)
        log_prob = -self.model.log_prob(inputs, context=context)
        return log_prob.mean()

    def sample_pointcloud(self, num_samples, cond):
        self.model.eval()
        cond = cond.to(self.device)
        with torch.no_grad():
            pc_pr = self.model.sample(num_samples, context=cond)
        return pc_pr
