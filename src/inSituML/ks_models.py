"""
Define the AI model
"""

import torch
from torch import nn
from torch import sigmoid as t_sigmoid
from torch import tanh as t_tanh
from torch.nn import functional as F

from nflows.flows.base import Flow
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal as nflows_StandardNormal

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


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
        Masked autoregressive flows model from
        https://papers.nips.cc/paper/2017/hash/
        6c1da886822c67822bcf3679d04369fa-Abstract.html
        Args:
            dim_condition(integer): dimensionality of condition
            dim_input(integer): dimensionality of input
            num_coupling_blocks(integer): number of coupling blocks in the model
            hidden_size(integer): number of hidden units per
              hidden layer in subnetworks
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
            "sigmoid": t_sigmoid,
            "tanh": t_tanh,
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
        base_dist = nflows_StandardNormal(shape=[self.dim_input])

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
            pc_pr = self.model.sample(num_samples, context=cond).squeeze()
        return pc_pr


class INNModel(nn.Module):
    def __init__(
        self,
        ndim_tot,
        ndim_x,
        ndim_y,
        ndim_z,
        loss_fit,
        loss_latent,
        loss_backward,
        lambd_predict=1.0,
        lambd_latent=1.0,
        lambd_rev=1.0,
        zeros_noise_scale=5e-5,
        y_noise_scale=1e-2,
        hidden_size=4,
        num_coupling_layers=0,
        activation="relu",
        device="cpu",
    ):
        super(INNModel, self).__init__()

        self.device = torch.device(device)
        self.ndim_tot = ndim_tot
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z
        self.loss_fit = loss_fit
        self.loss_latent = loss_latent
        self.loss_backward = loss_backward
        self.lambd_predict = lambd_predict
        self.lambd_latent = lambd_latent
        self.lambd_rev = lambd_rev
        self.zeros_noise_scale = zeros_noise_scale
        self.y_noise_scale = y_noise_scale
        self.hidden_size = hidden_size

        # Activation functions
        activation_functions = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
        }
        self.activation = activation_functions.get(activation)

        # Define the subnet_constructor using a method
        def subnet_constructor(c_in, c_out):
            return self.subnet_fc(c_in, c_out)

        self.model = self._build_model(subnet_constructor, num_coupling_layers)

    def subnet_fc(self, c_in, c_out):
        # Define the fully connected subnet
        return nn.Sequential(
            nn.Linear(c_in, self.hidden_size),
            self.activation,
            nn.Linear(self.hidden_size, c_out),
        )

    def _build_model(self, subnet_constructor, num_coupling_layers):
        nodes = [InputNode(self.ndim_tot, name="input")]

        for k in range(num_coupling_layers):
            nodes.append(
                Node(
                    nodes[-1],
                    GLOWCouplingBlock,
                    {"subnet_constructor": subnet_constructor, "clamp": 2.0},
                    name=f"coupling_{k}",
                )
            )
            nodes.append(
                Node(
                    nodes[-1], PermuteRandom, {"seed": k}, name=f"permute_{k}"
                )
            )

        nodes.append(OutputNode(nodes[-1], name="output"))

        return ReversibleGraphNet(nodes, verbose=False)

    def compute_losses(self, x, y):
        device = self.device
        x = x.to(device)
        y = y.to(device)

        y_clean = y.clone()
        device = x.device
        batch_size = x.size(0)

        # Preparing the input
        pad_x = self.zeros_noise_scale * torch.randn(
            batch_size, self.ndim_tot - self.ndim_x, device=device
        )
        pad_yz = self.zeros_noise_scale * torch.randn(
            batch_size,
            self.ndim_tot - self.ndim_y - self.ndim_z,
            device=device,
        )
        y += self.y_noise_scale * torch.randn(
            batch_size, self.ndim_y, dtype=torch.float, device=device
        )

        x, y = (
            torch.cat((x, pad_x), dim=1),
            torch.cat(
                (
                    torch.randn(batch_size, self.ndim_z, device=device),
                    pad_yz,
                    y,
                ),
                dim=1,
            ),
        )

        # Forward pass
        output, _ = self.model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)
        l_fit = self.lambd_predict * self.loss_fit(
            output[:, self.ndim_z:], y[:, self.ndim_z:]
        )
        output_block_grad = torch.cat(
            (output[:, :self.ndim_z], output[:, -self.ndim_y:].data), dim=1
        )
        l_latent = self.lambd_latent * self.loss_latent(
            output_block_grad, y_short
        )
        total_loss = l_fit + l_latent

        # Backward step preparation
        pad_yz = self.zeros_noise_scale * torch.randn(
            batch_size,
            self.ndim_tot - self.ndim_y - self.ndim_z,
            device=device,
        )
        y = y_clean + self.y_noise_scale * torch.randn(
            batch_size, self.ndim_y, device=device
        )
        orig_z_perturbed = output.data[
            :, :self.ndim_z
        ] + self.y_noise_scale * torch.randn(
            batch_size, self.ndim_z, device=device
        )
        y_rev = torch.cat((orig_z_perturbed, pad_yz, y_clean), dim=1)
        y_rev_rand = torch.cat(
            (
                torch.randn(batch_size, self.ndim_z, device=device),
                pad_yz,
                y_clean,
            ),
            dim=1,
        )

        # Backward pass
        output_rev, _ = self.model(y_rev, rev=True)
        output_rev_rand, _ = self.model(y_rev_rand, rev=True)
        l_rev = self.lambd_rev * self.loss_backward(
            output_rev_rand[:, :self.ndim_x], x[:, :self.ndim_x]
        )
        l_rev += self.lambd_predict * self.loss_fit(output_rev, x)

        total_loss += l_rev

        return total_loss, l_fit, l_latent, l_rev

    def forward(self, x, y=None, rev=False):
        device = self.device
        x = x.to(device)
        if y is not None:
            y = y.to(device)

        if not rev:
            # Forward pass
            output, _ = self.model(x)
            return output
        else:
            # Backward pass requires 'y' to be specified
            if y is None:
                raise ValueError("y must be provided for the backward pass")
            device = x.device
            batch_size = x.size(0)
            pad_yz = self.zeros_noise_scale * torch.randn(
                batch_size,
                self.ndim_tot - self.ndim_y - self.ndim_z,
                device=device,
            )
            y_clean = y.clone()

            # Preparing the input for the backward pass
            y = y_clean + self.y_noise_scale * torch.randn(
                batch_size, self.ndim_y, device=device
            )
            orig_z_perturbed = torch.randn(
                batch_size, self.ndim_z, device=device
            )
            y_rev = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)

            # Execute the backward pass
            output_rev, _ = self.model(y_rev, rev=True)
            return output_rev
