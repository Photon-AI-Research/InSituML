import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .utilities import inspect_and_select
import math
# some part of this module is taken from  https://github.com/lingjiekong/CS236Project


def adjust_args(arg1, arg2, kernel_size):
    """
    Simple function for encapsulation the dealing with different
    arguments required for Conv1d and Linear layers.
    """

    if kernel_size is not None:
        return (arg1, arg2, kernel_size)

    else:
        return (arg1, arg2)


class AddLayersMixin:
    """
    Class for the common method used by both Encoder and MLPDecoder,
    for adding sequential layers with common properties like batch normalisation
    and activation after convolutional layers and linear, fully connected layers.
    """

    def add_layers_seq(
        self,
        layer_kind,
        config,
        input_dim,
        add_activation=True,
        add_batch_normalisation=True,
        kernel_size=None,
    ):
        """
        This methods creates a python array to be later added to nn.Sequential with similar layer block like convolution, activation, batchnorm. Allows the turning of batch normalisation, activation for different functions.

        Args:

        layer_kind(str): Kind of layer to use. Can be any string in the
        torch.nn module members. Though, only tested for Conv1d, Linear.

        config (list): List of sizes of layers to be used.

        add_activation (Bool): Whether to add activation, after every
        layer_kind layer.

        add_batch_normalisation (Bool): Whether to add batch normalisation, after every
        layer_kind layer.

        kernel_size (int): Kernel size in case of Conv1d.
        """
        layers = []

        for idx, channel_size in enumerate(config):

            input_args = []

            if idx == 0:
                layers.append(
                    getattr(nn, layer_kind)(
                        *adjust_args(input_dim, channel_size, kernel_size)
                    )
                )
            else:
                layers.append(
                    getattr(nn, layer_kind)(
                        *adjust_args(
                            config[idx - 1], channel_size, kernel_size
                        )
                    )
                )

            if add_batch_normalisation:
                layers.append(nn.BatchNorm1d(channel_size))

            if add_activation:
                layers.append(nn.ReLU())

        return layers


class Encoder(AddLayersMixin, nn.Module):
    """
    Encoder for adding a mixture of convolution layers and fully connected layers.
    Like the one used here:
    https://arxiv.org/abs/1906.12320

    Args:

    zdim (int): Dimensions of latent space.

    ae_config (str): Configuration of Encoder required. Could deterministic, non_deterministic, or simple.

    conv_layer_config (List): List of sizes of convolutional layers.

    conv_add_bn (Bool): Whether to add batch normalisation after the convolutional layers.

    conv_add_activation (Bool): Whether to add activation function after the convolutional layers.

    kernel_size (int): Kernel size for convolutional layers.

    fc_layer_config (List): List of sizes of fully connected layers.

    fc_add_bn (Bool): Whether to add batch normalisation after the fully connected layers.

    fc_add_activation (Bool): Whether to add activation function after the fully connected layers.

    """

    def __init__(
        self,
        z_dim,
        input_dim=3,
        particles_to_sample=None,
        ae_config="deterministic",
        conv_layer_config=[128, 128, 256, 512],
        conv_add_bn=True,
        conv_add_activation=True,
        kernel_size=1,
        fc_layer_config=[256, 256],
        fc_add_bn=True,
        fc_add_activation=True,
    ):

        super().__init__()

        conv_layers = self.add_layers_seq(
            "Conv1d",
            conv_layer_config,
            input_dim,
            add_batch_normalisation=conv_add_bn,
            add_activation=conv_add_activation,
            kernel_size=kernel_size,
        )

        self.ae_config = ae_config

        self.ll_size = conv_layer_config[-1]

        conv_layers += [nn.AdaptiveMaxPool1d(1), nn.Flatten()]

        if ae_config == "deterministic":

            fc_layers = self.add_layers_seq(
                "Linear",
                fc_layer_config,
                self.ll_size,
                add_batch_normalisation=fc_add_bn,
                add_activation=fc_add_activation,
            )

            final_layers = (
                conv_layers
                + fc_layers
                + [nn.Linear(fc_layer_config[-1], z_dim)]
            )

            self.layers = nn.Sequential(*final_layers)

        elif ae_config == "non_deterministic":

            self.layers = nn.Sequential(*conv_layers)

            fc_layers_mean = self.add_layers_seq(
                "Linear",
                fc_layer_config,
                self.ll_size,
                add_batch_normalisation=fc_add_bn,
                add_activation=fc_add_activation,
            )

            fc_layers_var = self.add_layers_seq(
                "Linear",
                fc_layer_config,
                self.ll_size,
                add_batch_normalisation=fc_add_bn,
                add_activation=fc_add_activation,
            )

            partition_mean = fc_layers_mean + [
                nn.Linear(fc_layer_config[-1], z_dim)
            ]

            partition_var = fc_layers_var + [
                nn.Linear(fc_layer_config[-1], z_dim),
                nn.Softplus(),
            ]

            self.mean = nn.Sequential(*partition_mean)
            self.variance = nn.Sequential(*partition_var)

        elif ae_config == "simple":
            # take away the maxpool, and flatten
            final_layers = (
                conv_layers[:-2]
                + [nn.Conv1d(conv_layer_config[-1], z_dim, kernel_size)]
                + conv_layers[-2:]
            )

            self.layers = nn.Sequential(*final_layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)

        if self.ae_config == "deterministic":
            return x, 0
        elif self.ae_config == "non_deterministic":
            return self.mean(x), self.variance(x) + 1e-8
        else:
            return x


# decoder design
class MLPDecoder(AddLayersMixin, nn.Module):
    """
    Decoder for adding fully connected layers.
    Like the one used here:
    https://arxiv.org/abs/1906.12320

    Args:

    zdim (int): Dimensions of latent space.

    n_point (int): Number of points in the point clouds.

    point_dim (int): Dimension of point cloud.

    layer_config (List): The list of configuration for decoder setup

    """

    def __init__(
        self,
        z_dim,
        particles_to_sample,
        input_dim,
        ae_config=None,
        layer_config=[256],
        add_batch_normalisation=False,
    ):

        super().__init__()

        out_dims = input_dim * particles_to_sample

        # normalisation was removed in this setup:
        # https://arxiv.org/abs/1906.12320 see Appendix.
        layers = self.add_layers_seq(
            "Linear",
            layer_config,
            z_dim,
            add_batch_normalisation=add_batch_normalisation,
        )

        layers = layers + [nn.Linear(layer_config[-1], out_dims)]

        layers = layers + [
            nn.Flatten(),
            nn.Unflatten(1, (particles_to_sample, input_dim)),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


class Conv3DDecoder(AddLayersMixin, nn.Module):
    """
    Convolutional 3D Decoder Module.

    Args:
        z_dim (int): Dimension of the latent space.
        input_dim (int): Dimension of the input.
        initial_conv3d_size (list): Initial size for unflattening the latent vector.
        conv3d_layer_config (list): Configuration of convolutional layers.
        fc_layer_config (list, optional): Configuration of fully connected layers.
        kernel_size (int, optional): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution operation.
        padding (int, optional): Padding of the convolution operation.
        add_batch_normalisation (bool): Whether to add batch normalization layers.
        add_activation (bool): Whether to add activation functions.
        output_points (int): Specify exact number of output points
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        z_dim,
        input_dim,
        initial_conv3d_size=[16, 4, 4, 4],
        conv3d_layer_config=[8],
        fc_layer_config=[],
        kernel_size=2,
        stride=2,
        padding=0,
        add_batch_normalisation=True,
        add_activation=True,
        output_points=None,
        **kwargs,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.output_points = output_points
        self.initial_conv3d_size = initial_conv3d_size
        self.conv3d_layer_config = conv3d_layer_config
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        layers = []

        if fc_layer_config:
            # Compute the expected size after fully connected layers
            expected_fc_output_size = (
                torch.tensor(initial_conv3d_size).prod().item()
            )
            assert (
                fc_layer_config[-1] == expected_fc_output_size
            ), f"Last FC layer output size {fc_layer_config[-1]} does not match the expected size {expected_fc_output_size}"

            fc_layers = self.add_layers_seq(
                "Linear",
                fc_layer_config,
                z_dim,
                add_batch_normalisation=add_batch_normalisation,
                add_activation=add_activation,
            )
            layers += fc_layers

        else:
            expected_unflatten_size = (
                torch.tensor(initial_conv3d_size).prod().item()
            )
            assert (
                z_dim == expected_unflatten_size
            ), f"z_dim {z_dim} does not match the expected size {expected_unflatten_size}"

        # Unflatten layer to reshape the latent vector
        layers.append(nn.Unflatten(1, initial_conv3d_size))

        # Add ConvTranspose3d layers
        in_channels = initial_conv3d_size[0]
        for out_channels in conv3d_layer_config:
            layers.append(
                nn.ConvTranspose3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if add_batch_normalisation:
                layers.append(nn.BatchNorm3d(out_channels))
            if add_activation:
                layers.append(nn.ReLU())
            in_channels = out_channels

        # Final layer to adjust to the correct number of output channels
        layers.append(
            nn.ConvTranspose3d(
                out_channels,
                input_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

        # Flatten the output
        layers.append(nn.Flatten(2))

        if self.output_points:
            self.output_size = self.calculate_output_size()
            layers.append(nn.Linear(self.output_size, self.output_points))

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        z = self.layers(z)
        z = z.transpose(1, 2)
        return z

    def calculate_output_size(self):
        output_size = list(self.initial_conv3d_size[1:])
        for _ in range(len(self.conv3d_layer_config) + 1):
            output_size = [
                (size - 1) * self.stride - 2 * self.padding + self.kernel_size
                for size in output_size
            ]

        output_size = math.prod(output_size)
        return output_size
