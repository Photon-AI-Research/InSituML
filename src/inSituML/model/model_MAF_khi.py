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
import matplotlib.pyplot as plt

from model.modules import data_preprocessing
from model.modules import loader


class PC_MAF(nn.Module):
    def __init__(
        self,
        dim_condition,
        dim_input,
        num_coupling_layers=1,
        hidden_size=128,
        device="cpu",
        enable_wandb=False,
        weight_particles=False,
    ):
        """
        Masked autoregressive flows model from https://papers.nips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html
        Args:
            dim_condition(integer): dimensionality of condition
            dim_input(integer): dimensionality of input
            num_coupling_blocks(integer): number of coupling blocks in the model
            hidden_size(integer): number of hidden units per hidden layer in subnetworks
            device: "cpu" or "cuda"
            enable_wandb(boolean): True to watch training progress at wandb.ai

        """

        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_coupling_layers = num_coupling_layers
        self.hidden_size = hidden_size

        self.dim_input = dim_input
        self.dim_condition = dim_condition
        self.model = self.init_model().to(self.device)
        self.vmin_ps = None
        self.vmax_ps = None
        self.vmin_rad = None
        self.vmax_rad = None
        self.a = None
        self.b = None

        self.enable_wandb = enable_wandb
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
                )
            )
        transform = CompositeTransform(transforms)

        return Flow(transform, base_dist).to(self.device)

    def forward(self, x, p):
        loss = self.model(x, c=p)
        return loss

    def train_(
        self,
        dataset_tr,
        dataset_val,
        optimizer,
        loader,
        epochs=100,
        batch_size=2,
        test_epoch=25,
        test_pointcloud=None,
        test_radiation=None,
        log_plots=None,
        path_to_models=None,
    ):
        """
        Train model
        Args:
            dataset_tr(torch dataset, see modules/dataset.py): dataset with all needed information about data to be used for training
            dataset_val: dataset with all needed information about data to be used for validation
            optimizer: torch.optim optimizer
            epochs(integer): number of training iterations
            batch_size(integer): size of a batch in loader
            test_epoch(integer): at each test_epoch save perform validation, save pre-trained model is path_to_models is given and register logs to wandb
            test_pointcloud(string): path to a test pointcloud to watch recontsructions at wandb.ai
            log_plots(function): function to log necessary plots of groundtruth and reconstruction, should be not None in case test_pointcloud is not None
            path_to_models(string): path where to save pre-trained models if None then models won't be saved
        """
        # self.vmin_ps = dataset_tr.vmin_ps
        # self.vmax_ps = dataset_tr.vmax_ps
        # self.vmin_rad = dataset_tr.vmin_rad
        # self.vmax_rad = dataset_tr.vmax_rad
        # self.a = dataset_tr.a
        # self.b = dataset_tr.b

        # loader_tr = loader.get_loader(dataset_tr, batch_size=batch_size)
        loader_tr = loader
        self.model.train()

        for i_epoch in range(epochs):
            loss_avg = []

            for phase_space, radiation in loader_tr:
                # if dataset_tr.normalize:
                #     phase_space = phase_space.squeeze(0)
                #     radiation = radiation.squeeze(0)

                #     phase_space = data_preprocessing.normalize_point(phase_space, dataset_tr.vmin_ps, dataset_tr.vmax_ps, dataset_tr.a, dataset_tr.b)
                #     radiation = data_preprocessing.normalize_point(radiation, dataset_tr.vmin_rad, dataset_tr.vmax_rad, dataset_tr.a, dataset_tr.b)
                # print(radiation)

                # erase particle with nan values and a corresponding radiation
                # radiation = radiation[~torch.any(phase_space.isnan(), dim=1)]
                # phase_space = phase_space[~torch.any(phase_space.isnan(), dim=1)]
                phase_space = phase_space.view(-1, phase_space.size(-1))
                # radiation = radiation.view(-1, radiation.size(-1))

                print("phase_space", phase_space.shape)
                print("radiation", radiation.shape)

                for param in self.model.parameters():
                    param.grad = None
                loss = -self.model.log_prob(
                    inputs=phase_space.to(self.device),
                    context=radiation.to(self.device),
                )
                if self.weight_particles:
                    loss = loss * phase_space[:, -1]
                loss = loss.mean()
                loss_avg.append(loss.item())
                loss.backward()
                optimizer.step()

                if self.enable_wandb:
                    wandb.log({"NLL-loss": float(loss.item())})

            if self.enable_wandb:
                wandb.log({"NLL-loss_avg": sum(loss_avg) / len(loss_avg)})

            if i_epoch % test_epoch == 0:
                if (
                    (not test_pointcloud == None)
                    and (not test_radiation == None)
                    and (not log_plots == None)
                    and self.enable_wandb
                ):
                    log_plots(test_pointcloud, test_radiation, self)

                if path_to_models != None:
                    # print('Save model to '+path_to_models)
                    if not os.path.exists(path_to_models):
                        os.makedirs(path_to_models)
                    self.save_checkpoint(
                        self.model, optimizer, path_to_models, i_epoch
                    )

                print(
                    "epoch : {}/{},\n\tloss_avg = {:.15f}".format(
                        i_epoch + 1, epochs, sum(loss_avg) / len(loss_avg)
                    )
                )
                # self.validation(dataset_val, dataset_tr.vmin, dataset_tr.vmax, batch_size)
                # self.model.train()

    def validation(self, dataset_val, batch_size, normalize):
        loader_val = loader.get_loader(dataset_val, batch_size=batch_size)
        with torch.no_grad():
            loss_avg = []
            for phase_space, radiation in loader_val:
                if normalize:
                    phase_space = phase_space.squeeze(0)
                    radiation = radiation.squeeze(0)

                    phase_space = data_preprocessing.normalize_point(
                        phase_space, self.vmin_ps, self.vmax_ps, self.a, self.b
                    )
                    radiation = data_preprocessing.normalize_point(
                        radiation, self.vmin_rad, self.vmax_rad, self.a, self.b
                    )

                # erase particle with nan values and a corresponding radiation
                radiation = radiation[~torch.any(phase_space.isnan(), dim=1)]
                phase_space = phase_space[
                    ~torch.any(phase_space.isnan(), dim=1)
                ]

                loss = -self.model.log_prob(
                    inputs=phase_space.to(self.device),
                    context=radiation.to(self.device),
                ).mean()
                loss_avg.append(loss.item())
            if self.enable_wandb:
                wandb.log({"NLL-loss_val": sum(loss_avg) / len(loss_avg)})
            print(
                "Validation:\n\tloss_avg = {:.15f}".format(
                    sum(loss_avg) / len(loss_avg)
                )
            )

    def save_checkpoint(self, model, optimizer, path, epoch):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vmin_ps": self.vmin_ps,
            "vmax_ps": self.vmax_ps,
            "vmin_rad": self.vmin_rad,
            "vmax_rad": self.vmax_rad,
            "a": self.a,
            "b": self.b,
        }

        torch.save(state, path + "model_" + str(epoch))

    def sample_pointcloud(self, cond):
        self.model.eval()
        with torch.no_grad():
            pc_pr = (self.model.sample(1, cond)).squeeze(1)
            pc_pr = data_preprocessing.denormalize_point(
                pc_pr.to("cpu"), self.vmin_ps, self.vmax_ps, self.a, self.b
            )
        return pc_pr
