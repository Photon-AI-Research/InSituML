import os
import random

import FrEIA.framework as Ff
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import wandb

from .modules import data_preprocessing
from .modules import dist_utils
from .modules.dist_utils import print0
from .modules import loader

torch.autograd.set_detect_anomaly(True)
#plt.rcParams['image.cmap'] = 'bwr'
#plt.set_cmap('bwr')

class PC_NF(nn.Module):
    def __init__(self, 
                 dim_condition,
                 dim_input,
                 num_coupling_layers=1,
                 num_linear_layers=1,
                 hidden_size=128,
                 device='cpu',
                 enable_wandb=False):
        
        '''
        Conditional normalizing flow model from https://arxiv.org/abs/1907.02392
        Args:
            dim_condition(integer): dimensionality of condition
            dim_input(integer): dimensionality of input
            num_coupling_blocks(integer): number of coupling blocks in the model
            num_linear_layers(integer): number of hidden linear layers in subnetworks
            hidden_size(integer): number of hidden units per hidden layer in subnetworks
            device: "cpu" or "cuda"
            enable_wandb(boolean): True to watch training progress at wandb.ai
            
        '''

        super().__init__()
        self.device = device
        self.num_coupling_layers = num_coupling_layers
        self.num_linear_layers = num_linear_layers
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

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        
        self.enable_wandb = enable_wandb
    
    def init_model(self):
        """
        Initialize coupling blocks
        """
        nodes = [InputNode(self.dim_input, name='input')]
        cond = Ff.ConditionNode(self.dim_condition) 
        
        for k in range(0, self.num_coupling_layers):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':self.subnet_fc, 'clamp':2.0},
                              name=F'coupling_{k}', conditions=cond))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))
        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def subnet_fc(self, c_in, c_out):
        """
        Initialize subnetwork
        """
        layers=[nn.Linear(c_in, self.hidden_size), nn.LeakyReLU()]
        for i in range(self.num_linear_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.LeakyReLU())
            
        layers.append(nn.Linear(self.hidden_size,  c_out))
        mlp = nn.Sequential(*layers)
        for lin_layer in mlp:
            if(isinstance(lin_layer, nn.Linear)):
                nn.init.constant_(lin_layer.bias, 0)
                nn.init.xavier_uniform_(lin_layer.weight)

        nn.init.constant_(mlp[-1].bias, 0)
        nn.init.constant_(mlp[-1].weight, 0)
        return mlp
    
    def forward(self, x, p):
        z, jac = self.model(x, c=p)
        return z, jac

    def train_(self, 
               dataset_tr,
               dataset_val,
               optimizer,
               epochs=100,
               batch_size=2,
               test_epoch=25,
               test_pointcloud=None, test_radiation=None, log_plots=None,
               path_to_models='./RESmodels_freia/'):
        '''
        Train model
        Args:
            dataset_tr(torch dataset, see modules/dataset.py): dataset with all needed information about data to be used for training
            dataset_val: dataset with all needed information about data to be used for validation
            optimizer: torch.optim optimizer 
            epochs(integer): number of training iterations
            batch_size(integer): size of a batch in loader
            test_epoch(integer): at each test_epoch do validation, save pre-trained model if path_to_models is given and register logs to wandb
            test_pointcloud(string): path to a test pointcloud to watch recontsructions at wandb.ai
            log_plots(function): function to log necessary plots of groundtruth and reconstruction, should be not None in case test_pointcloud is not None
            path_to_models(string): path where to save pre-trained models if None then models won't be saved
        '''
        
        self.vmin_ps = dataset_tr.vmin_ps
        self.vmax_ps = dataset_tr.vmax_ps
        self.vmin_rad = dataset_tr.vmin_rad
        self.vmax_rad = dataset_tr.vmax_rad
        self.a = dataset_tr.a
        self.b = dataset_tr.b


        #loader_tr(torch data loader, see modules/loader.py): iterator over training data, 
        #each returned elem should be a tuple of radiation tensor and a chunk of particle data (chunk_size, 6)
        if path_to_models is not None and dist_utils.is_rank_0():
            os.makedirs(path_to_models, exist_ok=True)

        loader_tr = loader.get_loader(dataset_tr, batch_size=batch_size)
        self.model.train()
        
        for i_epoch in range(epochs):
            loss_avg = []
            loss_z = []
            loss_j = []

            loader_tr.sampler.set_epoch(i_epoch)
            for phase_space, radiation in loader_tr:
                if dataset_tr.normalize:
                    phase_space = phase_space.squeeze(0)
                    radiation = radiation.squeeze(0)
                    
                    phase_space = data_preprocessing.normalize_point(phase_space, dataset_tr.vmin_ps, dataset_tr.vmax_ps, dataset_tr.a, dataset_tr.b)
                    radiation = data_preprocessing.normalize_point(radiation, dataset_tr.vmin_rad, dataset_tr.vmax_rad, dataset_tr.a, dataset_tr.b)

                #erase particle with nan values and a corresponding radiation
                radiation = radiation[~torch.any(phase_space.isnan(), dim=1)]
                phase_space = phase_space[~torch.any(phase_space.isnan(), dim=1)]
                for param in self.model.parameters():
                    param.grad = None
                #optimizer.zero_grad()
                z, log_j = self(phase_space.to(self.device),
                                radiation.to(self.device))
                loss = 0.

                loss_z.append(torch.mean(z**2) / 2)
                loss_j.append(torch.mean(log_j)/2)
                loss = torch.mean(z**2) / 2 - torch.mean(log_j)/2

                torch.nn.utils.clip_grad_norm_(self.trainable_parameters, 10.)

                # Average across distributed ranks.
                loss_z[-1] = float(
                    dist_utils.all_reduce_avg(loss_z[-1].detach()))
                loss_j[-1] = float(
                    dist_utils.all_reduce_avg(loss_j[-1].detach()))
                curr_loss_avg = dist_utils.all_reduce_avg(loss.detach())
                loss_avg.append(curr_loss_avg.item())

                loss.backward()
                optimizer.step()

                if self.enable_wandb and dist_utils.is_rank_0():
                    wandb.log({'loss_z_tr': loss_z[-1],
                               'loss_jac_tr': loss_j[-1],
                               'loss_tr': loss_avg[-1]})

            if self.enable_wandb and dist_utils.is_rank_0():
                wandb.log({'loss_z_avg_tr': sum(loss_z)/len(loss_z),
                           'loss_jac_avg_tr': sum(loss_j)/len(loss_j),
                           'loss_avg_tr': sum(loss_avg)/len(loss_avg)})

            if i_epoch % test_epoch == 0: 
                if (
                        not test_pointcloud == None
                        and not test_radiation == None
                        and not log_plots == None
                        and self.enable_wandb
                        and dist_utils.is_rank_0()
                ):
                    log_plots(test_pointcloud, test_radiation, self)

                if path_to_models != None:
                    self.save_checkpoint(self.model, optimizer, path_to_models, i_epoch)

                print0(
                    (
                        "epoch : {}/{},\n\tloss_avg = {:.15f},\n"
                        "\tloss_z = {:.15f},\n\tloss_j = {:.15f}"
                    ).format(i_epoch + 1, epochs,
                             sum(loss_avg)/len(loss_avg),
                             sum(loss_z)/len(loss_z),
                             sum(loss_j)/len(loss_j)))
                #self.validation(dataset_val, 
                #                dataset_tr.vmin_ps, dataset_tr.vmax_ps,
                #                dataset_tr.vmin_rad, dataset_tr.vmax_rad,
                #                batch_size, dataset_tr.normalize, dataset_tr.a, dataset_tr.b)
                self.model.train()
                
    def validation(self, dataset_val,
                   batch_size, normalize):
        loader_val = loader.get_loader(dataset_val, batch_size=batch_size)
        self.model.eval()
        with torch.no_grad():
            loss_avg = []
            loss_z = []
            loss_j = []

            for phase_space, radiation in loader_val:
                if normalize:
                    phase_space = phase_space.squeeze(0)
                    radiation = radiation.squeeze(0)
                    
                    phase_space = data_preprocessing.normalize_point(phase_space, self.vmin_ps, self.vmax_ps, self.a, self.b)
                    radiation = data_preprocessing.normalize_point(radiation, self.vmin_rad, self.vmax_rad, self.a, self.b)

                #erase particle with nan values and a corresponding radiation
                radiation = radiation[~torch.any(phase_space.isnan(), dim=1)]
                phase_space = phase_space[~torch.any(phase_space.isnan(), dim=1)]

                z, log_j = self(phase_space.to(self.device),
                                radiation.to(self.device))
                loss = 0.

                loss_z.append(torch.mean(z**2) / 2)
                loss_j.append(torch.mean(log_j)/2)
                loss = torch.mean(z**2) / 2 - torch.mean(log_j)/2

                # Average across distributed ranks.
                loss_z[-1] = float(
                    dist_utils.all_reduce_avg(loss_z[-1].detach()))
                loss_j[-1] = float(
                    dist_utils.all_reduce_avg(loss_j[-1].detach()))
                curr_loss_avg = dist_utils.all_reduce_avg(loss.detach())

                loss_avg.append(curr_loss_avg.item())
            if self.enable_wandb and dist_utils.is_rank_0():
                wandb.log({'loss_z_val': sum(loss_z)/len(loss_z),
                       'loss_jac_val': sum(loss_j)/len(loss_j),
                       'loss_val': sum(loss_avg)/len(loss_avg)})
            print0(
                (
                    "Validation:\n\tloss_avg = {:.15f},\n"
                    "\tloss_z = {:.15f},\n\tloss_j = {:.15f}"
                ).format(sum(loss_avg)/len(loss_avg),
                         sum(loss_z)/len(loss_z),
                         sum(loss_j)/len(loss_j)))
        self.model.train()

    def save_checkpoint(self, model, optimizer, path, epoch):
        if dist_utils.is_rank_0():
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vmin_ps': self.vmin_ps,
                'vmax_ps': self.vmax_ps,
                'vmin_rad': self.vmin_rad,
                'vmax_rad': self.vmax_rad,
                'a': self.a,
                'b': self.b
            }

            torch.save(state, os.path.join(path, 'model_' + str(epoch)))

    def sample_pointcloud(self, cond, num_points):
        '''
        sample a point cloud of "num_points" particles for given "cond" condition
        '''
        self.model.eval()
        print(self.a, self.b, self.vmin_ps, self.vmax_ps)
        with torch.no_grad():
            z = torch.randn(num_points, self.dim_input).to(self.device)
            pc_pr, _ = self.model(z, c=cond, rev=True)
            pc_pr = data_preprocessing.denormalize_point(pc_pr.to('cpu'), self.vmin_ps, self.vmax_ps, self.a, self.b)
        return pc_pr
