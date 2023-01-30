import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import time
import pytorch_ssim

from ModelsEnum import ModelsEnum
from ModelHelpers.AutoEncoder2D import AutoEncoder2D
from ModelHelpers.Autoencoder3D import AutoEncoder3D
from ModelHelpers.DeviceHelper import get_default_device, to_device
# Does not exist.
# from ModelHelpers.DimensionAutoEncoderModelWithPool import DimensionAutoEncoderModelWithPool
from ModelHelpers.mlp import MLP

class Trainer():
    def __init__(self, model_path, model_loss_func, input_channels, number_model_layers, number_conv_layers ,filters, latent_size, epochs, learning_rate, run_name, input_sizes, saveModelInterval ,model_type = ModelsEnum.Autoencoder2D,activation = "leaky_relu", optimizer = "adam",batch_size = 3, onlineEWC = False, ewc_lambda = 0.0, gamma = 0.0, mas_lambda = 0.0, agem_l_enc_lambda = 1):
        self.device = get_default_device()
        self.l1_loss = nn.L1Loss()
        self.model_path = model_path
        self.input_sizes = input_sizes
        self.save_model_interval = saveModelInterval
        self.input_channels = input_channels
        self.model_type = model_type
        self.is_mlp = True if self.model_type == ModelsEnum.MLP else False
        self.ssim_loss = pytorch_ssim.SSIM3D(window_size = 11)
        self.loss_func = self._get_loss_func(model_loss_func)
        self.number_model_layers = number_model_layers
        self.number_conv_layers = number_conv_layers
        self.filters = filters
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.run_name = run_name
        self.onlineEWC = onlineEWC
        self.ewc_lambda = ewc_lambda
        self.mas_lambda = mas_lambda
        self.agem_l_enc_lambda = agem_l_enc_lambda
        self.gamma = gamma
        self.activation = self._get_activation(activation)
        self.model, self.opt_kwargs = self._init_model()
        self.opt = optimizer
        self.optimizer = self._init_optimizer(optimizer, self.opt_kwargs)
        self.elp_training_time = []
        self._send_model_to_device()
        
    def _init_model(self):
        if self.model_type == ModelsEnum.Autoencoder2D:
            model_class = AutoEncoder2D
        elif self.model_type == ModelsEnum.Autoencoder3D:
            model_class = AutoEncoder3D
        elif self.model_type == ModelsEnum.MLP:
            model_class = MLP
        else:
            model_class = DimensionAutoEncoderModelWithPool
        model = model_class(self.input_channels,self.input_sizes,self.number_model_layers, self.number_conv_layers,self.filters,self.latent_size,self.activation,self.onlineEWC, self.ewc_lambda, self.gamma)
        model.apply(model._weights_init)
        return model, {}
    
    def _send_model_to_device(self):
        if self.model_type == ModelsEnum.Autoencoder3D:
            self.model.split_gpus()
        else:
            to_device(self.model, self.device)

    def _custom_loss(self, predicted, actual):
        l1loss = F.l1_loss(predicted, actual)
        ssim_loss = self.ssim_loss(predicted, actual)
        return l1loss + (1 - ssim_loss)


    def _get_loss_func(self, loss):
        print(loss)
        if loss == 'L1':
            return nn.L1Loss()
        if loss == 'custom':
            return self._custom_loss

        return nn.MSELoss()
    
    def _get_activation(self, activation):
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        return nn.ReLU()
    
    def _init_optimizer(self, optimizer, opt_kwargs):
        # If `--lr` is 0, use the learning rate from `opt_kwargs`.
        if 'lr' in opt_kwargs and self.learning_rate == 0:
            opt_kwargs = opt_kwargs.copy()
            learning_rate = opt_kwargs.pop('lr')
        else:
            assert self.learning_rate != 0
            learning_rate = self.learning_rate

        if optimizer == "sgd":
            return SGD(self.model.parameters(), lr=learning_rate, **opt_kwargs)
        elif optimizer == "adam":
            return Adam(self.model.parameters(), lr=learning_rate, **opt_kwargs)
        else:
            raise ValueError('unknown optimizer')
    
    def _reset_optimizer(self):
        self.optimizer = self._init_optimizer(self.opt, self.opt_kwargs)

    def _save_model(self, name, path, after_task):
        self.model.save_checkpoint(path,name, after_task)
    
    def _modify_batch(self,batch, reverse = False):
        batch_size = batch.size(0)
        if reverse:
            return batch.view(batch_size,self.input_channels,self.input_sizes[0],self.input_sizes[1])
        else:
            return batch.view(batch_size,-1)
    
    def _infinity_norm_loss(self, actual, target):
        inf_norm_actual = torch.max(torch.sum(torch.abs(actual), dim = 1))
        inf_norm_target = torch.max(torch.sum(torch.abs(target), dim = 1))
        return torch.abs(inf_norm_actual - inf_norm_target)
