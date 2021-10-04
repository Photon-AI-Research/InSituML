from StreamDataReader.StreamBuffer import StreamBuffer
from ModelHelpers.DeviceHelper import get_default_device, to_device , DeviceDataLoader
from ModelHelpers.DimensionAutoEncoderModel import DimensionAutoEncoderModel
from ModelHelpers.DimensionAutoEncoderModelWithPool import DimensionAutoEncoderModelWithPool
from ModelHelpers.MeshDimensionDataset import MeshDimensionDataset
from ModelHelpers.PlotHelper import prepare_data_for_plot
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, SGD
import torch.nn as nn
import wandb
import os
import numpy as np
import time
from threading import Thread

class ModelTrainerTaskWise():
    def __init__(self, model_path, model_loss_func, number_model_layers, number_conv_layers ,filters, latent_size, epochs, learning_rate, run_name, min_norm, max_norm,activation = "leaky_relu", optimizer = "sgd",batch_size = 3, onlineEWC = False, ewc_lambda = 2.0, gamma = 0.90):
        
        self.streamBuffer = StreamBuffer(buffer_size = batch_size)
        self.device = get_default_device()
        self.model_path = model_path
        self.model = None
        self.model_loss_func = self._get_loss_func(model_loss_func)
        self.number_model_layers = number_model_layers
        self.number_conv_layers = number_conv_layers
        self.filters = filters
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.run_name = run_name
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.onlineEWC = onlineEWC
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.activation = self._get_activation(activation)
        self.optimizer = optimizer
        self.elp_training_time = []
    
    def _get_loss_func(self, loss):
        if loss is 'L1':
            return nn.L1Loss()
        return nn.MSELoss()
    
    def _get_activation(self, activation):
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        return nn.ReLU()
    
    def _init_optimizer(self):
        if self.optimizer == "sgd":
            return SGD(self.model.parameters(), lr = self.learning_rate)
        
        return Adam(self.model.parameters(), lr = self.learning_rate)
    
    def save_model(self, name, path, after_task):
        self.model.save_checkpoint(path,name, after_task)
    
    def train_tasks(self, number_of_tasks, save_model_interval):
        val_data_set, val_data_loader, val_device_data_loader = None, None, None
        optimizer = None
        iterations, rem = divmod(number_of_tasks, self.batch_size)
        for i in range(1, iterations + 2):
            data_dict = self.streamBuffer.read_buffer()
            #handle empty iterations
            if data_dict is None:
                print("Didn't find anything in buffer")
                break
            print("iterations ", data_dict[0])
            iteration_id = max(data_dict[0])
            if i == iterations + 1 and rem > 0:
                data = data_dict[1][:rem]
            else:
                data = data_dict[1]
            shape = data_dict[2]
            #initializing everything first iteration
            if self.model is None:
#                 self.model = DimensionAutoEncoderModel(1, shape[1:], self.model_loss_func, self.number_model_layers, self.number_conv_layers, self.filters, self.latent_size, self.activation, onlineEWC = self.onlineEWC, ewc_lambda = self.ewc_lambda, gamma = self.gamma)
                self.model = DimensionAutoEncoderModelWithPool(1, shape[1:], self.model_loss_func, self.number_model_layers, self.number_conv_layers, self.filters, self.latent_size, self.activation, onlineEWC = self.onlineEWC, ewc_lambda = self.ewc_lambda, gamma = self.gamma)

                to_device(self.model,self.device)
                self.model.apply(self.model._weights_init)
                optimizer = self._init_optimizer()
                wandb.watch(self.model, log="all")

            data_set = MeshDimensionDataset(iteration_id, shape[1:], data,  None, self.min_norm, self.max_norm, self.batch_size)
            data_loader = DataLoader(data_set, self.batch_size, num_workers = 2, pin_memory=True)
            device_data_loader = DeviceDataLoader(data_loader,self.device)
            
            fill_buff_t = Thread(target=self.streamBuffer.fill_buffer)
            fill_buff_t.start()
            start = time.time()
            print("Started Fitting----------Batch ", i)
            output = None
            for batch in device_data_loader:
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    loss , ewc_loss = self.model.training_step(batch)#, ewc_loss = self.model.training_step(batch)
                    loss.backward()
                    optimizer.step()
                    
                    wandb.log({
                               "loss_model": loss.item(),
                               "batch": i
                              })
                    if self.onlineEWC:
                        wandb.log({
                                   "ewc_loss": ewc_loss.item(),
                                   "batch": i
                                  })
                        
            if self.onlineEWC:
                self.model.estimate_fisher(data_set)
            
            if i % save_model_interval == 0:
                self.save_model(self.run_name,self.model_path, i)
            if i % save_model_interval == 0:
                with torch.no_grad():
                    _ , op = self.model(batch)
                    prepare_data_for_plot(i,batch, op,'y',32, data_set.min_norm, data_set.max_norm)
            fill_buff_t.join()
            self.elp_training_time.append(time.time() - start)
            print("Finished Fitting----------Batch", i)