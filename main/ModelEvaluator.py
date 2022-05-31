from ModelHelpers.DeviceHelper import get_default_device, to_device , DeviceDataLoader
from ModelHelpers.Autoencoder3D import AutoEncoder3D
from ModelHelpers.MeshDimensionDataset import MeshDimensionDataset
from ModelHelpers.PlotHelper import prepare_data_for_plot
from torch.utils.data.dataloader import DataLoader
from ModelsEnum import ModelsEnum
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt

from main.ModelHelpers.AutoEncoder2D import AutoEncoder2D

class ModelEvaluator():
    
    def __init__(self, model_path, model_name, task_intervals, model_enum):
        self.model_path = model_path
        self.device = get_default_device()
        self.model_name = model_name
        self.model_enum = model_enum
        self.task_intervals = task_intervals
        self.models = self._init_all_models()
    
    def _init_model(self, model_name):
        model_state = torch.load(self.model_path + model_name)
        if model_state is None:
            return None
        input_channels = model_state["input_channels"]
        input_sizes = model_state["input_sizes"]
        loss_func = model_state["loss_func"]
        n_layers = model_state["n_layers"]
        shape = (128,1280,128)
        n_conv_layers = model_state["n_conv_layers"]
        latent_size = model_state["latent_size"]
        act = model_state["act"]
        filters = model_state["filters"]
        
        if self.model_enum is ModelsEnum.Autoencoder3D:
            model = AutoEncoder3D(input_channels, input_sizes, loss_func, n_layers,n_conv_layers, filters, latent_size, act)
        elif self.model_enum is ModelsEnum.Autoencoder2D:
            model = AutoEncoder2D(input_channels, shape, loss_func, n_layers, n_conv_layers, filters, latent_size, act)
        else:
            return None
            
        model.load_state_dict(model_state['model'])
        
        return model
    
    def _init_all_models(self):
        models = []
        for interval in self.task_intervals:
            models.append(self._init_model(self.model_name + "_" + str(interval)))
            
        return models
    
    def evaluate(self, number_of_tasks = 100):
        #send all models to device and eval mode
        for model in self.models:
            if model is None:
                continue
            model.eval()
            to_device(model,self.device)
            
        for i in range(1, number_of_tasks + 2):
            iteration_id = (i-1) * 100
            """ 
                change data path accordingly 
            """ 
            data = np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_{}.npy".format(iteration_id))
            data = np.expand_dims(data, axis=0)
            shape = (128,1280,128)
            
            data_set = MeshDimensionDataset(iteration_id, shape, data,  None, None, None, 1)
            data_loader = DataLoader(data_set, 1, num_workers = 2, pin_memory=True)
            device_data_loader = DeviceDataLoader(data_loader,self.device)
            
            with torch.no_grad():
                for batch in device_data_loader:
                    for idx, interval in enumerate(self.task_intervals):
                        if self.models[idx] is None:
                            continue
                        
                        loss = self.models[idx].validation_step(batch)

                        wandb.log({
                                    "val_loss_"+str((idx+1)*20): loss.item(),
                                    "task": i
                                  })
    
    
    def image_show(self):
        data_f = [np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_2000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_4000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_6000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_8000.npy")]
        
        
        for i,data in enumerate(data_f):
            op_models = []
            data_set = MeshDimensionDataset(2000,  (128,1280,128), np.expand_dims(data, axis=0),  None, -2.1393063, 2.140939, 1)
            data_loader = DataLoader(data_set, 1, num_workers = 2, pin_memory=True)
            device_data_loader = DeviceDataLoader(data_loader,self.device)
            with torch.no_grad():
                for batch in device_data_loader:
                    for model in self.models:
                        if model is not None:
                            _,op = model(batch)
                            op_models.append(op)
            fig, ax = plt.subplots(1,len(op_models) + 1, figsize=(7*(1+len(op_models)),15))
            fig.suptitle("Task {}".format((i+1) * 20), fontsize=60)
            pred_axis = ax[0].imshow(data[1][32], cmap ="jet")
            ax[0].set_title("Original", fontsize=30)
            ax[0].yaxis.set_visible(False)
            ax[0].xaxis.set_visible(False)
            pred_cbar = fig.colorbar(pred_axis,ax = ax[0])
            pred_cbar.ax.tick_params(labelsize=30)
            
            for op_id, op in enumerate(op_models):
                op = op.detach().cpu().numpy()[0]
                ax[op_id + 1].imshow(op[0][32], cmap ="jet")
                ax[op_id + 1].set_title("Model State - {}".format((op_id+1) * 20), fontsize=30)
                ax[op_id + 1].yaxis.set_visible(False)
                ax[op_id + 1].xaxis.set_visible(False)
                ax_cbar = fig.colorbar(pred_axis,ax =ax[op_id + 1])
                ax_cbar.ax.tick_params(labelsize=30)
            
            wandb.log({"Report Images {}".format((i+1) * 20):fig})
