from ModelHelpers.DeviceHelper import get_default_device, to_device , DeviceDataLoader
from ModelHelpers.DimensionAutoEncoderModel import DimensionAutoEncoderModel
from ModelHelpers.DimensionAutoEncoderModelWithPool import DimensionAutoEncoderModelWithPool
from ModelHelpers.MeshDimensionDataset import MeshDimensionDataset
from ModelHelpers.PlotHelper import prepare_data_for_plot
from torch.utils.data.dataloader import DataLoader
from ModelsEnum import ModelsEnum
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

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
        
        if self.model_enum is ModelsEnum.Autoencoder_Pooling:
            model = DimensionAutoEncoderModelWithPool(input_channels, input_sizes, loss_func, n_layers,n_conv_layers, filters, latent_size, act)
        elif self.model_enum is ModelsEnum.Autoencoder_Sampling:
            model = DimensionAutoEncoderModel(input_channels, shape, loss_func, n_layers, n_conv_layers, filters, latent_size, act)
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
            data = np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_{}.npy".format(iteration_id))
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
        data_f = [np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_2000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_4000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_6000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_8000.npy")]
        
        
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
                
            
    
if __name__== "__main__":    
    print("Here")
    with(wandb.init(project="streamed_ml")):
        
        device = get_default_device()
        model_20 = load_model("electric-sponge-670_21")
        model_40 = load_model("electric-sponge-670_41")
        model_60 = load_model("electric-sponge-670_61")
        model_80 = load_model("electric-sponge-670_81")
        
        to_device(model_20,device)
        to_device(model_40,device)
        to_device(model_60,device)
        to_device(model_80,device)
        data_f = [np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_2000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_4000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_6000.npy"),
                np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_8000.npy")]
        op_20 = None
        op_40 = None
        op_60 = None
        op_80 = None
        for i,data in enumerate(data_f):
            data_set = MeshDimensionDataset(2000,  (128,1280,128), data,  None, -2.1393063, 2.140939, 1)
            data_loader = DataLoader(data_set, 1, num_workers = 2, pin_memory=True)
            device_data_loader = DeviceDataLoader(data_loader,device)
            with torch.no_grad():
                for batch in device_data_loader:
                        _,op_20 = model_20(batch)
                        _,op_40 = model_40(batch)
                        _,op_60 = model_60(batch)
                        _,op_80 = model_80(batch)
        
            max_norm = 2.140939
            min_norm = -2.1393063
            predicted_20 = op_20.detach().cpu().numpy()[0]
            #predicted_20 = predicted_20 * (max_norm - min_norm) + min_norm
            predicted_40 = op_40.detach().cpu().numpy()[0]
            #predicted_40 = predicted_40 * (max_norm - min_norm) + min_norm
            predicted_60 = op_60.detach().cpu().numpy()[0]
            #predicted_60 = predicted_60 * (max_norm - min_norm) + min_norm
            predicted_80 = op_80.detach().cpu().numpy()[0]
            #predicted_80 = predicted_80 * (max_norm - min_norm) + min_norm

            fig, (ax1,ax2,ax3, ax4,ax5) = plt.subplots(1,5, figsize=(35,15))
            #fig.suptitle("Prediction Epoch {}".format(task_id))
            a = ax1.imshow(data[1][32], cmap ="jet")
            ax1.set_title("Original", fontsize=30)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)

            b = ax2.imshow(predicted_20[0][32], cmap ="jet")
            #b = ax2.imshow(data2[1][32], cmap ="jet")
            ax2.set_title("Model State - 20", fontsize=30)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)

            c = ax3.imshow(predicted_40[0][32], cmap ="jet")
            #c = ax3.imshow(data3[1][32], cmap ="jet")
            ax3.set_title("Model State - 40", fontsize=30)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)

            d = ax4.imshow(predicted_60[0][32], cmap ="jet")
            #d = ax4.imshow(data4[1][32], cmap ="jet")
            ax4.set_title("Model State - 60", fontsize=30)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)

            e = ax5.imshow(predicted_80[0][32], cmap ="jet")
            ax5.set_title("Model State - 80", fontsize=30)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)

            ax1_cbar = fig.colorbar(a,ax =ax1)
            ax2_cbar = fig.colorbar(b,ax =ax2)
            ax3_cbar = fig.colorbar(c,ax =ax3)
            ax4_cbar = fig.colorbar(d,ax =ax4)
            ax5_cbar = fig.colorbar(e,ax =ax5)

            ax1_cbar.ax.tick_params(labelsize=30)
            ax2_cbar.ax.tick_params(labelsize=30) 
            ax3_cbar.ax.tick_params(labelsize=30) 
            ax4_cbar.ax.tick_params(labelsize=30)
            ax5_cbar.ax.tick_params(labelsize=30)

            fig.suptitle("Task {}".format((i+1) * 20), fontsize=60)

            wandb.log({"Report Images {}".format((i+1) * 20):fig})
#         me = ModelEvaluator(None,None,None,"/home/h5/vama551b/home/streamed-ml/StreamedML/main/Model/","upbeat-cloud-523",0,0)
#         print("okay")
#         me.image_show()