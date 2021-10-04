import wandb
from ModelHelpers.DeviceHelper import to_device
from ModelHelpers.DimensionAutoEncoderModelWithPool import DimensionAutoEncoderModelWithPool
from ModelHelpers.DimensionAutoEncoderModel import DimensionAutoEncoderModel
from ModelHelpers.DeviceHelper import DeviceDataLoader
from torch.utils.data.dataloader import DataLoader
from ModelTrainerTaskWise import ModelTrainerTaskWise
import torch
import numpy as np
from torch.utils.data import Dataset

class EfieldDataset(Dataset):
    def __init__(self, num_tasks):
        self.iterations = list(range(num_tasks))
    
    def __len__(self):
        return len(self.iterations)
    
    def __getitem__(self, idx):
        iteration_id = self.iterations[idx]
        data = np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_{}.npy".format(iteration_id * 100))
        data = data[1]
        dim = (128,1280,128)
        
        norm_tensor = torch.from_numpy(data)
        #norm_tensor = torch.sub(norm_tensor, self.min_norm)
        #norm_tensor = torch.div(norm_tensor, self.max_norm - self.min_norm)
        return norm_tensor.view(-1,dim[0],dim[1],dim[2])

class ModelTrainerOffline(ModelTrainerTaskWise):

    def __init__(self, model_path, model_loss_func, number_model_layers, number_conv_layers, filters, latent_size, epochs, learning_rate, run_name, min_norm, max_norm, activation, optimizer, batch_size, onlineEWC, ewc_lambda, gamma):
        super().__init__(model_path, model_loss_func, number_model_layers, number_conv_layers, filters, latent_size, epochs, learning_rate, run_name, min_norm, max_norm, activation=activation, optimizer=optimizer, batch_size=batch_size, onlineEWC=onlineEWC, ewc_lambda=ewc_lambda, gamma=gamma)


    def train_tasks(self, number_of_tasks, on_fft = False, fft_prop = None):
        print("Offline Training")
        dataset = EfieldDataset(number_of_tasks)
        data_loader = DataLoader(dataset, self.batch_size, num_workers = 2, pin_memory=True, shuffle=True)
        device_data_loader = DeviceDataLoader(data_loader,self.device)
        
        dim = (128,1280,128)

        if self.model is None:
            self.model = DimensionAutoEncoderModel(1, dim, self.model_loss_func, self.number_model_layers, self.number_conv_layers, self.filters, self.latent_size, self.activation, onlineEWC = self.onlineEWC, ewc_lambda = self.ewc_lambda, gamma = self.gamma)
            # self.model = DimensionAutoEncoderModelWithPool(1,dim, self.model_loss_func, self.number_model_layers, self.number_conv_layers, self.filters, self.latent_size, self.activation, onlineEWC = self.onlineEWC, ewc_lambda = self.ewc_lambda, gamma = self.gamma)
            to_device(self.model,self.device)
            self.model.apply(self.model._weights_init)
            optimizer = self._init_optimizer()
            wandb.watch(self.model, log="all")
        
        for epoch in range(self.epochs):
            losses = []
            for x in device_data_loader:
                optimizer.zero_grad()
                loss , ewc_loss = self.model.training_step(x)#, ewc_loss = self.model.training_step(batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                    
            wandb.log({
                        "loss_model": sum(losses) / len(losses),
                        "epoch": epoch
                        })
        
            self.save_model(self.run_name,self.model_path, number_of_tasks)
