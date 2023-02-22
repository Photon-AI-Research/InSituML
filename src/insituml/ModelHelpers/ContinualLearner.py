from ModelHelpers.DeviceHelper import get_default_device, to_device, DeviceDataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import math

class ContinualLearner(nn.Module):
    def __init__(self, onlineEWC = False, ewc_lambda = 0.0, gamma = 0.0):
        super().__init__()
        torch.manual_seed(32)
        self.device=get_default_device()
        self.onlineEWC = onlineEWC
        self.ewc_lambda = ewc_lambda #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = gamma #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.EWC_task_count = 0

    def from_dec_to_enc(self,x):
        print("Not this one. Base class need to implement it's own.")

    #----------- EWC functions--------------------------------------#
    def estimate_fisher(self, data_set, loss_func, is_mlp = False):
        
        """
            Estimates Fisher value for each parameter based on current task

            Stores fisher value of each parameter in buffere registry on the device
            the model is being trained on.
            Additonally, registers current model parameters values.

            Parameters
            ----------
            data_set : pytorch dataset
                Current Task dataset.
            loss_func : pytorch loss_fucntion reference
                loss function to be used for calculating fisher norm

        """
        
        print("Estimating Fisher..")
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()
        
        data_loader = DataLoader(data_set, 1, num_workers = 2, pin_memory=True)
        device_data_loader = DeviceDataLoader(data_loader,self.device)

        for data, labels in device_data_loader:
            if is_mlp:
                data = data.view(1,-1)
            _,output, _ = self(data)
            loss = loss_func(output, data)
            self.zero_grad()
            loss.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
        
        print("Length DL:",len(device_data_loader))
        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/len(device_data_loader) for n, p in est_fisher_info.items()}
                        
        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task'.format(n), p.detach().clone())
                
                if self.EWC_task_count == 1:
                    # -precision (approximated by diagonal Fisher Information matrix)
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher'.format(n),
                                     est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        """
            Calculates EWC Loss based on current and previous parameter values.

            Returns
            -------
            torch tensor
                EWC Loss value

        """
        
        if self.EWC_task_count == 1:
            losses = []
            for n, p in self.named_parameters():

                if p.requires_grad:
                    # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_EWC_prev_task'.format(n))
                    fisher = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    
                    # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                    fisher = self.gamma * fisher
                    
                    # Calculate EWC-loss
                    losses.append((fisher * (p-mean)**2).sum())
            losses = to_device(losses,'cuda:0')   
            return (1./2)*sum(losses)
        else:
            return torch.tensor(0., device=self.device)
    #----------- End of EWC functions--------------------------------------#

    #----------- Start of Replayer- GEM related functions--------------------------------------#

    def calculate_ref_gradients(self, episodic_memory, loss_func, b_size, store_encoded = False):        
        
        """
            Calculates refrence gradients based on sampled memory from previous episodes

            Parameters
            ----------
            episodic_memory : set of tensors
                Data from past tasks.
            loss_func : pytorch loss_fucntion reference
                loss function to be used for calculating gradients
            b_size : int
                batch size for data loader

            Returns
            -------
            torch tensor
                Loss value for tracking between current model data output and replay memory

        """

        mode = self.training
        self.eval()
        data_set = EpisodicMemoryDataset(episodic_memory)
        data_loader = DataLoader(data_set, b_size, num_workers = 2, pin_memory=True)
        device_data_loader = DeviceDataLoader(data_loader,self.device)
        data_count = len(device_data_loader)
        loss = 0
        for data, labels in device_data_loader:
            if store_encoded:
                output = self.from_dec_to_enc(data)
            else:
                _,output, _ = self(data)
            loss += loss_func(output, data)
        loss = loss / data_count
        loss.backward()

        grad_ref = []
        # Square gradients and keep running sum
        for p in self.parameters():
            if p.requires_grad:
                grad_ref.append(p.grad.view(-1))
        grad_ref = to_device(grad_ref,'cuda:0')  
        grad_ref = torch.cat(grad_ref)
        

        self.register_buffer('Grad_Ref_Estimate',
                                     grad_ref)
        
        self.zero_grad()
        # Set model back to its initial mode
        self.train(mode=mode)
        return loss
    

    def overwrite_grad(self):

        """
            Overwrite gradients for parameter update based on dot products
            as mentioned in A-GEM method
        """

        # -reorganize gradient (of current batch) as single vector
        grad_cur = []
        for p in self.parameters():
            if p.requires_grad:
                grad_cur.append(p.grad.view(-1))
        
        grad_cur = to_device(grad_cur,'cuda:0')  
        grad_cur = torch.cat(grad_cur)

        grad_rep = getattr(self, 'Grad_Ref_Estimate')
        # -check inequality constrain
        angle = (grad_cur*grad_rep).sum()
        if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep*grad_rep).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param

    #----------- End of Replayer- GEM related functions--------------------------------------#

    #----------- Start of Replayer- LAYERWISE GEM related functions--------------------------------------#

    def calculate_ref_gradients_layerwise(self, episodic_memory, loss_func, b_size, store_encoded = False, loss_lambda = 1):        
        # Set model to evaluation mode
        mode = self.training
        self.eval()
        data_set = EpisodicMemoryDataset(episodic_memory)
        data_loader = DataLoader(data_set, b_size, num_workers = 2, pin_memory=True)
        device_data_loader = DeviceDataLoader(data_loader,self.device)
        data_count = len(device_data_loader)
        loss = 0
        for data, labels in device_data_loader:
            if store_encoded:
                output = self.from_dec_to_enc(data)
            else:
                _,output, _ = self(data)
            loss += loss_func(output, data)
        loss = (loss / data_count) * loss_lambda
        loss.backward()

        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_Grad_Ref_Estimate'.format(n),
                                     p.grad.view(-1))
        
        self.zero_grad()
        # Set model back to its initial mode
        self.train(mode=mode)
        return loss
    
    def overwrite_grad_layerwise(self):
        # -reorganize gradient (of current batch) as single vector
        grad_cur = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                grad_cur[n] = p.grad.view(-1)
          
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                grad_rep = getattr(self, '{}_Grad_Ref_Estimate'.format(n))
                # -check inequality constrain
                angle = (grad_cur[n]*grad_rep).sum()
                
                if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                    length_rep = (grad_rep*grad_rep).sum()
                    grad_proj = grad_cur[n]-(angle/length_rep)*grad_rep
                    p.grad.copy_(grad_proj.view_as(p))

    #----------- End of Replayer- GEM related functions--------------------------------------#

class EpisodicMemoryDataset(Dataset):

    def __init__(self, ep_memory):
        self.ep_memory = ep_memory
    
    def __len__(self):
        return len(self.ep_memory)
    
    def __getitem__(self, idx):
        return self.ep_memory[idx]
    