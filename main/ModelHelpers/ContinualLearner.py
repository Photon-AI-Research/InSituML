from ModelHelpers.DeviceHelper import get_default_device, to_device, DeviceDataLoader
import torch
import torch.nn as nn
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
    
    def estimate_fisher(self, data_set):
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

        
        for data in device_data_loader:
            _,output = self(data)
            loss = self.loss_func(output, data)
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
        #Calculate EWC loss
        
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
                    
            return (1./2)*sum(losses)
        else:
            return torch.tensor(0., device=self.device)