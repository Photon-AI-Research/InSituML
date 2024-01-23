import wandb
from data_loaders import Loader
from networks import ConvAutoencoder, VAE
from loss_functions import EarthMoversLoss, ChamfersLoss
import torch
import torch.nn as nn
import torch.optim as optim
import os
from trainer import train_AE

MAPPING_TO_LOSS = {
    "earthmovers":EarthMoversLoss,
    "chamfersloss":ChamfersLoss,
    "mse":nn.MSELoss
    }

MAPPING_TO_NETWORK = {
    "convAE":ConvAutoencoder,
    "VAE":VAE
    }

def train_with_wandb():
    
    hyperparameter_defaults = dict(
    t0 = 1000,
    t1 = 2001,
    timebatchsize = 4,
    particlebatchsize = 4,
    hidden_size = 1024,
    dim_pool = 1,
    lr = 0.001,
    num_epochs = 20000,
    activation = 'relu',
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
    loss_function = "earthmovers",
    loss_function_params = {'reduction':'mean', 'p': 2},
    network ="VAE"
    )
    
    print('New session...')
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    start_epoch = 0
    
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    
    pathpattern1 = config["pathpattern1"]
    pathpattern2 = config["pathpattern2"]

    
    data_loader = Loader(pathpattern1=pathpattern1,
                         pathpattern2=pathpattern2,
                         t0=config["t0"], t1=config["t1"],
                         timebatchsize=config["timebatchsize"],
                         particlebatchsize=config["particlebatchsize"])
    
        
    criterion = MAPPING_TO_LOSS[config["loss_function"]](**config["loss_function_params"])
    
    # Initialize the convolutional autoencoder
    model = MAPPING_TO_NETWORK[config["network"]](criterion)

    epoch = data_loader[0]
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    
    directory ='./checkpoints/'+str(wandb.run.id)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    
    train_AE(model, criterion, optimizer, scheduler, epoch, wandb) 
    
if __name__ == "__main__":
    train_with_wandb()
