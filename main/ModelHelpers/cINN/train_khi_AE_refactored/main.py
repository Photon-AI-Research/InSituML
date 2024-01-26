import wandb
from data_loaders import Loader
from networks import ConvAutoencoder, VAE
from loss_functions import EarthMoversLoss, ChamfersLoss
import torch
import torch.nn as nn
import torch.optim as optim
import os
from trainer import train_AE
import argparse
from datetime import datetime

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
    learning_rate = 0.001,
    num_epochs = 1,
    activation = 'relu',
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
    loss_function = "chamfersloss",
    loss_function_params = {},
    network ="VAE",
    z_dim = 4
    )
    
    point_dim = 9 if property_ == "all" else 3
    
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

    info_image_path = f"lr_{config['learning_rate']}_z_{config['z_dim']}_lf_{config['loss_function']}"
    os.mkdir(info_image_path)
        
    criterion = MAPPING_TO_LOSS[config["loss_function"]](**config["loss_function_params"])
    
    # Initialize the convolutional autoencoder
    model = MAPPING_TO_NETWORK[config["network"]](criterion, point_dim, config["z_dim"])

    epoch = data_loader[0]
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1000,
                                                gamma=0.9)

    
    directory ='./checkpoints/'+str(wandb.run.id)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    
    train_AE(model, criterion, optimizer, scheduler, epoch, wandb, directory,
             property_= property_, info_image_path=info_image_path) 
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""For running Autoencoders on khi data"""
    )
    
    parser.add_argument('--property_',
                        type=str,
                        default='positions',
                        help="Whether to train on positions, momentum, forces or all")
    
    property_ = parser.parse_args().property_
    
    sweep_config = {
    'method': 'random',
    'parameters':{
        'loss_function': {
            'values': ["earthmovers", "chamfersloss"]
            },
        'learning_rate': {
            'values': [1e-2, 1e-5]
            },
        'z_dim': {
            'values': [5, 10, 15]
            }
        }
        }

    time_now = datetime.now().strftime("%H:%M").replace(":","_")
            
    sweep_id = wandb.sweep(sweep_config, project=f"khi_vae_{property_}_{time_now}")
    
    wandb.agent(sweep_id, train_with_wandb)
