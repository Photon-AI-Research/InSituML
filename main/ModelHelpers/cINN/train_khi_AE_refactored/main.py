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
    learning_rate = args.learning_rate,
    num_epochs = 3,
    activation = 'relu',
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
    loss_function = args.lossfunction,
    loss_function_params = {},
    network ="VAE",
    z_dim = args.z_dim
    )
    
    point_dim = 9 if args.property_ == "all" else 3
    
    print('New session...')
    
    info_image_path = f"lr_{args.learning_rate}_z_{args.z_dim}_{args.property_}"
    time_now = datetime.now().strftime("%H:%M").replace(":","_")
    
    # Pass your defaults to wandb.init
    run = wandb.init(config=hyperparameter_defaults, project=f'newruns_{time_now}', name=info_image_path)
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
             property_= args.property_) 
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""For running Autoencoders on khi data"""
    )
    
    parser.add_argument('--property_',
                        type=str,
                        default='positions',
                        help="Whether to train on positions, momentum, forces or all")

    parser.add_argument('--learning_rate',
                        type=float,
                        default='1e-3',
                        help="Set the learning rate")

    parser.add_argument('--z_dim',
                        type=int,
                        default='5',
                        help="Set the latent space dimensions")
    
    parser.add_argument('--lossfunction',
                        type=str,
                        default='chamfersloss',
                        help="Choose the loss function")
    
    args = parser.parse_args()
    
    train_with_wandb()
