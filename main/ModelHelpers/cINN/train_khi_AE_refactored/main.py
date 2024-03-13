import wandb
from data_loaders import TrainLoader, ValidationFixedBoxLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
from trainer import train_AE
import argparse
from datetime import datetime
from args_transform import main_args_transform
from utilities import Normalizer

def train_with_wandb():
    
    hyperparameter_defaults = dict(
    t0 = 1000,
    t1 = 2001,
    timebatchsize = 4,
    particlebatchsize = 4,
    hidden_size = 1024,
    dim_pool = 1,
    learning_rate = args.learning_rate,
    num_epochs = 5,
    val_boxes =  [19,5,3],
    activation = 'relu',
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
    loss_function = args.lossfunction,
    loss_function_params = {}
    )

    hyperparameter_defaults.update(vars(args))
    
    hyperparameter_defaults = main_args_transform(hyperparameter_defaults)
    
    print('New session...')
    hd = hyperparameter_defaults
    info_image_path = f"encoder_kwargs_{hd['encoder_kwargs']}_decoder_kwargs_{hd['decoder_kwargs']}"
        
    # Pass your defaults to wandb.init
    run = wandb.init(config=hyperparameter_defaults, project=f'newruns_{args.project_kw}', name=info_image_path)
    start_epoch = 0
    
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    
    pathpattern1 = config["pathpattern1"]
    pathpattern2 = config["pathpattern2"]
    
    data_loader = TrainLoader(pathpattern1=pathpattern1,
                            pathpattern2=pathpattern2,
                            t0=config["t0"],
                            t1=config["t1"],
                            timebatchsize=config["timebatchsize"],
                            particlebatchsize=config["particlebatchsize"],
                            blacklist_box = config["val_boxes"],
                            particles_to_sample = config["particles_to_sample"],
                            normalisation = Normalizer(),
                            norm_method = 'mean_6d')
    # data_loader = TrainLoader(pathpattern1=pathpattern1,
    #                           pathpattern2=pathpattern2,
    #                           t0=config["t0"], t1=config["t1"],
    #                           timebatchsize=config["timebatchsize"],
    #                           particlebatchsize=config["particlebatchsize"],
    #                           blacklist_box = config["val_boxes"], 
    #                           particles_to_sample = config["particles_to_sample"])
    
    valid_data_loader = ValidationFixedBoxLoader(pathpattern1,
                                                 pathpattern2,
                                                 config["val_boxes"],
                                                 t0=config["t0"],
                                                 t1=config["t1"],
                                                 particles_to_sample = config["particles_to_sample"],
                                                 normalisation = Normalizer(),
                                                 norm_method = 'mean_6d')
            
    # Initialize the convolutional autoencoder
    model = hyperparameter_defaults["model"]
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
    
    train_AE(model, optimizer, scheduler, epoch, valid_data_loader, wandb, directory,
             property_= args.property_) 
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""For running Autoencoders on khi data"""
    )
    
    parser.add_argument('--property_',
                        type=str,
                        default='momentum_force',
                        help="Whether to train on positions, momentum, forces or all")

    parser.add_argument('--learning_rate',
                        type=float,
                        default='1e-3',
                        help="Set the learning rate")

    parser.add_argument('--z_dim',
                        type=int,
                        default='128',
                        help="Set the latent space dimensions")

    parser.add_argument('--num_epochs',
                        type=int,
                        default='5',
                        help="Number of epochs")
    
    parser.add_argument('--timebatchsize',
                        type=int,
                        default='4',
                        help="Set the timebatchsize")
    
    parser.add_argument('--particlebatchsize',
                        type=int,
                        default='2',
                        help="Set the particlebatchsize")
    
    parser.add_argument('--val_boxes',
                        type=str,
                        default='[19,5,3]',
                        help="Validation boxes")
    
    parser.add_argument('--lossfunction',
                        type=str,
                        default='chamfersloss',
                        help="Choose the loss function")

    parser.add_argument('--network',
                        type=str,
                        default='VAE',
                        help="Choose the loss function")
    
    parser.add_argument('--project_kw',
                        type=str,
                        default='',
                        help="Choose the project keyword for runs")
    
    parser.add_argument('--ae_config',
                        type=str,
                        default="deterministic",
                        help="Three choices for encoder config: simple, non_deterministic, or deterministic")

    parser.add_argument('--use_encoding_in_decoder',
                        type=bool,
                        default=False,
                        help="Whether to use encodings in the decoder or otherwise")

    parser.add_argument('--particles_to_sample',
                        type=int,
                        default=4000,
                        help="How many particles to sample.")

    parser.add_argument('--pathpattern1',
                        type=str,
                        default="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
                        help="Path to the particles data")

    parser.add_argument('--pathpattern2',
                        type=str,
                        default= "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
                        help="Path to radiation data")

    parser.add_argument('--t0',
                        type=int,
                        default = 1994,
                        help="Start time step from the data")

    parser.add_argument('--t1',
                        type=int,
                        default = 2001,
                        help="Last time step from the data")

    parser.add_argument('--encoder_type',
                        type=str,
                        default = "encoder_simple",
                        help="Kind of Encoder")

    parser.add_argument('--encoder_kwargs',
                        type=str,
                        default = '{"z_dim":128,"input_dim":6,"ae_config":"deterministic"}',
                        help="Encoder keyword arguments")

    parser.add_argument('--decoder_type',
                        type=str,
                        default = "mlp_decoder",
                        help="Kind of Decoder")

    parser.add_argument('--decoder_kwargs',
                        type=str,
                        default = '{"z_dim":128, "particles_to_sample":4000, "input_dim":6}',
                        help="Decoder keyword arguments")
    
    args = parser.parse_args()
    
    train_with_wandb()
