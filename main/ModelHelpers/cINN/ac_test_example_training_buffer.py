import time

from ks_transform_policies import *
from ks_producer_openPMD_streaming import *

from ac_train_batch_buffer import TrainBatchBuffer
from ac_consumer_trainer import ModelTrainer
from threading import Thread
import torch
from time import sleep
from random import random
from queue import Queue

from torch import optim
import torch.nn as nn

from utilities import MMD_multiscale, fit
from ks_models import PC_MAF, INNModel

from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
from train_khi_AE_refactored.networks import VAE, ConvAutoencoder
from wandb_logger import WandbLogger
import torch.multiprocessing as mp

print("Done importing modules.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

openPMDBuffer = Queue() ## Buffer shared between openPMD data loader and model

class DummyOpenPMDProducer(Thread):
    def __init__(self, openPMDBuffer):
        Thread.__init__(self)        
        self.openPMDBuffer = openPMDBuffer
    
    def run(self):
        print('Producer: Running')

        # generate openpmd stuff.
        for i in range(10):
            # generate a value
            loaded_particles = torch.rand(gpu_boxes, ps_dims, number_of_particles)
            radiation = torch.rand(gpu_boxes, rad_dims)
            # block, to simulate effort
            sleep(i)
            # create a tuple
            item = [loaded_particles, radiation]
            # add to the queue
            self.openPMDBuffer.put(item)
            # report progress
            print(f'>producer added {i}')
        # signal that there are no further items
        self.openPMDBuffer.put(None)
        print('Producer: Done')






#########################
## Model configuration ##
#########################

rad_dims = 512 # Number of frequencies in radiation data

latent_space_dims = 1024

config = dict(
dim_input = 1024,
dim_condition = rad_dims,
num_coupling_layers = 4,
hidden_size = 256,
num_blocks_mat = 6,
activation = 'relu',
lr = 0.00001,
y_noise_scale = 1e-1,
zeros_noise_scale = 5e-2,
lambd_predict = 3.,
lambd_latent = 300.,
lambd_rev = 400.,
ndim_tot = 1024,
ndim_x = 1024,
ndim_y = 512,
ndim_z = 512,
load_model = '2vsik2of'
)

config_inn = dict(

)

class ModelFinal(nn.Module):
    def __init__(self,
                 base_network,
                 inner_model,
                 loss_function_IM = None,
                 weight_AE=1.0,
                 weight_IM=1.0):
        super().__init__()

        self.base_network = base_network
        self.inner_model = inner_model
        self.loss_function_IM = loss_function_IM
        self.weight_AE = weight_AE
        self.weight_IM = weight_IM

    def forward(self, x, y):

        loss_AE,loss_ae_reconst,kl_loss, _, encoded = self.base_network(x)

        # Check if the inner model is an instance of INNModel
        if isinstance(self.inner_model, INNModel):
            # Use the compute_losses function of INNModel
            loss_IM, l_fit,l_latent,l_rev = self.inner_model.compute_losses(encoded, y)
            total_loss = loss_AE*self.weight_AE + loss_IM*self.weight_IM

            losses = {
                'total_loss': total_loss,
                'loss_AE': loss_AE*self.weight_AE,
                'loss_IM': loss_IM*self.weight_IM,
                'loss_ae_reconst': loss_ae_reconst,
                'kl_loss': kl_loss,
                'l_fit': l_fit,
                'l_latent': l_latent,
                'l_rev': l_rev,
                    }

            return losses
        else:
            # For other types of models, such as MAF
            loss_IM = self.inner_model(inputs=encoded, context=y)
            total_loss = loss_AE*self.weight_AE + loss_IM * self.weight_IM

            losses = {
                'total_loss': total_loss,
                'loss_AE': loss_AE*self.weight_AE,
                'loss_IM': loss_IM*self.weight_IM,
                'loss_ae_reconst': loss_ae_reconst,
                'kl_loss': kl_loss
                    }

            return losses

    def reconstruct(self,x, y, num_samples = 1):

        if isinstance(self.inner_model, INNModel):
            lat_z_pred = self.inner_model(x, y, rev = True)
            y = self.base_network.decoder(lat_z_pred)
        else:
            lat_z_pred = self.inner_model.sample_pointcloud(num_samples = num_samples, cond=y)
            y = self.base_network.decoder(lat_z_pred)

        return y, lat_z_pred


VAE_encoder_kwargs = {"ae_config":"non_deterministic",
                   "z_dim":latent_space_dims,
                   "input_dim":ps_dims,
                   "conv_layer_config":[16, 32, 64, 128, 256, 512, 1024],
                   "conv_add_bn": False,
                   "fc_layer_config":[1024]}

VAE_decoder_kwargs = {"z_dim":latent_space_dims,
                   "input_dim":ps_dims,
                   "initial_conv3d_size":[16, 4, 4, 4],
                   "add_batch_normalisation":False,
                    "fc_layer_config":[1024]}
def load_things():

    VAE = VAE(encoder = Encoder,
            encoder_kwargs = VAE_encoder_kwargs,
            decoder = Conv3DDecoder,
            z_dim=latent_space_dims,
            decoder_kwargs = VAE_decoder_kwargs,
            loss_function = EarthMoversLoss(),
            property_="momentum_force",
            particles_to_sample = number_of_particles,
            ae_config="non_deterministic",
            use_encoding_in_decoder=False)


    conv_AE_encoder_kwargs = {"ae_config":"simple",
                    "z_dim":latent_space_dims,
                    "input_dim":ps_dims,
                    "conv_layer_config":[16, 32, 64, 128, 256, 512],
                    "conv_add_bn": False}

    conv_AE_decoder_kwargs = {"z_dim":latent_space_dims,
                    "input_dim":ps_dims,
                    "add_batch_normalisation":False}

    conv_AE = ConvAutoencoder(encoder = Encoder,
                            encoder_kwargs = conv_AE_encoder_kwargs,
                            decoder = Conv3DDecoder,
                            decoder_kwargs = conv_AE_decoder_kwargs,
                            loss_function = EarthMoversLoss(),
                            )

    inner_model = PC_MAF(dim_condition=config["dim_condition"],
                            dim_input=config["dim_input"],
                            num_coupling_layers=config["num_coupling_layers"],
                            hidden_size=config["hidden_size"],
                            device=device,
                            num_blocks_mat = config["num_blocks_mat"],
                            activation = config["activation"]
                            )

    # INN
    inner_model = INNModel(ndim_tot=config["ndim_tot"],
                    ndim_x=config["ndim_x"],
                    ndim_y=config["ndim_y"],
                    ndim_z=config["ndim_z"],
                    loss_fit=fit,
                    loss_latent=MMD_multiscale,
                    loss_backward=MMD_multiscale,
                    lambd_predict=config["lambd_predict"],
                    lambd_latent=config["lambd_latent"],
                    lambd_rev=config["lambd_rev"],
                    zeros_noise_scale=config["zeros_noise_scale"],
                    y_noise_scale=config["y_noise_scale"],
                    hidden_size=config["hidden_size"],
                    activation=config["activation"],
                    num_coupling_layers=config["num_coupling_layers"],
                    device = device)

    #model = ModelFinal(VAE, inner_model, EarthMoversLoss())
    #model = ModelFinal(conv_AE, inner_model, EarthMoversLoss())
    model = ModelFinal(VAE, inner_model, EarthMoversLoss())


    #Load a pre-trained model
    filepath = 'trained_models/{}/best_model_'

    original_state_dict = torch.load(filepath.format(config["load_model"]))
    updated_state_dict = {key.replace('VAE.', 'base_network.'): value for key, value in original_state_dict.items()}
    model.load_state_dict(updated_state_dict)
    print('Loaded pre-trained model successfully')


    #model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    #wandb_logger = WandbLogger(project="khi_public",args=config, entity='jeyhun')
    return optimizer, scheduler, model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def demo_basic(rank, world_size):

    setup(rank, world_size)

    optimizer, scheduler, model = load_things()

    timeBatchLoader = DummyOpenPMDProducer(openPMDBuffer)
    trainBF = TrainBatchBuffer(openPMDBuffer)
    modelTrainer = ModelTrainer(trainBF, model, optimizer, scheduler, logger = None)

    ####################
    ## Start training ##
    ####################
    start_time = time.time()

    modelTrainer.start()
    trainBF.start()
    timeBatchLoader.start()


    modelTrainer.join()
    print("Join model trainer")
    stdout.flush()

    trainBF.join()
    print("Join continual learning buffer")
    stdout.flush()
    timeBatchLoader.join()
    print("Join openPMD data loader")
    stdout.flush()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


n_gpus = torch.cuda.device_count()
assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
world_size = n_gpus
run_demo(demo_basic, world_size)
