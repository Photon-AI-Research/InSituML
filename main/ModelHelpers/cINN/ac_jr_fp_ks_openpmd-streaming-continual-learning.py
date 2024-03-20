"""
Main file/module to train ML model from PIConGPU openPMD data using streaming and threads.
"""
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

print("Done importing modules.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

openPMDBuffer = Queue() ## Buffer shared between openPMD data loader and model


#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 100

normalization_values = dict(
    momentum_mean = 0.,
    momentum_std = 1.,
    force_mean = 0.,
    force_std = 1.,
)

streamLoader_config = dict(
    t0 = 500,
    t1 = 509, # endpoint=false, t1 is not used in training
    pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/openPMD/simData_%T.bp", # files on frontier
    pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/radiationOpenPMD/e_radAmplitudes%T.bp", # files on frontier
    # pathpattern1 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/openPMD/simData_%T.bp5", # files on hemera
    # pathpattern2 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files on hemera
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    normalization = normalization_values,
    number_particles_per_gpu = 1000
)

particleDataTransformationPolicy = BoxesAttributesParticles() #returns particle data of shape (local ranks, number_of_particles, ps_dims)
#particleDataTransformationPolicy = ParticlesAttributes() #returns particle data of shape (number_of_particles, ps_dims)

# radiationDataTransformationPolicy = PerpendicularAbsoluteAndPhase() #returns radiation data of shape (local ranks, frequencies)
radiationDataTransformationPolicy = AbsoluteSquare() #returns radiation data of shape (local ranks, frequencies)
#radiationDataTransformationPolicy = AbsoluteSquareSumRanks() # returns radiation data of shape (frequencies)

timeBatchLoader = StreamLoader(openPMDBuffer, streamLoader_config, particleDataTransformationPolicy, radiationDataTransformationPolicy) ## Streaming ready


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
filepath = '/autofs/nccs-svm1_home1/ksteinig/src/InSituML/main/ModelHelpers/cINN/trained_models/{}/best_model_'

original_state_dict = torch.load(filepath.format(config["load_model"]))
updated_state_dict = {key.replace('VAE.', 'base_network.'): value for key, value in original_state_dict.items()}
model.load_state_dict(updated_state_dict)
print('Loaded pre-trained model successfully')

        
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

#wandb_logger = WandbLogger(project="khi_public",args=config, entity='jeyhun')

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

