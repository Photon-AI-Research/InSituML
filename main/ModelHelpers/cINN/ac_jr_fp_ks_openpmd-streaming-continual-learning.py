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

from model import model_MAF as model_MAF

from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
from train_khi_AE_refactored.networks import VAE, ConvAutoencoder

print("Done importing modules.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

openPMDBuffer = Queue() ## Buffer shared between openPMD data loader and model


#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 100

streamLoader_config = dict(
    t0 = 500,
    t1 = 509, # endpoint=false, t1 is not used in training
    pathpattern1 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/openPMD/simData_%T.bp5", # files on hemera
    pathpattern2 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files on hemera
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu = 10000
)

# particleDataTransformationPolicy = BoxesParticlesAttributes()
particleDataTransformationPolicy = None

# radiationDataTransformationPolicy = PerpendicularAbsoluteAndPhase()
radiationDataTransformationPolicy = AbsoluteSquare()

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
 lr = 0.00001
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
        
        loss_AE, _, encoded = self.base_network(x)
        loss_IM = self.inner_model(inputs=encoded, context=y)*self.weight_IM
        
        return loss_AE*self.weight_AE + loss_IM



VAE_encoder_kwargs = {"ae_config":"non_deterministic",
                   "z_dim":latent_space_dims,
                   "input_dim":ps_dims,
                   "conv_layer_config":[16, 32, 64, 128, 256, 512],
                   "conv_add_bn": False, 
                   "fc_layer_config":[256]}
 
VAE_decoder_kwargs = {"z_dim":latent_space_dims,
                   "input_dim":ps_dims,
                   "initial_conv3d_size":[16, 8, 4, 4],
                   "add_batch_normalisation":False}
                          
VAE = VAE(encoder = Encoder, 
           encoder_kwargs = VAE_encoder_kwargs, 
           decoder = Conv3DDecoder, 
           z_dim=latent_space_dims,
           decoder_kwargs = VAE_decoder_kwargs,
           loss_function = EarthMoversLoss(),
           property_="momentum_force",
           particles_to_sample = number_of_particles,
           ae_config="non_deterministic")


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

inner_model = (model_MAF.PC_MAF(dim_condition=config["dim_condition"],
                           dim_input=config["dim_input"],
                           num_coupling_layers=config["num_coupling_layers"],
                           hidden_size=config["hidden_size"],
                           device=device,
                           num_blocks_mat = config["num_blocks_mat"],
                           activation = config["activation"]
                         ))

#model = ModelFinal(VAE, inner_model, EarthMoversLoss())
model = ModelFinal(conv_AE, inner_model, EarthMoversLoss())

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

trainBF = TrainBatchBuffer(openPMDBuffer)
modelTrainer = ModelTrainer(trainBF, model, optimizer, scheduler)

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

