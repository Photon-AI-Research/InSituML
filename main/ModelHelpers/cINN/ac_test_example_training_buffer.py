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
from train_khi_AE_refactored.networks import VAE, ConvAutoencoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchsize = 4
gpu_boxes = 2
number_of_particles = 100
ps_dims = 6

rad_dims = 512

latent_space_dims = 1024

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



openPMDBuffer = Queue()

openpmdProducer = DummyOpenPMDProducer(openPMDBuffer)

config = dict(
dim_input = 1024,
dim_condition = rad_dims,
num_coupling_layers = 4,
hidden_size = 256,
num_blocks_mat = 6,
activation = 'relu',
 lr = 0.00001
)

config_inn = dict(
y_noise_scale = 1e-1,
zeros_noise_scale = 5e-2,
lambd_predict = 3.,
lambd_latent = 300.,
lambd_rev = 400.,
ndim_tot = 1024,
ndim_x = 1024,
ndim_y = 512,
ndim_z = 512,
num_coupling_layers = 4,
hidden_size = 256,
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
        
        # Check if the inner model is an instance of INNModel
        if isinstance(self.inner_model, INNModel):
            # Use the compute_losses function of INNModel
            loss_IM = self.inner_model.compute_losses(encoded, y) * self.weight_IM
        else:
            # For other types of models, such as MAF
            loss_IM = self.inner_model(inputs=encoded, context=y) * self.weight_IM
        
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

# MAF
# inner_model = PC_MAF(dim_condition=config["dim_condition"],
#                            dim_input=config["dim_input"],
#                            num_coupling_layers=config["num_coupling_layers"],
#                            hidden_size=config["hidden_size"],
#                            device=device,
#                            num_blocks_mat = config["num_blocks_mat"],
#                            activation = config["activation"]
#                          )

# INN
inner_model = INNModel(ndim_tot=config_inn["ndim_tot"],
                 ndim_x=config_inn["ndim_x"],
                 ndim_y=config_inn["ndim_y"],
                 ndim_z=config_inn["ndim_z"],
                 loss_fit=fit,
                 loss_latent=MMD_multiscale,
                 loss_backward=MMD_multiscale,
                 lambd_predict=config_inn["lambd_predict"],
                 lambd_latent=config_inn["lambd_latent"],
                 lambd_rev=config_inn["lambd_rev"],
                 zeros_noise_scale=config_inn["zeros_noise_scale"],
                 y_noise_scale=config_inn["y_noise_scale"],
                 hidden_size=config_inn["hidden_size"],
                 activation=config["activation"],
                 num_coupling_layers=config_inn["num_coupling_layers"],
                 device = device)

#model = ModelFinal(VAE, inner_model, EarthMoversLoss())
model = ModelFinal(conv_AE, inner_model, EarthMoversLoss())

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

trainBF = TrainBatchBuffer(openPMDBuffer)
modelTrainer = ModelTrainer(trainBF, model, optimizer, scheduler)

modelTrainer.start()
trainBF.start()
openpmdProducer.start()

openpmdProducer.join()
trainBF.join()
modelTrainer.join()
