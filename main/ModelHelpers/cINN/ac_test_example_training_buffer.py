from ac_train_batch_buffer import TrainBatchBuffer
from ac_consumer_trainer import ModelTrainer
from threading import Thread
import torch
from time import sleep
from random import random
from threading import Thread
from queue import Queue

from torch import optim
import torch.nn as nn

from model import model_MAF as model_MAF

from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.networks import VAE 
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchsize = 4
number_of_particles = 100
ps_dims = 9

rad = 100
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
            loaded_particles = torch.rand(number_of_particles, ps_dims)
            radiation = torch.rand(rad_dims)
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

class ModelFinal(nn.Module):
    def __init__(self, 
                 VAE,
                 inner_model,
                 loss_function_IM = None,
                 weight_AE=1.0,
                 weight_IM=1.0):
        super().__init__()
        
        self.VAE = VAE
        self.inner_model = inner_model
        self.loss_function_IM = loss_function_IM
        self.weight_AE = weight_AE
        self.weight_IM = weight_IM
    
    def forward(self, x, y):
        
        loss_AE, _, encoded = self.VAE(x)
        loss_IM = self.inner_model(inputs=encoded, context=y)*self.weight_IM
        
        return loss_AE*self.weight_AE + loss_IM

# encoder = Encoder(ae_config="simple",
#                   z_dim=latent_space_dims,
#                   input_dim = ps_dims,
#                   conv_layer_config = [16, 32, 64, 128, 256, 512],
#                   conv_add_bn = False)

encoder_kwargs = {"ae_config":"non_deterministic",
                  "z_dim":latent_space_dims,
                  "input_dim":ps_dims,
                  "conv_layer_config":[16, 32, 64, 128, 256, 512],
                  "conv_add_bn": False, 
                  "fc_layer_config":[]}

decoder_kwargs = {"z_dim":latent_space_dims,
                  "input_dim":ps_dims,
                  "add_batch_normalisation":False}
                         
inner_model = (model_MAF.PC_MAF(dim_condition=config["dim_condition"],
                           dim_input=config["dim_input"],
                           num_coupling_layers=config["num_coupling_layers"],
                           hidden_size=config["hidden_size"],
                           device=device,
                           num_blocks_mat = config["num_blocks_mat"],
                           activation = config["activation"]
                         ))

VAE = VAE(encoder = Encoder, 
          encoder_kwargs = encoder_kwargs, 
          decoder = Conv3DDecoder, 
          decoder_kwargs = decoder_kwargs,
          loss_function = EarthMoversLoss(),
          property_="momentum_force",
          particles_to_sample = number_of_particles,
          ae_config="non_deterministic")

model = ModelFinal(VAE, inner_model, EarthMoversLoss())

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
