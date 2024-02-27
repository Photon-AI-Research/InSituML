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
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchsize = 4
number_of_particles = 100
ps_dims = 9

rad = 100
rad_dims = 9

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
            radiation = torch.rand(rad, rad_dims)
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
dim_condition = rad*rad_dims,
num_coupling_layers = 6,
hidden_size = 256,
num_blocks_mat = 6,
activation = 'relu',
 lr = 0.00001
)

# model = (model_MAF.PC_MAF(dim_condition=config["dim_condition"],
#                             dim_input=number_of_particles*ps_dims,
#                             num_coupling_layers=config["num_coupling_layers"],
#                             hidden_size=config["hidden_size"],
#                             device='cuda',
#                             weight_particles=False,
#                             num_blocks_mat = config["num_blocks_mat"],
#                             activation = config["activation"]
#                             ))
class ModelFinal(nn.Module):
    def __init__(self, encoder, decoder, inner_model,
                 loss_function_AE, loss_function_IM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inner_model = inner_model
        self.loss_function_AE = loss_function_AE
        self.loss_function_IM = loss_function_IM
    
    def forward(self, x, y):
        
        encoded = self.encoder(x)
        decoded  = self.decoder(encoded)
        
        loss_AE = self.loss_function_AE(decoded, x) 

        inner_model_out = self.inner_model(encoded)
        loss_IM = self.loss_function_IM(inner_model_out,
                                        y)
        return loss_AE + loss_IM

model = ModelFinal(Encoder(ae_config="simple", z_dim=latent_space_dims,input_dim = ps_dims),
                   MLPDecoder(z_dim=latent_space_dims, input_dim = ps_dims, particles_to_sample=number_of_particles),
                    MLPDecoder(z_dim=latent_space_dims, input_dim = rad_dims, particles_to_sample=number_of_particles),
                    EarthMoversLoss(),
                    nn.MSELoss())
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
