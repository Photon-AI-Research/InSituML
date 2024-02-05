"""

"""
import os
import numpy as np
import time

import torch
from torch import optim

#from threading import Thread
from queue import Queue

from ks_helperfuncs import *
from ks_consumer_MAF_khi_radiation import *

hyperparameter_defaults = dict(
t0 = 1990,
t1 = 2001,
dim_input = 90000,
timebatchsize = 4,
particlebatchsize = 32,
dim_condition = 2048,
num_coupling_layers = 3,
hidden_size = 64,
lr = 0.001,
num_epochs = 10,
num_blocks_mat = 2,
activation = 'gelu',
pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy"
)

enable_wandb = False
start_epoch = 0
min_valid_loss = np.inf

assert (hyperparameter_defaults["t1"] - hyperparameter_defaults["t0"])%hyperparameter_defaults["timebatchsize"] == 0, "t1-t0 must be devisible by timebatchsize"
totalTimebatchNumber = int((hyperparameter_defaults["t1"] - hyperparameter_defaults["t0"])/hyperparameter_defaults["timebatchsize"])



wandb_run = None

if enable_wandb:
    print('New session...')
    # Pass your defaults to wandb.init
    wandb_run = wandb.init(entity="jeyhun", config=hyperparameter_defaults, project="khi_public")

    # Access all hyperparameter values through wandb.config
    config = wandb_run.config


""" Replace by an instance of the Producer
l = Loader(pathpattern1 = hyperparameter_defaults["pathpattern1"],
           pathpattern2 = hyperparameter_defaults["pathpattern2"],
           t0 = hyperparameter_defaults["t0"],
           t1 = hyperparameter_defaults["t1"],
           timebatchsize = hyperparameter_defaults["timebatchsize"],
           particlebatchsize = hyperparameter_defaults["particlebatchsize"])
"""

model = (PC_MAF(dim_condition = hyperparameter_defaults["dim_condition"],
                           dim_input = hyperparameter_defaults["dim_input"],
                           num_coupling_layers = hyperparameter_defaults["num_coupling_layers"],
                           hidden_size = hyperparameter_defaults["hidden_size"],
                           device = 'cuda',
                           num_blocks_mat = hyperparameter_defaults["num_blocks_mat"],
                           activation = hyperparameter_defaults["activation"]
                         ))

# dataTransformationPolicy = BoxesParticlesAttributes()
dataTransformationPolicy = None

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=hyperparameter_defaults["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

if enable_wandb:
    directory ='/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/'+str(wandb_run.id)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

start_time = time.time()

# create the shared buffer
batchDataBuffer = Queue(maxsize=int(1.33*totalTimebatchNumber))

# start the consumer
modelTrainer = MafModelTrainer(batchDataBuffer, totalTimebatchNumber, model, optimizer, scheduler, enable_wandb, wandb_run)
modelTrainer.start()

# start the producer
timeBatchLoader = Loader(batchDataBuffer, hyperparameter_defaults, dataTransformationPolicy)
timeBatchLoader.start()

modelTrainer.join()
timeBatchLoader.join()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total elapsed time: {elapsed_time:.6f} seconds")
 
