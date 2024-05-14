"""
Main file/module to train ML model from PIConGPU openPMD data using threads.

Start training by executing `python ks_main.py`.
"""
import os
import numpy as np
import time

from sys import stdout

import torch
from torch import optim

from queue import Queue

from ks_helperfuncs import *
from ks_consumer_MAF_khi_radiation import *
from ks_transform_policies import *
from ks_producer_openPMD import *
from ks_producer_openPMD_streaming import *

print("Done importing modules.")

normalization_values = dict(
    momentum_mean = 0.,
    momentum_std = 1.,
    force_mean = 0.,
    force_std = 1.,
)

hyperparameter_defaults = dict(
t0 = 500,
t1 = 509, # endpoint=false, t1 is not used in training
dim_input = 90000,
dim_condition = 2048,
num_coupling_layers = 3,
hidden_size = 64,
lr = 0.001,
num_epochs = 8,
num_blocks_mat = 2,
activation = 'gelu',
pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/openPMD/simData_%T.bp", # files on frontier
pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/radiationOpenPMD/e_radAmplitudes%T.bp", # files on frontier
#pathpattern1 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/openPMD/simData_%T.bp5", # files on hemera
#pathpattern2 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files on hemera
amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
phase_space_variables = ["position", "momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
normalization = normalization_values,
number_particles_per_gpu = 1000,
)

enable_wandb = False
start_epoch = 0
min_valid_loss = np.inf

wandb_run = None

if enable_wandb:
    print('New session...')
    # Pass your defaults to wandb.init
    wandb_run = wandb.init(entity="jeyhun", config=hyperparameter_defaults, project="khi_public")

    # Access all hyperparameter values through wandb.config
    config = wandb_run.config


"""
model = (PC_MAF(dim_condition = hyperparameter_defaults["dim_condition"],
                           dim_input = hyperparameter_defaults["dim_input"],
                           num_coupling_layers = hyperparameter_defaults["num_coupling_layers"],
                           hidden_size = hyperparameter_defaults["hidden_size"],
                           device = 'cuda',
                           num_blocks_mat = hyperparameter_defaults["num_blocks_mat"],
                           activation = hyperparameter_defaults["activation"]
                         ))
"""
particleDataTransformationPolicy = BoxesAttributesParticles()

# radiationDataTransformationPolicy = PerpendicularAbsoluteAndPhase()
radiationDataTransformationPolicy = AbsoluteSquare()
"""
# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=hyperparameter_defaults["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
"""
if enable_wandb:
    directory ='/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/'+str(wandb_run.id)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

start_time = time.time()

# create the shared buffer
batchDataBuffer = Queue(maxsize=int(1.33*(hyperparameter_defaults['t1']-hyperparameter_defaults['t0'])))

class DummyTimebatchConsumer(Thread):
    """Dummy consumer task"""
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        from time import sleep
        print('Consumer: Running')
        # consume items
        while True:
            # get a unit of work
            print("Get next queue item.")
            item = self.queue.get()
            print("Got queue item")
            stdout.flush()
            # check for stop
            if item is None:
                print("Reached end of queue")
                stdout.flush()
                break
            # block, to simulate effort
            sleep(3)
            print(f'>consumer got item')
            print(f'Number of boxes times number of timesteps {len(item)}')
            stdout.flush()
            # report
        # all done
        print('Consumer: Done')
        stdout.flush()


dummyConsumer = DummyTimebatchConsumer(batchDataBuffer)
dummyConsumer.start()


# start the producer
timeBatchLoader = RandomLoader(batchDataBuffer, hyperparameter_defaults, particleDataTransformationPolicy, radiationDataTransformationPolicy) ## Normal load offline data with random order of iterations
#timeBatchLoader = StreamLoader(batchDataBuffer, hyperparameter_defaults, particleDataTransformationPolicy, radiationDataTransformationPolicy) ## Streaming ready
timeBatchLoader.start()

#modelTrainer.join()
dummyConsumer.join()
print("Join consumer")
stdout.flush()
timeBatchLoader.join()
print("Join producer")
stdout.flush()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total elapsed time: {elapsed_time:.6f} seconds")

