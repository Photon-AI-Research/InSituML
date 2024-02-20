"""
Consumer of PIConGPU particle and radiation data to create a train dataset buffer. The train dataset buffer is implemented to have some data always there for training of ML Model, while the PIConGPU simulate more timesteps.
"""

from threading import Thread
import torch
from cl_memory import ExperienceReplay
from random import sample

class TrainBatchBuffer(Thread):

    def __init__(self,
                 openPMDBuffer,
                 training_batch,
                 training_bs = 4,
                 buffersize = 8,
                 use_continual_learning=False,
                 continual_bs = 4,
                 cl_mem_size=2048):

        Thread.__init__(self)

        self.openPMDbuffer = openPMDBuffer
        self.training_batch = training_batch

        self.training_bs = training_bs
        self.continual_bs = continual_bs

        self.use_continual_learning = use_continual_learning
        ## continual learning related required variables
        if self.use_continual_learning:
            self.er_mem = ExperienceReplay(mem_size=cl_mem_size)
            self.n_obs = 0

        self.buffer_ = []
        self.buffersize = buffersize


    def run(self):

        while True:
            print("running inside tb, flush=True")
            # get a particles, radiation from the queue
            particles_radiation = self.openPMDbuffer.get()

            if particles_radiation is None:
                break

            if len(self.buffer_) < self.buffersize:
                self.buffer_.append(particles_radiation)
            else:
                #extracts the first element.
                last_element = self.buffer_.pop(0)
                self.buffer_.append(particles_radiation)

                if self.use_continual_learning:
                    self.er_mem.update_memory(*last_element,
                                              n_obs = self.n_obs,
                                              i_step = self.n_obs) #i_step = n_obs in this case
                    self.n_obs += 1
            if len(self.buffer_)>=self.training_bs:
               print("before batch put")  
               self.training_batch.put(self.get_batch())
        
        self.training_batch.put(None)

    def get_batch(self):
        
        print("in get batch")
        random_sample = sample(self.buffer_, self.training_bs)

        particles_batch = torch.cat([x[0] for x in random_sample])
        radiation_batch = torch.cat([x[1] for x in random_sample])

        if self.use_continual_learning:

            mem_part_batch, mem_rad_batch = self.er_mem.sample(self.continual_bs)

            particles_batch = torch.cat([particles_batch, mem_part_batch])
            radiation_batch = torch.cat([radiation_batch, mem_rad_batch])

        return particles_batch, radiation_batch
