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
                 buffersize = 10,
                 use_continual_learning=True,
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
            # get a particles, radiation from the queue
            particles_radition = self.openPMDbuffer.get()

            if particles_radition is None:
                break

            if len(self.buffer_) < self.buffersize:
                self.buffer_.append(particles_radition)
            else:
                #extracts the first element.
                last_element = self.buffer_.pop(0)
                self.buffer_.append(particles_radition)

                if self.use_continual_learning:
                    self.er_mem.update_memory(*last_element,
                                              n_obs = self.n_obs,
                                              i_step = self.n_obs) #i_step = n_obs in this case
                    self.n_obs += 1
                    
            self.training_batch.put(self.get_batch())
        
        self.training_batch.put(None)

    def get_batch(self):
        
        random_sample = sample(self.trainbuffer.buffer_, self.training_bs)

        particles_batch = torch.cat(map(lambda x:x[0], random_sample))
        radiation_batch = torch.cat(map(lambda x:x[1], random_sample))

        if self.trainbuffer.use_continual_learning:

            mem_part_batch, mem_rad_batch = self.er_mem.sample(self.continual_bs)

            particle_batch = torch.cat([particle_batch, mem_part_batch])
            radiation_batch = torch.cat([radiation_batch, mem_rad_batch])

        return particle_batch, radiation_batch
