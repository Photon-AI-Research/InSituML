"""
Consumer of PIConGPU particle and radiation data to create a train dataset buffer. The train dataset buffer is implemented to have some data always there for training of ML Model, while the PIConGPU simulate more timesteps.
"""

from threading import Thread
import torch
from cl_memory import memory
from random import sample

class TrainBuffer(Thread):

    def __init__(self,
                 batchDataBuffer,
                 buffersize = 10,
                 use_continual_learning=True,
                 cl_mem_size=2048):

        Thread.__init__(self)

        self.data = batchDataBuffer

        self.use_continual_learning = use_continual_learning
        ## continual learning related required variables
        if self.use_continual_learning:
            self.er_mem = memory.ExperienceReplay(mem_size=cl_mem_size)
            self.n_obs = 0

        self.memory_update = []
        self.buffer_ = []
        self.buffersize = buffersize


    def run(self):

        while True:
            # get a timebatch from the queue
            particles_radition = self.data.get()

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

class TrainingBatchLoader:

    def __init__(self,
                 trainbuffer,
                 batchsize=4):

        self.batchsize = batchsize
        self.trainbuffer = trainbuffer

        def __len__(self):

            exp_len = len(self.trainbuffer.buffer_)//self.batchsize

            return 2*exp_len if self.use_continual_learning else exp_len

        def __getitem__(self, idx):

            random_sample = sample(self.trainbuffer.buffer_, self.batchsize)

            particles_batch = torch.cat(map(lambda x:x[0], random_sample))
            radiation_batch = torch.cat(map(lambda x:x[1], random_sample))

            if self.trainbuffer.use_continual_learning:

                mem_part_batch, mem_rad_batch = trainbuffer.er_mem.sample(self.batchsize)

                particle_batch = torch.cat([particle_batch, mem_part_batch])
                radiation_batch = torch.cat([radiation_batch, mem_rad_batch])

           return particle_batch, radiation_batch
