from threading import Thread
import torch
from cl_memory import ExperienceReplay
from random import sample

class TrainBatchBuffer(Thread):
    """
    This class creates a ring buffer where oldest enteries produced openPMDProducer are either discarded or sent to Continual Learning based ExperienceReplay memory buffer.
    
    Args:
    
    openPMDBuffer (Queue): Queue shared between openPMD producer and train buffer.
         
    training_bs (int): training batch size to send to the model to train on.

    buffersize (int): Size of train buffer.

    max_tb_from_unchanged_now_bf (int): Maximum number of training batches that can be extracted from the unchanged
    state of train(or now) buffer. After extracting these many batches trainer would wait for more data to read from
    openPMDBuffer. State of train/now buffer only changes once there is more data to be read from the openPMDBuffer.

    use_continual_learning (Bool): Whether to use use continual learning or not. If yes, will create memory buffer for the continual learning.

    cl_mem_size (int): Continual learning memory buffer size.
    
    """

    def __init__(self,
                 openPMDBuffer,
                 training_bs = 4,
                 buffersize = 5,
                 max_tb_from_unchanged_now_bf = 3,
                 use_continual_learning=True,
                 continual_bs = 4,
                 cl_mem_size=2048):

        Thread.__init__(self)

        self.openPMDbuffer = openPMDBuffer

        self.training_bs = training_bs
        self.continual_bs = continual_bs

        self.use_continual_learning = use_continual_learning
        ## continual learning related required variables
        if self.use_continual_learning:
            self.er_mem = ExperienceReplay(mem_size=cl_mem_size)
            self.n_obs = 0

        self.buffer_ = []
        self.buffersize = buffersize
        
        # to indicate whether there are
        # still production from openPMD production.
        self.openpmdProduction = True
        self.noReadCount = 0
        self.max_tb_from_unchanged_now_bf = max_tb_from_unchanged_now_bf

    def run(self):

        openPMDBufferReadCount = 0
        openPMDBufferSize = self.openPMDbuffer.qsize()

        while openPMDBufferReadCount < min(self.training_bs, openPMDBufferSize):
            # get a particles, radiation from the queue
            particles_radiation = self.openPMDbuffer.get()

            if particles_radiation is None:
                self.openpmdProduction = False
                break

            particles_radiation = self.reshape(particles_radiation)

            if len(self.buffer_) < self.buffersize:
                self.buffer_.append(particles_radiation)
            else:
                #extracts the first element.
                last_element = self.buffer_.pop(0)
                self.buffer_.append(particles_radiation)

                if self.use_continual_learning:
                    #add the last element to memory, if continual learning is
                    #required.
                    self.er_mem.update_memory(*last_element,
                                                n_obs = self.n_obs,
                                                i_step = self.n_obs) #i_step = n_obs in this case
                    self.n_obs += 1

            openPMDBufferReadCount += 1
            self.noReadCount = 0
        
        else:
            self.noReadCount += 1

    def reshape(self, particles_radiation):
        # adds a batch dims assuming the data is coming as
        # (number_of_particles, dims) -> (1, number_of_particles, dims)
        particles, radiation = particles_radiation
        return [torch.unsqueeze(particles, 0), torch.unsqueeze(radiation,0)]
    
    def get_batch(self):

        self.run()
        # No training until there batch size element in the buffer.
        if len(self.buffer_)<self.training_bs or (self.noReadCount>self.max_tb_from_unchanged_now_bf and
                                                  self.openpmdProduction):
            return None
        
        #random sampling
        random_sample = sample(self.buffer_, self.training_bs)

        particles_batch = torch.cat([x[0] for x in random_sample])
        radiation_batch = torch.cat([x[1] for x in random_sample])

        if self.use_continual_learning and self.n_obs>=self.continual_bs:
            #sample from memory
            mem_part_batch, mem_rad_batch = self.er_mem.sample(self.continual_bs)

            particles_batch = torch.cat([particles_batch, mem_part_batch])
            radiation_batch = torch.cat([radiation_batch, mem_rad_batch])

        return particles_batch, radiation_batch
