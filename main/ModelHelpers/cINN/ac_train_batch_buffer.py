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
                 cl_mem_size=2048,
                 model_MAF=False):

        Thread.__init__(self)

        self.openPMDbuffer = openPMDBuffer
        #reshaping is assumed to be different in the case of MAF model.
        self.reshape = self.reshape_MAF if model_MAF else self.reshape_norm

        self.training_bs = training_bs
        self.continual_bs = continual_bs

        self.use_continual_learning = use_continual_learning
        ## continual learning related required variables
        if self.use_continual_learning:
            self.er_mem = ExperienceReplay(mem_size=cl_mem_size)
            self.n_obs = 0
        
        self.i_step = 0
        
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
        updating = False
        if openPMDBufferSize:
            updating = True
            print("Updating the train buffer")
            
        while openPMDBufferReadCount < min(self.training_bs, openPMDBufferSize):
            # get a particles, radiation from the queue
            particles_radiation = self.openPMDbuffer.get()

            if particles_radiation is None:
                self.openpmdProduction = False
                break

            particles_radiation = self.reshape(particles_radiation)

            if len(self.buffer_) < self.buffersize:
                self.buffer_ += particles_radiation
            else:
                #extracts the first elements.
                last_elements = self.buffer_[:len(particles_radiation)]
                self.buffer_ = self.buffer_[len(particles_radiation):]
                
                self.buffer_ += particles_radiation

                if self.use_continual_learning:
                    #add the last element to memory, if continual learning is
                    #required.
                    X = torch.cat([ele[0] for ele in last_elements])
                    Y = torch.cat([ele[1] for ele in last_elements])

                    self.er_mem.update_memory(X,
                                              Y,
                                              n_obs = self.n_obs,
                                              i_step = self.i_step) 
                    
                    self.n_obs += len(last_elements)
                    self.i_step += 1

            openPMDBufferReadCount += 1
            self.noReadCount = 0
        
        else:
            self.noReadCount += 1
        if updating: print("Train Buffer Updated")

    def reshape_norm(self, particles_radiation):
        # adds a batch dims assuming the data is coming as
        # (number_of_particles, dims) -> (1, number_of_particles, dims)
        particles, radiation = particles_radiation
        particles_radiation = [[particles[idx:idx+1].permute(0,2,1), radiation[idx:idx+1]] for idx in range(len(particles))]
        return particles_radiation

    def reshape_MAF(self, particles_radiation):
        # adds a batch dims assuming the data is coming as
        # (number_of_particles, dims) -> (1, number_of_particles*dims)
        particles, radiation = particles_radiation
        return [particles.reshape(1, -1), radiation.reshape(1, -1)]

    def get_batch(self):
        print("Attempting a batch extraction from train buffer")
        self.run()
        # No training until there batch size element in the buffer.
        if len(self.buffer_)<self.training_bs or (self.noReadCount>self.max_tb_from_unchanged_now_bf and
                                                  self.openpmdProduction):
            print(f"Batch extraction failed.. \n"
                   "Either train buffer has less element than training size \n"
                   f"Train Buffer Size: {len(self.buffer_)}, training batch size: {self.training_bs} \n"
                   "Or maximum number batches have extracted from unmodified train buffer state. Maximum train batches "
                   f"allowed from unchanged trainbuffer state: {self.max_tb_from_unchanged_now_bf}\n")
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
