"""
Consumer of PIConGPU particle and radiation data to train an ML model.
The data to train on is supposed to be provided in a buffer which allows accessing it by a `get()` method.
The particle and radiation data itself is expected to be provided as 'timebatches'.
A timebatch is a list holding particle and radiation data from a number of timesteps.
It is supposed to behave like a list. That is, it can be accessed and iterated over with the `[]` operator and it has a `length`.
"""
import time

from threading import Thread
import torch
from ks_helperfuncs import *

try:
    ## requires submoduling from here:
    ## https://github.com/elcorto/nmbx
    from nmbx.convergence import SlopeZero

    conv_control_possible = True
except ImportError:
    conv_control_possible = False
    print("no convergence control possible, we use max_epoch")

import memory

class MafModelTrainer(Thread):

    def __init__(self, batchDataBuffer, 
                 totalTimebatchNumber, model, optimizer, 
                 scheduler, enable_wandb,
                 particle_batchsize,
                 num_epoch = 1,
                 wandbRunObject=None, use_mem=True, mem_size=2048):
        
        Thread.__init__(self)
        # instantiate all required parameters
        self.data = batchDataBuffer
        self.numTbs = totalTimebatchNumber # number of timebatches the data has been divided in
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.enable_wandb = enable_wandb
        self.wandb_run = wandbRunObject
        
        ## continual learning related required variables        
        if use_mem:
            self.er_mem = memory.ExperienceReplay(mem_size=mem_size)
            # epdisodic memory batch
            # here http://arxiv.org/abs/1902.10486 it was kept equal 
            # to regular particle batch size entered.
            self.epdisodic_mem_bs = particle_batchsize
            self.n_obs = 0
            
        if conv_control_possible:
            # taking values as they are from here
            #https://github.com/Photon-AI-Research/InSituML/blob/feature-fix-plotting-toy-cl/examples/streaming_toy_data/toy_cl.py
            self.conv = SlopeZero(wlen=25, tol=1e-2, wait=20, reduction=np.mean)
        
        #counter keeping track of how times training
        #run has been called
        self.training_run = 0

    def run(self):
        
        self.training_run += 1
        
        i_epoch = int(0)
        tb_count = int(0)
        loss_overall = []
        
        while True:
            # get a timebatch from the queue
            timebatch = self.data.get()
            if timebatch is None:
                break
            loss_avg = []

            start_timebatch = time.time()
            for b in range(len(timebatch)):
                
                self.optimizer.zero_grad()
                phase_space, radiation = timebatch[b]
                
                # Don't use memory in first training run since there is nothing to
                # remember.
                if self.use_mem and self.training_run > 0:
                    phase_space_mem, radiation_mem = er_mem.sample(
                                                     self.epdisodic_mem_bs)

                    phase_space_clb = torch.vstack((phase_space, phase_space_mem))
                    radiation_clb = torch.vstack((radiation, radiation_mem))
                else:
                    phase_space_clb, radiation_clb = phase_space, radiation
                
                
                #adding the continual learning batch
                loss = - self.model.model.log_prob(inputs=phase_space_clb.to(self.model.device),
                                                   context=radiation_clb.to(self.model.device))

                loss = loss.mean()
                loss_avg.append(loss.item())
                loss.backward()
                self.optimizer.step()
                
                if self.use_mem:
                    er_mem.update_memory(phase_space, 
                                         timebatch,
                                         self.n_obs,
                                         self.training_run)
                    
                    self.n_obs += self.epdisodic_mem_bs

            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch

            loss_timebatch_avg = sum(loss_avg)/len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print('i_epoch:{}, tb: {}, last timebatch loss: {}, avg_loss: {}, time: {}'.format(i_epoch,tb,loss.item(), loss_timebatch_avg, elapsed_timebatch))
            
            tb_count += 1
            # end processing of a single timebatch

            if tb_count%self.numTbs == 0:
                """All timebatches have been read -> an epoch has passed"""
                i_epoch += 1
                loss_overall_avg = sum(loss_overall)/len(loss_overall)

                if min_valid_loss > loss_overall_avgi:
                    print(f'Training Loss Decreased({min_valid_loss:.6f}--->{loss_overall_avg:.6f}) \t Saving The Model')
                    min_valid_loss = loss_overall_avg
                    # Saving State Dict
                    # torch.save(model.state_dict(), directory + '/best_model_', _use_new_zipfile_serialization=False)

                if (i_epoch) % 10 == 0 and self.enable_wandb and self.wandb_run is not None:
                    save_checkpoint(self.model, optimizer, directory, loss, min_valid_loss, i_epoch, self.wandb_run.id)

                self.scheduler.step()

                if self.enable_wandb and self.wandb_run is not None:
                    # Log the loss and accuracy values at the end of each epoch
                    self.wandb_run.log({
                        "Epoch": i_epoch,
                        "loss_timebatch_avg_loss": loss_timebatch_avg,
                        "loss_overall_avg": loss_overall_avg,
                        "min_valid_loss": min_valid_loss,
                    })
                    
            # termination (reconstruction loss converged, etc)
            if self.conv_control_possible and self.conv.check(loss_overall):
                print(f"converged at {i_epoch=}, last {loss_timebatch_avg=}")
                break
            elif (i_epoch + 1) == num_epoch:
                print(f"hit max_epoch, last {loss_timebatch_avg=}")
                break

