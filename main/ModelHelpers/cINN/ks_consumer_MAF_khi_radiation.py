"""
Consumer of PIConGPU particle and radiation data to train an ML model.
The data to train on is supposed to be provided in a buffer which allows accessing it by a `get()` method.
The particle and radiation data itself is expected to be provided as 'timebatches'.
A timebatch is a list holding particle and radiation data from a number of timesteps.
It is supposed to behave like a list. That is, it can be accessed and iterated over with the `[]` operator and it has a `length`.
"""
import time
from threading import Thread

class MafModelTrainer(Thread):

    def __init__(self,
                 training_batch,
                 model, optimizer, scheduler,
                 enable_wandb=None, wandbRunObject=None):
        
        Thread.__init__(self)

        # instantiate all required parameters
        self.training_batch = training_batch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = []
        self.enable_wandb = enable_wandb
        self.wandb_run = wandbRunObject
        
        self.batch_passes = 0
        
    def run(self):
        
        while True:
            self.batch_passes += 1
            phase_space_radiation = self.training_batch.get()

            if phase_space_radiation is None:
                break
            
            phase_space, radiation = phase_space_radiation
            
            self.optimizer.zero_grad()
            loss = - self.model.model.log_prob(inputs=phase_space.to(self.model.device),
                                            context=radiation.to(self.model.device))

            loss = loss.mean()
            self.losses.append(loss.item())
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            
            if self.enable_wandb is not None:
                self.wandb_run.log({
                        "batch_passes": self.batch_passes,
                        "loss_avg": sum(losses)/len(losses),
                        "loss": loss,
                    })

