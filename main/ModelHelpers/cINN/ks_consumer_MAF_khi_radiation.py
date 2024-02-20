"""
Consumer of PIConGPU particle and radiation data to train an ML model.
The data to train on is supposed to be provided in a buffer which allows accessing it by a `get()` method.
The particle and radiation data itself is expected to be provided as 'timebatches'.
A timebatch is a list holding particle and radiation data from a number of timesteps.
It is supposed to behave like a list. That is, it can be accessed and iterated over with the `[]` operator and it has a `length`.
"""
import time

class MafModelTrainer():

    def __init__(self,
                 train_data_loader,
                 model, optimizer, scheduler, enable_wandb, wandbRunObject=None):

        # instantiate all required parameters
        self.train_data_loader = train_data_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):

        for phase_space, radiation in self.train_data_loader:

            self.optimizer.zero_grad()
            loss = - self.model.model.log_prob(inputs=phase_space.to(self.model.device),
                                            context=radiation.to(self.model.device))

            loss = loss.mean()
            loss_avg.append(loss.item())
            loss.backward()
            self.optimizer.step()

            loss_timebatch_avg = sum(loss_avg)/len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            self.scheduler.step()

