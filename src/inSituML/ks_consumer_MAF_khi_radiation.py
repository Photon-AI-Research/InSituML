"""
Consumer of PIConGPU particle and radiation data to train an ML model.
The data to train on is supposed to be provided in a buffer which allows accessing it by a `get()` method.
The particle and radiation data itself is expected to be provided as 'timebatches'.
A timebatch is a list holding particle and radiation data from a number of timesteps.
It is supposed to behave like a list. That is, it can be accessed and iterated over with the `[]` operator and it has a `length`.
"""
import time

from threading import Thread

from ks_helperfuncs import *

class MafModelTrainer(Thread):

    def __init__(self, batchDataBuffer, totalTimebatchNumber, model, optimizer, scheduler, enable_wandb, wandbRunObject=None):
        Thread.__init__(self)
        # instantiate all required parameters
        self.data = batchDataBuffer
        self.numTbs = totalTimebatchNumber # number of timebatches the data has been divided in
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.enable_wandb = enable_wandb
        self.wandb_run = wandbRunObject

    def run(self):
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

                loss = - self.model.model.log_prob(inputs=phase_space.to(self.model.device),context=radiation.to(self.model.device))

                loss = loss.mean()
                loss_avg.append(loss.item())
                loss.backward()
                self.optimizer.step()

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

