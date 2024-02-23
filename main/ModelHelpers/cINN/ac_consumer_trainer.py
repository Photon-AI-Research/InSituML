"""
Consumer of PIConGPU particle and radiation data to train an ML model.
The data to train on is supposed to be provided in a buffer which allows accessing it by a `get()` method.
The particle and radiation data itself is expected to be provided as 'timebatches'.
A timebatch is a list holding particle and radiation data from a number of timesteps.
It is supposed to behave like a list. That is, it can be accessed and iterated over with the `[]` operator and it has a `length`.
"""
import time
from threading import Thread

class ModelTrainer(Thread):
    """
    This class implements a trainer based on extracting random batches from the trainer buffer and training the model.
    
    Args:
    
    training_buffer (ac_train_batch_buffer.TrainBatchBuffer object): A TrainBatchBuffer class.

    model_requirements: Model model_requirements to be trained. encoder, decoder and inner_model.

    optimizer: Optimizer object used for model training.

    scheduler: Scheduler object used for optimizer

    sleep_before_retry(int): As the train buffer is being filled. The trainer waits till it has number of items which are equal to training batch size.
    

    ts_after_stopped_production (int): After the simulation has stopped, openpmdProducer stops producing for train buffer. The trainer will continue training for this many training steps extracting random batches from last state of the training buffer.

    enable_wandb: Whether to log training params to wandb.
    
    wandbRunObject: wandb run object.
    
    """


    def __init__(self,
                 training_buffer,
                 model_requirements, loss_function, optimizer, scheduler,
                 sleep_before_retry=10,
                 ts_after_stopped_production=10,
                 enable_wandb=None, wandbRunObject=None):
        
        Thread.__init__(self)

        # training buffer object.
        self.training_buffer = training_buffer

        self.encoder, self.decoder, self.inner_model = model_requirements
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function_AE = loss_function_AE
        self.loss_function_IM = loss_function_IM
        self.losses = []
        self.sleep_before_retry = sleep_before_retry
        self.enable_wandb = enable_wandb
        self.wandb_run = wandbRunObject
        
        self.batch_passes = 0
        self.ts_after_stopped_production = ts_after_stopped_production
        
    def run(self):

        rest_training_left_counter=0

        while True:
            
            phase_space_radiation = self.training_buffer.get_batch()
            
            #this is now only indicating that there 
            #is not enough data in the now buffer 
            #for training to begin
            if phase_space_radiation is None:
                time.sleep(self.sleep_before_retry)
                continue

            self.batch_passes += 1
            
            phase_space, radiation = phase_space_radiation
            
            self.optimizer.zero_grad()
            # loss = - self.model.model.log_prob(inputs=phase_space.to(self.model.device),
            #                                 context=radiation.to(self.model.device))
            
            encoded = self.encoder(phase_space.to(self.encoder.device))
            decoded = self.decoded(encoded)
            
            loss_AE = self.loss_function_AE(decoded, phase_space) 
            loss_IM = self.loss_function_IM(self.inner_model(decoded),
                                            self.radiation.to(self.inner_model.device))
            
            loss = loss_AE + loss_IM

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


            if self.training_buffer.openpmdProduction == False:
                rest_training_left_counter+=1
                if rest_training_left_counter>self.ts_after_stopped_production:
                    break

