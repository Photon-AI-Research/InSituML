"""


"""
import time
from threading import Thread
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelTrainer(Thread):
    """
    This class implements a trainer based on extracting random batches from the trainer buffer and training the model.
    
    Args:
    
    training_buffer (ac_train_batch_buffer.TrainBatchBuffer object): A TrainBatchBuffer class.

    model: Model to be trained.

    optimizer: Optimizer object used for model training.

    scheduler: Scheduler object used for optimizer

    sleep_before_retry(int): As the train buffer is being filled. The trainer waits till it has number of items which are equal to training batch size.
    

    ts_after_stopped_production (int): After the simulation has stopped, openpmdProducer stops producing for train buffer. The trainer will continue training for this many training steps extracting random batches from last state of the training buffer.

    enable_wandb: Whether to log training params to wandb.
    
    wandbRunObject: wandb run object.
    
    """


    def __init__(self,
                 training_buffer,
                 model, 
                 optimizer,
                 scheduler,
                 sleep_before_retry=10,
                 ts_after_stopped_production=10,
                 enable_wandb=None, wandbRunObject=None):
        
        Thread.__init__(self)

        # training buffer object.
        self.training_buffer = training_buffer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.losses = []
        self.sleep_before_retry = sleep_before_retry
        self.enable_wandb = enable_wandb
        self.wandb_run = wandbRunObject
        
        self.batch_passes = 0
        self.ts_after_stopped_production = ts_after_stopped_production
        
    def run(self):

        rest_training_left_counter=0

        while True:
            
            if self.enable_wandb is not None:
                self.wandb_run.log({
                        "batch_passes": self.batch_passes,
                        "loss_avg": sum(losses)/len(losses),
                        "loss": loss,
                    })
            
            phase_space_radiation = self.training_buffer.get_batch()
            
            #this is now only indicating that there 
            #is not enough data in the now buffer 
            #for training to begin
            if phase_space_radiation is None:
                print(f"Trainer will wait for {self.sleep_before_retry} seconds, for data to be "
                        "streamed before reattempting batch extraction." )
                time.sleep(self.sleep_before_retry)
                continue

            self.batch_passes += 1
            
            phase_space, radiation = phase_space_radiation
            
            self.optimizer.zero_grad()
            # loss = - self.model.model.log_prob(inputs=phase_space.to(self.model.device),
            #                                 context=radiation.to(self.model.device))
            
            
            loss = self.model(phase_space.to(device),
                              radiation.to(device))

            loss = loss.mean()
            self.losses.append(loss.item())
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            
            print("Trained on a batch succesfully")
            
            if self.training_buffer.openpmdProduction == False:
                print(f"Note: The streaming has stopped, the trainer will run for "
                        f"{self.ts_after_stopped_production} training steps (batch passes) "
                        "before stopping.\n" 
                         f"Training step:{rest_training_left_counter} after the streaming has stopped.")
                rest_training_left_counter+=1
                if rest_training_left_counter>self.ts_after_stopped_production:
                    break
