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
                 logger=None):
        
        Thread.__init__(self)

        # training buffer object.
        self.training_buffer = training_buffer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.cumulative_loss = 0.0
        self.sleep_before_retry = sleep_before_retry
        self.logger = logger
        
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
                print(f"Trainer will wait for {self.sleep_before_retry} seconds, for data to be "
                        "streamed before reattempting batch extraction." )
                time.sleep(self.sleep_before_retry)
                continue   
            
            phase_space, radiation = phase_space_radiation
            
            phase_space = phase_space.to(device)
            radiation = radiation.to(device)
            
            if self.batch_passes !=0:
                loss = loss.item()
                loss_avg = loss_avg.item()
                loss_AE = loss_AE.item()
                loss_IM = loss_IM.item()
                loss_ae_reconst = loss_ae_reconst.item()
                kl_loss = kl_loss.item()
                l_fit = l_fit.item()
                l_latent = l_latent.item()
                l_rev = l_rev.item()
                batch_index = self.batch_passes-1
                print('batch_index: {} | loss_avg: {:.4f} | loss: {:.4f} | loss_AE: {:.4f} | loss_IM: {:.4f} | loss_ae_reconst: {:.4f} | kl_loss: {:.4f} | l_fit: {:.4f} | l_latent: {:.4f} | l_rev: {:.4f}'.format(batch_index, loss_avg, loss,loss_AE,loss_IM,loss_ae_reconst,kl_loss,l_fit,l_latent,l_rev))

                if self.logger is not None:
                    self.logger.log_scalar(scalar=loss_avg, name="loss_avg", epoch=batch_index)
                    self.logger.log_scalar(scalar=loss, name="loss", epoch=batch_index)
                    self.logger.log_scalar(scalar=loss_AE, name="loss_AE", epoch=batch_index)
                    self.logger.log_scalar(scalar=loss_IM, name="loss_IM", epoch=batch_index)
                    self.logger.log_scalar(scalar=loss_ae_reconst, name="loss_ae_reconst", epoch=batch_index)
                    self.logger.log_scalar(scalar=kl_loss, name="kl_loss", epoch=batch_index)
                    self.logger.log_scalar(scalar=l_fit, name="l_fit", epoch=batch_index)
                    self.logger.log_scalar(scalar=l_latent, name="l_latent", epoch=batch_index)
                    self.logger.log_scalar(scalar=l_rev, name="l_rev", epoch=batch_index)
                    
            self.batch_passes += 1
            
            self.optimizer.zero_grad()
            
            loss,loss_AE,loss_IM,loss_ae_reconst,kl_loss,l_fit,l_latent,l_rev = self.model(phase_space,
                              radiation)
            
            self.cumulative_loss += loss
            loss_avg = self.cumulative_loss / self.batch_passes
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
