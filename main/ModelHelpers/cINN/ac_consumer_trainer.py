"""


"""
import time
from threading import Thread
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
                 multigpu_run=True,
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

        if multigpu_run is not None:

            self.multigpu_run = multigpu_run

            dist.init_process_group("nccl")

            self.gpu_id = dist.get_rank()

            torch.cuda.set_device(self.gpu_id)
            torch.cuda.empty_cache()

            self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            self.gpu_id = device

    def run(self):

        rest_training_left_counter=0

        while True:
            
            phase_space_radiation = self.training_buffer.get_batch()

            
            #this is now only indicating that there 
            #is not enough data in the now buffer 
            #for training to begin
            if phase_space_radiation is None:
                print(f"Trainer will wait for {self.sleep_before_retry} seconds, for data to be "
                        f"streamed before reattempting batch extraction." )
                time.sleep(self.sleep_before_retry)
                continue   
            
            phase_space, radiation = phase_space_radiation
            
            phase_space = phase_space.to(self.gpu_id)
            radiation = radiation.to(self.gpu_id)

            # only logging from of the master process
            if self.gpu_id == 0 and self.batch_passes != 0:
                
                # print loss terms
                loss_message_parts = [f'batch_index: {self.batch_passes-1} | loss_avg: {loss_avg.item():.4f}']
                for loss_name, loss_value in losses.items():
                    loss_message_parts.append(f'{loss_name}: {loss_value.item():.4f}')

                loss_message = ' | '.join(loss_message_parts)
                print(loss_message)
                                
                if self.logger is not None:
                    
                    self.logger.log_scalar(scalar=loss_avg.item(), name="loss_avg", epoch=self.batch_passes-1)
                    
                    for loss_name, loss_value in losses.items():
                        self.logger.log_scalar(scalar=loss_value.item(), name=loss_name, epoch=self.batch_passes-1)
                    
            self.batch_passes += 1
            
            self.optimizer.zero_grad()
            
            losses = self.model(phase_space, radiation)
            loss = losses['total_loss']
            
            #reconstruction if needed
            # y, lat_z_pred = self.model.reconstruct(phase_space,radiation)
            
            self.cumulative_loss += loss
            loss_avg = self.cumulative_loss / self.batch_passes
            loss.backward()
            
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-5.00, 5.00)

            self.optimizer.step()
            self.scheduler.step()
            
            print("Trained on a batch succesfully")
            
            if self.training_buffer.openpmdProduction == False:
                print(f"Note: The streaming has stopped, the trainer will run for "
                        f"{self.ts_after_stopped_production} training steps (batch passes) "
                        f"before stopping.\n"
                        f"Training step:{rest_training_left_counter} after the streaming has stopped.")
                rest_training_left_counter+=1
                if rest_training_left_counter>self.ts_after_stopped_production:
                    if self.multigpu_run is not None: dist.destroy_process_group()
                    break
