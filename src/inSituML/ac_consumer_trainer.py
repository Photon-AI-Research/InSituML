"""
This class prints losses and optionally calls another logger to log losses in another way.
"""
import time
from threading import Thread
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .utilities import save_checkpoint_conditionally
from mpi4py import MPI
import os, time


class LossLogger:
    def __init__(self, otherLogger=None, max_logs=20, out_prefix=""):
        self.printedHeader = False
        self.logger = otherLogger

        comm = MPI.COMM_WORLD
        stride = (comm.size + max_logs) // max_logs - 1
        if stride == 0:
            stride += 1
        self.log = comm.rank % stride == 0

        if self.log:
            self.log_path = f"{out_prefix}loss_{comm.rank}.dat"

            checkpoint_dirname = os.path.dirname(self.log_path)
            if checkpoint_dirname and not os.path.exists(checkpoint_dirname):
                try:
                    os.mkdir(checkpoint_dirname)
                except FileExistsError:
                    pass

            self.out_stream = open(self.log_path, 'w')

            self.last_time_point = int(time.time() * 1000)

    def __del__(self):
        if self.log:
            self.out_stream.close()


    def __call__(self, losses, batch_index):
        if not self.log:
            return
        if not self.printedHeader:
            self.out_stream.write('\t'.join(['# batch_index', "time"] + list(losses.keys())) + '\n')
            self.printedHeader = True

        current = int(time.time() * 1000)
        # print loss terms
        self.out_stream.write('\t'.join([str(batch_index), str(current - self.last_time_point)] + list(str(v.item()) for v in losses.values())) + '\n')
        self.out_stream.flush()
        # loss_message_parts = [f'batch_index: {self.batch_passes-1} ']

        if self.logger is not None:
            
            for loss_name, loss_value in losses.items():
                self.logger.log_scalar(scalar=loss_value.item(), name=loss_name, epoch=self.batch_passes-1)

        self.last_time_point = current

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
                 gpu_id=None,
                 sleep_before_retry=2,
                 ts_after_stopped_production=0,
                 checkpoint_interval = 0,
                 out_prefix = "",
                 checkpoint_final = False,
                 max_logs = 20,
                 logger=None):
        
        Thread.__init__(self)

        # training buffer object.
        self.training_buffer = training_buffer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sleep_before_retry = sleep_before_retry
        
        self.checkpoint_interval = checkpoint_interval
        self.out_prefix = out_prefix
        self.checkpoint_final = checkpoint_final
        self.batch_passes = 0
        self.training_samples = 0
        self.ts_after_stopped_production = ts_after_stopped_production


        if gpu_id is not None:

            self.gpu_id = gpu_id

            self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            self.gpu_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # should we not still push the model there?

        self.logger = LossLogger(logger, max_logs=max_logs, out_prefix=out_prefix)

        # torch.cuda.memory._record_memory_history(max_entries=1000000)

    def run(self):

        rest_training_left_counter=0

        while True:

            phase_space_radiation = self.training_buffer.get_batch()

            #this is now only indicating that there 
            #is not enough data in the now buffer 
            #for training to begin
            if phase_space_radiation is None:
                if not self.training_buffer.openpmdProduction:
                    # something went wrong when reading the first data, abort
                    break
                print(f"Trainer will wait for {self.sleep_before_retry} seconds, for data to be "
                        f"streamed before reattempting batch extraction.", flush=True)
                time.sleep(self.sleep_before_retry)
                continue   
            
            phase_space, radiation = phase_space_radiation

            self.training_samples += phase_space.shape[0]
            
            phase_space = phase_space.to(self.gpu_id)
            radiation = radiation.to(self.gpu_id)
                    
            self.batch_passes += 1

            # torch.cuda.memory._dump_snapshot("profile_{}.pickle".format(self.batch_passes))
            
            self.optimizer.zero_grad()
            
            losses = self.model(phase_space, radiation)
            loss = losses['total_loss']

            # only logging from of the master process
            if self.batch_passes > 0:
                self.logger(losses, self.batch_passes-1)

                if self.checkpoint_interval and self.batch_passes % self.checkpoint_interval == 0:
                    save_checkpoint_conditionally(self.model, self.optimizer, self.out_prefix, self.batch_passes, losses)
            
            #reconstruction if needed
            # y, lat_z_pred = self.model.reconstruct(phase_space,radiation)
            
            loss.backward()
            
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-5.00, 5.00)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            if self.training_buffer.openpmdProduction == False:
                print(f"Note: The streaming has stopped, the trainer will run for "
                        f"{self.ts_after_stopped_production} training steps (batch passes) "
                        f"before stopping.\n"
                        f"Training step:{rest_training_left_counter} after the streaming has stopped.", flush=True)
                rest_training_left_counter+=1
                if rest_training_left_counter>self.ts_after_stopped_production:
                    if self.batch_passes > 0:
                        self.logger(losses, self.batch_passes-1) # log last batch
                        if self.checkpoint_final or ( self.checkpoint_interval and self.batch_passes % self.checkpoint_interval == 0 ):
                            save_checkpoint_conditionally(self.model, self.optimizer, self.out_prefix, self.batch_passes, losses)
                    break

        print("Training ended after {} samples in {} batches.".format(self.training_samples, self.batch_passes), flush=True)
