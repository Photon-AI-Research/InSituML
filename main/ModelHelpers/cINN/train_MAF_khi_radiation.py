#train_MAF_khi_box_v2-2-1-8-rad-ex-random
import os
import numpy as np
import torch
from model import model_MAF as model_MAF
import torch.optim as optim
import time
import wandb
import sys
import matplotlib.pyplot as plt

    
def sample_pointcloud(model, p_gt, num_samples, cond):
    model.model.eval()
    with torch.no_grad():
        pc_pr = (model.model.sample(num_samples, cond))
        
    return pc_pr


def random_sample(data, sample_size):
    
    # Check if the sample size is greater than the number of points in the data
    if sample_size > data.shape[0]:
        raise ValueError("Sample size exceeds the number of points in the data")

    random_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    sampled_data = data[random_indices]

    return sampled_data


def save_checkpoint(model, optimizer, path, last_loss, min_valid_loss, epoch, wandb_run_id):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_loss': last_loss.item(),
            'epoch': epoch,
            'min_valid_loss': min_valid_loss,
            'wandb_run_id': wandb_run_id,
        }

        torch.save(state, path + '/model_' + str(epoch))


class Loader:
    def __init__(self, pathpattern1="/bigdata/hplsim/aipp/Jeyhun/khi/particle_box/40_80_80_160_0_2/{}.npy", pathpattern2="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex/{}.npy", t0=0, t1=100, timebatchsize=20, particlebatchsize=10240):
        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        
        # TODO check if all files are there
        
        self.t0 = t0
        self.t1 = t1
        
        self.timebatchsize = timebatchsize
        self.particlebatchsize = particlebatchsize

        num_files = t1 - t0
        missing_files = [i for i in range(t0, t1) if not os.path.exists(pathpattern1.format(i))]
        num_missing = len(missing_files)
        all_files_exist = num_missing == 0

        if all_files_exist:
            print("All {} files from {} to {} exist in the directory.".format(num_files, t0, t1))
        else:
            print("{} files are missing out of {} in the directory.".format(num_missing, num_files))

    def __len__(self):
        return self.t1 - self.t0
        
    def __getitem__(self, idx):
        
        class Epoch:
            def __init__(self, loader, t0, t1, timebatchsize=20, particlebatchsize=10240):
                self.perm = torch.randperm(len(loader))
                self.loader = loader
                self.t0 = t0
                self.t1 = t1
                self.timebatchsize = timebatchsize
                self.particlebatchsize = particlebatchsize

            def __len__(self):
                return len(self.loader) // self.timebatchsize
        
            def __getitem__(self, timebatch):
                i = self.timebatchsize*timebatch
                bi = self.perm[i:i+self.timebatchsize]
                radiation = []
                particles = []
                for time in bi:
                    index = time + self.t0
                    
                    p = np.load(self.loader.pathpattern1.format(index), allow_pickle = True)
                    
                    p = [random_sample(element, sample_size=10000) for element in p]
                    p = torch.from_numpy(np.array(p, dtype = np.float32))
                    
                    p = p.view(p.shape[0],-1)
                    
                    r = torch.from_numpy(np.load(self.loader.pathpattern2.format(index)).astype(np.cfloat) )
                    r = r[:, 1:, :]
                    r = r.view(r.shape[0], -1)
                    
                    # Compute the phase (angle) of the complex number
                    phase = torch.angle(r)
                    
                    # Compute the absolute value of the complex number
                    absolute = torch.abs(r)
                    r = torch.cat((absolute, phase), dim=1).to(torch.float32)

                    particles.append( p )
                    radiation.append(r)
                
                particles = torch.cat(particles)
                radiation = torch.cat(radiation)
                
                class Timebatch:
                    def __init__(self, particles, radiation, batchsize):
                        self.batchsize = batchsize
                        self.particles = particles
                        self.radiation = radiation
                        
                        self.perm = torch.randperm(self.radiation.shape[0])
                        
                    def __len__(self):
                        return self.radiation.shape[0] // self.batchsize

                    def __getitem__(self, batch):
                        i = self.batchsize*batch
                        bi = self.perm[i:i+self.batchsize]
                    
                        return self.particles[bi], self.radiation[bi]
                
                return Timebatch(particles, radiation, self.particlebatchsize)
                    
        return Epoch(self, self.t0, self.t1, self.timebatchsize, self.particlebatchsize)


if __name__ == "__main__":
    
    hyperparameter_defaults = dict(
    t0 = 1000,
    t1 = 2001,
    timebatchsize = 4,
    particlebatchsize = 32,
    dim_condition = 2048,
    num_coupling_layers = 4,
    hidden_size = 256,
    lr = 0.00001,
    num_epochs = 20000,
    num_blocks_mat = 2,
    activation = 'gelu',
    #pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    #pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy"
    pathpattern1 = "/lustre/orion/csc380/proj-shared/vineethg/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/lustre/orion/csc380/proj-shared/vineethg/khi/part_rad/radiation_002_ex/{}.npy"
    )
    
    print('New session...')
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, project="khi_public")
    start_epoch = 0
    min_valid_loss = np.inf
    
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    
    pathpattern1 = config["pathpattern1"]
    pathpattern2 = config["pathpattern2"]
    
    l = Loader(pathpattern1=pathpattern1, pathpattern2=pathpattern2, t0=config["t0"], t1=config["t1"], timebatchsize=config["timebatchsize"], particlebatchsize=config["particlebatchsize"])
    
    model = (model_MAF.PC_MAF(dim_condition=config["dim_condition"],
                               dim_input=90000,
                               num_coupling_layers=config["num_coupling_layers"],
                               hidden_size=config["hidden_size"],
                               device='cuda',
                               enable_wandb=False,
                               weight_particles=False,
                               num_blocks_mat = config["num_blocks_mat"],
                               activation = config["activation"]
                             ))
    
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    
    #directory ='/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/'+str(wandb.run.id)
    directory ='/lustre/orion/csc380/world-shared/checkpoints/maf_khi/'+str(wandb.run.id)

    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    
    
    epoch = l[0]
    
    patience = 20
    slow_improvement_patience = 10
    no_improvement_count = 0
    slow_improvement_count = 0

    start_time = time.time()
    for i_epoch in range(start_epoch, config["num_epochs"]):   
        #print('i_epoch:', i_epoch)
        loss_overall = []
        for tb in range(len(epoch)):
            loss_avg = []
            timebatch = epoch[tb]
            
            start_timebatch = time.time()
            for b in range(len(timebatch)):
                optimizer.zero_grad()
                #print('b',b)
                phase_space, radiation = timebatch[b]
                
                print('phase_space', phase_space.shape)
                print('radiation', radiation.shape)
                
                loss = - model.model.log_prob(inputs=phase_space.to(model.device),context=radiation.to(model.device))

                loss = loss.mean()
                loss_avg.append(loss.item())
                loss.backward()
                optimizer.step()
                
            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch
            
            loss_timebatch_avg = sum(loss_avg)/len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print('i_epoch:{}, tb: {}, last timebatch loss: {}, avg_loss: {}, time: {}'.format(i_epoch,tb,loss.item(), loss_timebatch_avg, elapsed_timebatch))
    
        loss_overall_avg = sum(loss_overall)/len(loss_overall)  
    
        if min_valid_loss > loss_overall_avg:     
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss_overall_avg:.6f}) \t Saving The Model')
            min_valid_loss = loss_overall_avg
            no_improvement_count = 0
            slow_improvement_count = 0
            # Saving State Dict
            torch.save(model.state_dict(), directory + '/best_model_', _use_new_zipfile_serialization=False)
        else:
            no_improvement_count += 1
            if loss_overall_avg - min_valid_loss <= 0.001:  # Adjust this threshold as needed
                slow_improvement_count += 1
            
            save_checkpoint(model, optimizer, directory, loss, min_valid_loss, i_epoch, wandb.run.id)
        
        scheduler.step()
        
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": i_epoch,
            #"last time batch loss":loss.item(),
            "loss_timebatch_avg_loss": loss_timebatch_avg,
            "loss_overall_avg": loss_overall_avg,
            "min_valid_loss": min_valid_loss,
        })
            
        
        # if no_improvement_count >= patience or slow_improvement_count >= slow_improvement_patience:
        #     break  # Stop training
        
    # Code or process to be measured goes here
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")
