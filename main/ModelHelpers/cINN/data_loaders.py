import numpy as np
import torch
from utilities import Normalizer, random_sample
import os


class TrainLoader:
    def __init__(self, 
                 normalisation,
                 norm_method,
                 pathpattern1="/bigdata/hplsim/aipp/Jeyhun/khi/particle_box/40_80_80_160_0_2/{}.npy",
                 pathpattern2="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex/{}.npy",
                 t0=0,
                 t1=100,
                 timebatchsize=20,
                 particlebatchsize=10240,
                 particles_to_sample=150000,
                 blacklist_box=None,
                 data_stats_path ='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_002/global_stats.npz'):
        self.normalisation = normalisation
        self.norm_method = norm_method
        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        self.blacklist_box = blacklist_box
        
        # TODO check if all files are there
        
        self.t0 = t0
        self.t1 = t1
        
        self.timebatchsize = timebatchsize
        self.particlebatchsize = particlebatchsize
        self.particles_to_sample = particles_to_sample

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
            def __init__(self,
                         loader,
                         t0,
                         t1,
                         timebatchsize=20,
                         particlebatchsize=10240,
                         blacklist_box=None):
                self.perm = torch.randperm(len(loader))
                self.loader = loader
                self.t0 = t0
                self.t1 = t1
                self.timebatchsize = timebatchsize
                self.particlebatchsize = particlebatchsize
                self.blacklist_box = blacklist_box

            def __len__(self):
                return len(self.loader) // self.timebatchsize
        
            def __getitem__(self, timebatch):
                i = self.timebatchsize*timebatch
                bi = self.perm[i:i+self.timebatchsize]
                radiation = []
                particles = []
                for time in bi:
                    index = time + self.t0
                    
                    # Load particle data
                    p = np.load(self.loader.pathpattern1.format(index), allow_pickle = True)
                    # Load radiation data
                    r = torch.from_numpy(np.load(self.loader.pathpattern2.format(index)).astype(np.cfloat))
                    
                    # Exclude the blacklisted box if specified
                    # if self.blacklist_box is not None:
                    #     p = np.delete(p, self.blacklist_box, axis=0)
                    #     r = np.delete(r, self.blacklist_box, axis=0)
                    
                    p = [self.loader.normalisation.normalize_data(element, method=self.loader.norm_method) for element in p]
                    p = np.array(p, dtype=object)
                    
                    # random sample N points from each box
                    p = [random_sample(element, sample_size=self.loader.particles_to_sample) for element in p]
                    p = torch.from_numpy(np.array(p, dtype = np.float32))
                    
                    # ampliudes in each direction
                    amp_x = torch.abs(r[:, 0, :]).to(torch.float32)
                    amp_y = torch.abs(r[:, 1, :]).to(torch.float32)
                    amp_z = torch.abs(r[:, 2, :]).to(torch.float32)
                    
                    #spectra
                    r = amp_x**2 + amp_y**2 + amp_z**2
                    
                    #log transformation
                    r = torch.log(r+1)

#                     # choose relevant directions
#                     r = r[:, 1:, :]
#                     r = r.view(r.shape[0], -1)
                    
#                     # Compute the phase (angle) of the complex number in radians
#                     phase = torch.angle(r)
                    
#                     # Compute the amplitude (magnitude) of the complex number
#                     amplitude = torch.abs(r)
#                     r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

                    particles.append(p)
                    radiation.append(r)
                
                # concatenate particle and radiation data across randomly chosen timesteps
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
                    
        return Epoch(self, self.t0, self.t1, self.timebatchsize, self.particlebatchsize, self.blacklist_box)


class ValidationFixedBoxLoader:
    def __init__(self, 
                 pathpattern1,
                 pathpattern2,
                 validation_boxes,
                 normalisation,
                 norm_method,
                 t0=0,
                 t1=100,
                 particles_to_sample=4000,
                 select_timesteps=20,
                 load_radiation = False
          ):
        
        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        self.validation_boxes = validation_boxes
        self.t0 = t0
        self.t1 = t1
        self.particles_to_sample = particles_to_sample
        self.select_timesteps = select_timesteps
        self.load_radiation = load_radiation
        self.normalisation = normalisation
        self.norm_method = norm_method
    
    def __len__(self):
        self.perm =  torch.randperm((self.t1-self.t0))[:self.select_timesteps]
        return self.select_timesteps if len(self.validation_boxes) else 0

    def __getitem__(self, idx):

        timestep_index = self.t0 + self.perm[idx]
     
        # Load particle data for the validation boxes
        p_loaded = np.load(self.pathpattern1.format(timestep_index), allow_pickle=True)
        p = [p_loaded[box_index] for box_index in self.validation_boxes]
        
        #p = [normalize_columns(element) for element in p]
        p = [self.normalisation.normalize_data(element, method=self.norm_method) for element in p]
        p = np.array(p, dtype=object)
        
        p = [random_sample(element, 
                           sample_size=self.particles_to_sample) for element in p]
        
        p = torch.from_numpy(np.array(p, dtype = np.float32))        
        
        if self.load_radiation:
            # Load radiation data for the validation boxes
            r = torch.from_numpy(np.load(self.pathpattern2.format(timestep_index)).astype(np.cfloat))
            
            # ampliudes in each direction
            amp_x = torch.abs(r[:, 0, :]).to(torch.float32)
            amp_y = torch.abs(r[:, 1, :]).to(torch.float32)
            amp_z = torch.abs(r[:, 2, :]).to(torch.float32)

            #spectra
            r = amp_x**2 + amp_y**2 + amp_z**2

            #log transformation
            r = torch.log(r+1)
            
            r = r[self.validation_boxes, :]
            
            
            #r = r_loaded[self.validation_boxes, 1:, :]
#             r = r.view(r.shape[0], -1)
            
#             # Compute the phase (angle) of the complex number in radians
#             phase = torch.angle(r)

#             # Compute the amplitude (magnitude) of the complex number
#             amplitude = torch.abs(r)
#             r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

            return timestep_index, self.validation_boxes, p, r
        
        else:
            return timestep_index, self.validation_boxes, p, None
        
        
