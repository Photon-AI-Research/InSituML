import numpy as np
import torch
from utilities import random_sample, normalize_columns
import os

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
                    
                    # Load particle data
                    p = np.load(self.loader.pathpattern1.format(index), allow_pickle = True)
                    
                    # Normalize particle data for all boxes
                    p = [normalize_columns(element) for element in p]
                    p = np.array(p, dtype=object)
                    
                    # random sample N points from each box
                    p = [random_sample(element, sample_size=150000) for element in p]
                    p = torch.from_numpy(np.array(p, dtype = np.float32))
                    
                    # Load radiation data
                    r = torch.from_numpy(np.load(self.loader.pathpattern2.format(index)).astype(np.cfloat) )
                    r = r[:, 1:, :]
                    r = r.view(r.shape[0], -1)
                    
                    # Compute the phase (angle) of the complex number in radians
                    phase = torch.angle(r)
                    
                    # Compute the amplitude (magnitude) of the complex number
                    amplitude = torch.abs(r)
                    r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

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
                    
        return Epoch(self, self.t0, self.t1, self.timebatchsize, self.particlebatchsize)

