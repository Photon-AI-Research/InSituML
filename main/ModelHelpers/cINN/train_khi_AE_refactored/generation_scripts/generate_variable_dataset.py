import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from data_loaders import TrainLoader
import argparse
from geomloss import SamplesLoss
from scipy.spatial import distance
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emd = SamplesLoss(loss="sinkhorn", p=1, blur=.01)

class GenerateVariableDataset:
    def __init__(self, pathpattern1, pathpattern2,
                 t0, t1, timebatchsize, particlebatchsize, 
                 particles_to_sample, size_of_variable_unit,
                 num_iterations, tolerance_distance, outputfile_name):
        
        self.variable_units = torch.Tensor([]).to(device)
        self.current_minimum = -1 
        self.outputfile_name = outputfile_name
        self.tolerance_distance = tolerance_distance
        self.num_iterations = num_iterations
        self.size_of_variable_unit = size_of_variable_unit
        
        self.data_loader = TrainLoader(pathpattern1=pathpattern1,
                              pathpattern2=pathpattern2,
                              t0=t0, t1=t1,
                              timebatchsize = timebatchsize,
                              particlebatchsize = particlebatchsize,
                              particles_to_sample = particles_to_sample)[0]

        
        
    def save_files(self):
        
        torch.save(self.variable_units, "variable_units_"+self.outputfile_name)
        torch.save(self.relative_dis, "relative_dis_"+self.outputfile_name)

    def iterate_over_batch_examples(self, phase_space):
        for idx in range(len(phase_space)):
            self.check_and_add(phase_space[idx:idx+1])
            
            if (self.current_minimum > self.tolerance_distance):
                self.save_files()
                return True
        
        return False
                
    def cdist(self, X, Y, compute, same=True):
        
        relative_distances = []
        for idx_x, x in enumerate(X):
            for idx_y, y in enumerate(Y):
                if same and (idx_x==idx_y):
                    continue
                relative_distances.append([idx_x, idx_y, compute(x,y)])
        
        return torch.Tensor(relative_distances).to(device)
                
                
    def check_and_add(self,phase_space):
        
        if len(self.variable_units)<self.size_of_variable_unit:
            self.variable_units = torch.cat([phase_space, self.variable_units])
            return
        print(self.variable_units.shape)
        self.relative_dis = self.cdist(self.variable_units, self.variable_units, compute=emd)
        relative_dis_new = self.cdist(self.variable_units, phase_space, compute=emd, same=False)
        min_already, idx_row = torch.min(self.relative_dis[:,2], dim=0)

        min_new, idx_row_new = torch.min(relative_dis_new[:,2], dim=0)
        print("new_min", min_new, "already_min", min_already)

        if min_new > min_already:
            print("inside")
            
            [idx_1, idx_2,_] = self.relative_dis[idx_row]
            
            idx_remove = int(random.sample([idx_1, idx_2], 1)[0].item())

            self.current_minimum = min_new
            
            self.variable_units = torch.cat([self.variable_units[:idx_remove], 
                                            self.variable_units[idx_remove+1:],
                                            phase_space])
            
            self.relative_dis = self.cdist(self.variable_units, 
                                           self.variable_units, 
                                           compute=emd)
        else:
            #only needed for the first iteration
            self.current_minimum = min_already
            
    def reiterate_training_batches(self):

        for _ in range(self.num_iterations):
        
            for timeBatchIndex in range(len(self.data_loader)):
                
                timeBatch = self.data_loader[timeBatchIndex]
                
                for particleBatchIndex in range(len(timeBatch)):
                    phase_space, _ = timeBatch[particleBatchIndex]
                    finished = self.iterate_over_batch_examples(phase_space.to(device))
                    
                    if finished:
                        return None
        
        self.save_files()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""For generating the variable dataset"""
    )
    
    parser.add_argument('--pathpattern1',
                        type=str,
                        default="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
                        help="Path pattern for particles.")

    parser.add_argument('--pathpattern2',
                        type=str,
                        default="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
                        help="Path pattern for particles.")

    parser.add_argument('--t0',
                        type=int,
                        default=1000,
                        help="Initial simulation timestep to pick up")
    
    parser.add_argument('--t1',
                        type=int,
                        default=2001,
                        help="Last simulation timestep to pick up")

    parser.add_argument('--timebatchsize',
                        type=int,
                        default=4,
                        help="Time batch size")

    parser.add_argument('--particlebatchsize',
                        type=int,
                        default=4,
                        help="Particle batch size")

    parser.add_argument('--particles_to_sample',
                        type=int,
                        default=4000,
                        help="Particle to sample")

    parser.add_argument('--size_of_variable_unit',
                        type=int,
                        default=10,
                        help="Size of dataset to produce")

    parser.add_argument('--tolerance_distance',
                        type=int,
                        default=1,
                        help="tolerance minimum emd distance between the dataset points")

    parser.add_argument('--num_iterations',
                        type=int,
                        default=10,
                        help="""Number of iterations over the training batches
                                This would correspond to epoch in training case.""")

    parser.add_argument('--outputfile_name',
                        type=str,
                        default="dataset.pt",
                        help="File ending names for tensors to be saved on disk")

    args = parser.parse_args()
    
    generator = GenerateVariableDataset(**vars(args))
    generator.reiterate_training_batches()
