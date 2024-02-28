import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from data_loaders import TrainLoader
import argparse
from geomloss import SamplesLoss
from scipy.spatial import distance
import random

emd = SamplesLoss(loss="sinkhorn", p=1, blur=.01)

def check_and_add(phase_space, variable_units):
    
    #add the first two
    if (variable_units)<2:
        variable_units = torch.cat([variable_units, phase_space])
        return False
    
    relative_dis = distances.cdist(variable_units, variable_units, metric=emd)

    relative_dis_new = distances.cdist(variable_units, [phase_space], metric=emd)
    
    min_already = relative_dis.min()

    min_new = relative_dis_new.min()
    
    if min_new < min_already:
        idx_min = random([relative_dis.argmin()//len(variable_units), 
                      relative_dis.argmin()%len(variable_units], 1)
        
        return torch.cat([variable_units[:idx_min], 
                          variable_units[idx_min+1:], 
                          [phase_space]])
    else:
        return variable_units
    
def reiterate_training_batches(
    pathpattern1, pathpattern2,
    t0, t1, timebatchsize, particlebatchsize, 
    particles_to_sample, size_of_variable_unit = 10):
    
    variable_units = torch.Tensor([])
    
    data_loader = TrainLoader(pathpattern1=pathpattern1,
                              pathpattern2=pathpattern2,
                              t0=t0, t1=t1,
                              timebatchsize = timebatchsize,
                              particlebatchsize = particlebatchsize,
                              particles_to_sample = particles_to_sample)[0]
    
    for timeBatchIndex in range(len(data_loader)):
        
        timeBatch = data_loader[timeBatchIndex]
        
        for particleBatchIndex in range(len(timeBatch)):
            phase_space, _ = timeBatch[particleBatchIndex]
            variable_units = check_and_add(phase_space, variable_units)
            
            if variable_units >= size_of_variable_unit:
                return variable_units

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

    args = parser.parse_args()
    
    reiterate_training_batches(**vars(args))

