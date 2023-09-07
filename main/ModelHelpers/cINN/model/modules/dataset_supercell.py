import torch
from torch.utils.data import Dataset
import numpy as np

from . import data_preprocessing


class PCDataset(Dataset):
    def __init__(self, 
                 item_phase_space,
                 normalize=False, a=0., b=1.):
        '''
        Prepare dataset
        Args:
            items_phase_space(list of string): list of paths to files with particle phase space data
            items_radiation(list of string): list of paths to files with radiation, the order of paths should 
                                             correspond the order of paths in items_phase_space           
            num_points(integer): number of points to sample from each electron cloud,
                                 if -1 then take a complete electron cloud 
            num_files(integer): number of files to take for a dataset
            chunk_size(integer): number of particles to load per time
                                 (a complete point cloud does not pass into the memory)
            species(string): name of particle species to be loaded from openPMD file
            normalize(boolean): True if normalize each point to be in range [a, b]
        '''

        self.normalize = normalize
        self.a, self.b = a, b 

        self.item = torch.from_numpy(np.loadtxt(item_phase_space)).float()

        if normalize:
            self.vmin_ps, _ = torch.min(self.item, 0)
            self.vmax_ps, _ = torch.max(self.item, 0)
            self.vmin_rad, self.vmax_rad = torch.Tensor([0.05, 0.1]).float(), torch.Tensor([0.15, 0.3]).float()
            
            #print('PS Minima: ')
            #print('\t', self.vmin_ps)
            
            #print('PS Maxima: ')
            #print('\t', self.vmax_ps)
            #self.item = data_preprocessing.normalize_point(self.item, self.vmin_ps, self.vmax_ps, self.a, self.b)

        
    def __getitem__(self, index):
        return (self.item[index, :],
                torch.Tensor([0.1, 0.2]).float())

    def __len__(self):
        return self.item.shape[0]
