import torch
from torch.utils.data import Dataset
import numpy as np

from . import data_preprocessing


class PCDataset(Dataset):
    def __init__(self, 
                 items_phase_space,
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

        self.items_ps = items_phase_space
        self.get_time = data_preprocessing.time_to_hotvec

        if normalize:
            for j,item_ps in enumerate(self.items_ps):
                arr = np.loadtxt(item_ps)
                if j==0:
                    self.vmin_ps = [np.min(arr[:, i]) for i in range(arr.shape[1])]
                    self.vmax_ps = [np.max(arr[:, i]) for i in range(arr.shape[1])]
                else:
                    self.vmin_ps = [min(np.min(arr[:, i]), self.vmin_ps[i]) for i in range(arr.shape[1])]
                    self.vmax_ps = [max(np.max(arr[:, i]), self.vmax_ps[i]) for i in range(arr.shape[1])]

            self.vmin_ps = torch.Tensor(self.vmin_ps).float()
            self.vmax_ps = torch.Tensor(self.vmax_ps).float()
            
            print('PS Minima: ')
            print('\t', self.vmin_ps)

            print('PS Maxima: ')
            print('\t', self.vmax_ps)
            
            self.timesteps = ['10000', '10100', '10200', '10300', '10400', '10500', '10600', '10700', '10800', '10900', '11000']
            
    def __getitem__(self, index):
        x = np.loadtxt(self.items_ps[index])
        return (torch.Tensor(x).float(),
                self.get_time(data_preprocessing.get_time(self.items_ps[index]), self.timesteps))

    def __len__(self):
        return len(self.items_ps)
