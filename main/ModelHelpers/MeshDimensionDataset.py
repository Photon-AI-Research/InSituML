import torch
import numpy as np
from torch.utils.data import Dataset

class MeshDimensionDataset(Dataset):
    def __init__(self, iteration_id, dimensions, iteration_chunks, data_path):
        self.iteration_id = iteration_id
        self.dimensions = dimensions
        self.iteration_chunks = self._convert_to_numpy(iteration_chunks)
        self.data_path = data_path
        #use torch concat order = 0
        self.record_tensor = torch.from_numpy(self.iteration_chunks).view(1,self.iteration_chunks.shape[0],dimensions[0],dimensions[1],dimensions[2])

    def __len__(self):
        return len(self.record_tensor)

    def __getitem__(self, idx):
        #todo: any pre-processing
        return self.record_tensor[idx]
    
    def _convert_to_numpy(self, obj):
        return np.array(obj)#.astype('f')
    
    def save_data_set(self):
        np.save(self.data_path + '/data_' + str(self.iteration_id), self.iteration_chunks)

        
        