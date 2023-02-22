import torch
import numpy as np
from torch.utils.data import Dataset

class MeshDimensionDataset(Dataset):
    def __init__(self, iteration_id, dimensions, iteration_chunks, data_path, min_norm, max_norm, batch_size = 1):
        self.iteration_id = iteration_id
        self.dimensions = dimensions
        self.iteration_chunks =self._convert_to_numpy(iteration_chunks)
        self.data_path = data_path
        self.batch_size = batch_size
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.record_tensor = torch.from_numpy(self.iteration_chunks).view(len(iteration_chunks),-1,dimensions[0],dimensions[1],dimensions[2])

    def __len__(self):
        return len(self.record_tensor)

    def __getitem__(self, idx):
        norm_tensor = self.record_tensor[idx]
        #norm_tensor = torch.sub(norm_tensor, self.min_norm)
        #norm_tensor = torch.div(norm_tensor, self.max_norm - self.min_norm)
        return norm_tensor[1].view(-1,self.dimensions[0],self.dimensions[1],self.dimensions[2])
    
    def _convert_to_numpy(self, obj):
        return np.array(obj).astype('f')
    
    def save_data_set(self):
        np.save(self.data_path + '/data_' + str(self.iteration_id), self.iteration_chunks)

        
        