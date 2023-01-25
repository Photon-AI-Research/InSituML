import numpy as np
import torch

def generate(labels, tot_dataset_size):
    # print('Generating artifical data for setup "%s"' % (labels))

    verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

    label_maps = {
                  'all':  [0, 1, 2, 3, 4, 5, 6, 7],
                  'some': [0, 0, 0, 0, 1, 1, 2, 3],
                  'none': [0, 0, 0, 0, 0, 0, 0, 0],
                 }
    
    np.random.seed(0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.
   
    pos = torch.tensor(pos, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)

    return pos, labels

def generate_timeDependent(labels, tot_dataset_size, fctn, t = np.arange(10)):

    verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

    label_maps = {
                  'all':  [0, 1, 2, 3, 4, 5, 6, 7],
                  'some': [0, 0, 0, 0, 1, 1, 2, 3],
                  'none': [0, 0, 0, 0, 0, 0, 0, 0],
                 }
    
    np.random.seed(0)
    
    pos, labels = generate(labels, tot_dataset_size)

    pos_t, labels_t = pos, labels
    
    for ti in t:
        pos_t = torch.cat([pos_t, pos+fctn(ti)])
        labels_t = torch.cat([labels_t, (ti+1)*labels]) # 

    return pos_t, labels_t