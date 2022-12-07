import sys
sys.path.append('./model')
sys.path.append('./model/modules')

import dataset
import loader
import model_cINN 

import torch

paths_to_PS = ["./from_cloud/simData_68500.bp", "./from_cloud/simData_68500.bp"]
paths_to_radiation = ["./from_cloud/b_radAmplitudes_68500_0_0_0.h5", "./from_cloud/b_radAmplitudes_68500_0_0_0.h5"]

dataset_tr = dataset.PCDataset(items_phase_space=paths_to_PS,
                 items_radiation=paths_to_radiation,
                 num_points=100,
                 num_files=-1,
                 chunk_size=20,
                 normalize=True, a=0., b=1.)

loader_tr = loader.get_loader(dataset_tr, batch_size=1)
for ps, rad in loader_tr:
    print(ps.shape)
    print(rad.shape)
    break
    
model_f = model_cINN.PC_NF(dim_condition=440*2048*6,
                 dim_input=6,
                 num_coupling_layers=1,
                 num_linear_layers=1,
                 hidden_size=32,
                 device='cuda',
                 enable_wandb=False)

optimizer = torch.optim.Adam(model_f.trainable_parameters, lr=1e-3, betas=(0.8, 0.9),
                             eps=1e-6, weight_decay=2e-5)
test_pointcloud = '...'

model_f.train_(dataset_tr,
               dataset_tr,
               optimizer,
               epochs=1,
               batch_size=1,
               test_epoch=25,
               test_pointcloud=test_pointcloud, log_plots=None,
               path_to_models='./RESmodels_cinn/')