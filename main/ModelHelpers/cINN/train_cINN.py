import sys
sys.path.append('./model')
sys.path.append('./model/modules')
import os

import torch
import wandb

import dataset
import loader
import model_cINN 

run_settings = {}

#last line in cfg.txt file should be empty
with open('cfg.txt') as f:
    lines = f.readlines()

for line in lines:
    key, value = line.split(': ')
    run_settings[key] = value[:-1]

print(run_settings)

if bool(run_settings['enable_wandb']):

    config_defaults = {
        'chunk_size' : int(run_settings['chunk_size']),
        'num_coupling_layers' : int(run_settings['num_coupling_layers']),
        'num_linear_layers_in_subnetworks' : int(run_settings['num_linear_layers_in_subnetworks']),
        'hidden_size_of_layers_in_subnetworks' : int(run_settings['hidden_size_of_layers_in_subnetworks']),
        'learning_rate' : float(run_settings['learning_rate'])
    }

    os.environ['WANDB_API_KEY'] = run_settings['wandb_apikey']

    wandb.init(reinit=True, project=run_settings['wandb_project'],
               entity=run_settings['wandb_entity'], config=config_defaults)

path_to_example_data = run_settings['path_to_example_data']
paths_to_PS = [path_to_example_data + "/from_cloud/simData_68500.bp",
               path_to_example_data + "/from_cloud/simData_68500.bp"]
paths_to_radiation = [path_to_example_data + "/from_cloud/b_radAmplitudes_68500_0_0_0.h5",
                      path_to_example_data + "/from_cloud/b_radAmplitudes_68500_0_0_0.h5"]

dataset_tr = dataset.PCDataset(items_phase_space=paths_to_PS,
                 items_radiation=paths_to_radiation,
                 num_points=100,
                 num_files=-1,
                 chunk_size=int(run_settings['chunk_size']),
                 normalize=True, a=0., b=1.)

#keep batch size 1, number of points per batch is defined by chunk_size
loader_tr = loader.get_loader(dataset_tr, batch_size=1)
for ps, rad in loader_tr:
    print(ps.shape)
    print(rad.shape)
    break
    
model_f = model_cINN.PC_NF(dim_condition=440*2048*6,
                 dim_input=6,
                 num_coupling_layers=int(run_settings['num_coupling_layers']),
                 num_linear_layers=int(run_settings['num_linear_layers_in_subnetworks']),
                 hidden_size=int(run_settings['hidden_size_of_layers_in_subnetworks']),
                 device='cuda',
                 enable_wandb=bool(run_settings['enable_wandb']))

optimizer = torch.optim.Adam(model_f.trainable_parameters, lr=float(run_settings['learning_rate']),
                             betas=(0.8, 0.9), eps=1e-6, weight_decay=2e-5)
test_pointcloud = '...'

model_f.train_(dataset_tr,
               dataset_tr,
               optimizer,
               epochs=1,
               batch_size=1,
               test_epoch=25,
               test_pointcloud=test_pointcloud, log_plots=None,
               path_to_models='./RESmodels_cinn/')