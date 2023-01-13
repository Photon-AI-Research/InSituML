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

path_to_particle_data = run_settings['path_to_particle_data']
path_to_radiation_data = run_settings['path_to_radiation_data']
paths_to_PS = [path_to_particle_data + '/' + next_file for next_file in os.listdir(path_to_particle_data)]
paths_to_PS.sort()
paths_to_radiation = [path_to_radiation_data + '/' + next_file for next_file in os.listdir(path_to_radiation_data)]
paths_to_radiation.sort()

dataset_tr = dataset.PCDataset(items_phase_space=paths_to_PS[:-1],
                 items_radiation=paths_to_radiation[:-1],
                 num_points=20,
                 num_files=-1,
                 chunk_size=int(run_settings['chunk_size']),
                 species=run_settings['species'],
                 normalize=True, a=0., b=1.)

dataset_val = dataset.PCDataset(items_phase_space=paths_to_PS[-1:],
                 items_radiation=paths_to_radiation[-1:],
                 num_points=20,
                 num_files=-1,
                 chunk_size=int(run_settings['chunk_size']),
                 species=run_settings['species'],
                 normalize=True, a=0., b=1.)

#keep batch size 1, number of points per batch is defined by chunk_size
loader_tr = loader.get_loader(dataset_tr, batch_size=1)
ps_dim = 0
radiation_dim = 0
for ps, rad in loader_tr:
    print(ps.shape)
    print(rad.shape)

    ps_dim = ps.shape[-1]
    radiation_dim = rad.shape[-1]
    break
    
model_f = model_cINN.PC_NF(dim_condition=radiation_dim,
                 dim_input=ps_dim,
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