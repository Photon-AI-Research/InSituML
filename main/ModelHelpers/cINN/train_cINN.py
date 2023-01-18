import os

import torch
import wandb

from .model import model_cINN
from .model.modules import data_preprocessing
from .model.modules import dataset
from .model.modules import loader
from .model.modules import visualizations

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

#if there is only 1 simulation in direcotry
#dublicate it in the list of paths to simulation files
#because last simulation is used as validation data and all before as training data
#in case of 1 simulation, we use it for training and for validation

if len(paths_to_PS) == 1:
    paths_to_PS.append(paths_to_PS[0])
    paths_to_radiation.append(paths_to_radiation[0])

print(paths_to_PS)
print(paths_to_radiation)

dataset_tr = dataset.PCDataset(items_phase_space=paths_to_PS[:1],
                               items_radiation=paths_to_radiation[:1],
                               num_points=-1,
                               num_files=-1,
                               chunk_size=int(run_settings['chunk_size']),
                               species=run_settings['species'],
                               normalize=True, a=0., b=1.)

dataset_val = dataset.PCDataset(items_phase_space=paths_to_PS[:1],
                                items_radiation=paths_to_radiation[:1],
                                num_points=-1,
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

print('radiation dim ', radiation_dim)
model_f = model_cINN.PC_NF(dim_condition=2,
                           dim_input=ps_dim,
                           num_coupling_layers=int(run_settings['num_coupling_layers']),
                           num_linear_layers=int(run_settings['num_linear_layers_in_subnetworks']),
                           hidden_size=int(run_settings['hidden_size_of_layers_in_subnetworks']),
                           device='cuda',
                           enable_wandb=bool(run_settings['enable_wandb']))

optimizer = torch.optim.Adam(model_f.trainable_parameters, lr=float(run_settings['learning_rate']),
                             betas=(0.8, 0.9), eps=1e-6, weight_decay=2e-5)
#test_pointcloud = "/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/openPMD/simData_e_010700.bp"
#test_pointcloud = '/bigdata/hplsim/aipp/Anna/lwfa_macrocells/test_mc/10700_1_1_6.npy'
#test_pointclouds = ['/bigdata/hplsim/aipp/Anna/lwfa_supercells/10700_40_1123_101.npy',
#              '/bigdata/hplsim/aipp/Anna/lwfa_supercells/10700_24_1237_85.npy']
test_pointclouds = ['/bigdata/hplsim/aipp/Anna/lwfa_tests/'+nextfile for nextfile in os.listdir('/bigdata/hplsim/aipp/Anna/lwfa_tests')]

test_radiation = ["/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/radiationOpenPMD/e_radiation_10700_0_0_0.h5" for i in range(len(test_pointclouds))]

#log_plots = visualizations.log_plots
log_plots = visualizations.log_min_max_each

model_f.train_(dataset_tr,
               dataset_tr,
               optimizer,
               epochs=30001,
               batch_size=1,
               test_epoch=1000,
               test_pointcloud=test_pointclouds, test_radiation=test_radiation, log_plots=log_plots,
               path_to_models=('/bigdata/hplsim/aipp/Anna/res_models/RESmodels'
                                +'_'.join([run_settings['num_coupling_layers'],
                                           run_settings['num_linear_layers_in_subnetworks'],
                                           run_settings['hidden_size_of_layers_in_subnetworks']])+'/'))
