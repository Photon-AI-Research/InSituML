import os

import torch
import wandb

from model import model_MAF
from model.modules import data_preprocessing
from model.modules import dataset_supercell as dataset
from model.modules import dist_utils
from model.modules import loader
from model.modules import utils
from model.modules import visualizations

dist_utils.maybe_initialize()
run_settings = utils.load_run_settings('cfg.txt')

if bool(run_settings['enable_wandb']):

    config_defaults = {
        'chunk_size' : int(run_settings['chunk_size']),
        'num_coupling_layers' : int(run_settings['num_coupling_layers']),
        'num_linear_layers_in_subnetworks' : int(run_settings['num_linear_layers_in_subnetworks']),
        'hidden_size_of_layers_in_subnetworks' : int(run_settings['hidden_size_of_layers_in_subnetworks']),
        'learning_rate' : float(run_settings['learning_rate'])
    }

    os.environ['WANDB_API_KEY'] = run_settings['wandb_apikey']

    if dist_utils.is_rank_0():
        wandb.init(reinit=True, project=run_settings['wandb_project'],
                   entity=run_settings['wandb_entity'], config=config_defaults)

path_to_particle_data = "/bigdata/hplsim/aipp/Anna/lwfa_supercells/10700_40_1123_101.npy"


dataset_tr = dataset.PCDataset(item_phase_space=path_to_particle_data,
                               normalize=True, a=0., b=1.)

model_f = model_MAF.PC_MAF(dim_condition=2,
                           dim_input=6,
                           num_coupling_layers=int(run_settings['num_coupling_layers']),
                           hidden_size=int(run_settings['hidden_size_of_layers_in_subnetworks']),
                           device='cuda',
                           enable_wandb=bool(run_settings['enable_wandb']))
#model_f.model = dist_utils.maybe_ddp_wrap(model_f.model)

optimizer = torch.optim.Adam(model_f.model.parameters(), lr=float(run_settings['learning_rate']),
                             betas=(0.8, 0.9), eps=1e-6, weight_decay=2e-5)
#test_pointcloud = "/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/openPMD/simData_e_010700.bp"
#test_pointcloud = '/bigdata/hplsim/aipp/Anna/lwfa_macrocells/test_mc/10700_1_1_6.npy'
#test_pointclouds = "/bigdata/hplsim/aipp/Anna/lwfa_tests/10700_44_1151_51.npy"
test_pointclouds = "/bigdata/hplsim/aipp/Anna/lwfa_supercells/10700_40_1123_101.npy"

#test_pointclouds = ['/bigdata/hplsim/aipp/Anna/lwfa_tests/'+nextfile for nextfile in os.listdir('/bigdata/hplsim/aipp/Anna/lwfa_tests')]

#test_radiation = ["/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/radiationOpenPMD/e_radiation_10700_0_0_0.h5" for i in range(len(test_pointclouds))]
test_radiation = "/home/willma32/insitu_particles/InSituML/main/ModelHelpers/cINN/rad_test.npy"

#log_plots = visualizations.log_plots
log_plots = visualizations.log_one_plot

model_f.train_(dataset_tr,
               dataset_tr,
               optimizer,
               epochs=10001,
               batch_size=100000,
               test_epoch=250,
               test_pointcloud=test_pointclouds, test_radiation=test_radiation, log_plots=log_plots,
               path_to_models=('/bigdata/hplsim/aipp/Anna/res_models/RESmodelsMAF1Supercell'
                                +'_'.join([run_settings['num_coupling_layers'],
                                           run_settings['num_linear_layers_in_subnetworks'],
                                           run_settings['hidden_size_of_layers_in_subnetworks']])+'/'))