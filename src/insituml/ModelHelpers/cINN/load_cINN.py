import sys
sys.path.append('./model')
sys.path.append('./model/modules')
import os

import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import model_cINN 
import data_preprocessing

path_to_model = '/bigdata/hplsim/aipp/Anna/res_models/RESmodels7_1_512/model_9000'

test_radiation = "/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/radiationOpenPMD/e_radiation_10700_0_0_0.h5"
path_to_minmax = '/bigdata/hplsim/aipp/Anna/minmax/'

print('Prepare model...')
model_f = model_cINN.PC_NF(dim_condition=2,
                           dim_input=6,
                           num_coupling_layers=7,
                           num_linear_layers=1,
                           hidden_size=512,
                           device='cuda',
                           enable_wandb=False)

checkpoint = torch.load(path_to_model)
model_f.model.load_state_dict(checkpoint['model'])

print('Model is loaded')
model_f.vmin_ps, model_f.vmax_ps, model_f.a, model_f.b = torch.from_numpy(np.load(path_to_minmax+'vmin_ps.npy')), torch.from_numpy(np.load(path_to_minmax+'vmax_ps.npy')), torch.Tensor([0.]), torch.Tensor([1.])
vmin_rad, vmax_rad = np.load(path_to_minmax+'vmin_rad.npy'),np.load(path_to_minmax+'vmax_rad.npy')
model_f.eval()

num_points_to_sample = 1000

radiation_tensor = torch.Tensor([[1.0264e-13, 8.7699e-14]]).repeat(num_points_to_sample, 1)
radiation_tensor = data_preprocessing.normalize_point(radiation_tensor, 
                                                      torch.torch.full(radiation_tensor.shape, vmin_rad[0,0]), 
                                                      torch.torch.full(radiation_tensor.shape, vmax_rad[0,0]), 
                                                      torch.torch.full(radiation_tensor.shape, 0.), 
                                                      torch.torch.full(radiation_tensor.shape, 1.))
model_f.model.eval()

with torch.no_grad():
    z = torch.randn(num_points_to_sample, 6).to(model_f.device)
    pred_pointcloud, _ = model_f.model(z, c=radiation_tensor.to(model_f.device), rev=True)
    pred_pointcloud = data_preprocessing.denormalize_point(pred_pointcloud.to('cpu'), model_f.vmin_ps, model_f.vmax_ps, model_f.a, model_f.b)

pred_pointcloud = pred_pointcloud.detach().cpu().numpy()