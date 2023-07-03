import os

import torch
import numpy as np

#from model import model_cINN_without_distr as model_cINN
from model import model_MAF
from model.modules import data_preprocessing
from model.modules import dataset_supercell as dataset
from model.modules import loader
from model.modules import utils
from model.modules import visualizations

import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as tick
from matplotlib import cm

# Module for deterministic profiling of Python scripts/programs
import cProfile
import pstats
# Module for timing
import time
# Module for dealing with various types of I/O
import io
# Module Python codes profiling
import snakeviz

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['lines.linewidth']=6
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

supercells = ['/bigdata/hplsim/aipp/Anna/lwfa_2cells/'+nextfile for nextfile in os.listdir('/bigdata/hplsim/aipp/Anna/lwfa_2cells')]
datasets = []
models = []

for ind in range(1):
    datasets.append(dataset.PCDataset(item_phase_space=supercells[ind],
                                   normalize=True, a=0., b=1.))
    models.append(model_MAF.PC_MAF(dim_condition=2,
                               dim_input=6,
                               num_coupling_layers=3,
                               hidden_size=64,
                               device='cpu',
                               enable_wandb=False))

optimizers = []
test_pointclouds = []
for ind in range(1):
    optimizers.append(torch.optim.Adam(models[ind].model.parameters(), lr=1e-3,
                             betas=(0.8, 0.9), eps=1e-6, weight_decay=2e-5))
    test_pointclouds.append(supercells[ind])
    
test_radiation = "/home/willma32/insitu_particles/InSituML/main/ModelHelpers/cINN/rad_test.npy"
log_plots = visualizations.log_one_plot

'''
for ind in range(1):
    print('Train: ', supercells[ind].split('/')[-1].split('.')[0])
    models[0].device = 'cpu'
    models[0].to('cpu')
    s = time.time()
    models[0].train_(datasets[ind],
                   datasets[ind],
                   optimizers[ind],
                   epochs=1001,
                   batch_size=10000,
                   test_epoch=1000)
    e = time.time()
    print('time on cpu, cINN training: ', e-s, 's')
    break
'''
import time  

for ind in range(1):
    print(supercells[0])
    print('Train: ', supercells[ind].split('/')[-1].split('.')[0])
    models[0].device = 'cuda'
    models[0].to('cuda')
    s = time.time()
    models[0].train_(datasets[ind],
                   datasets[ind],
                   optimizers[ind],
                   epochs=1,
                   batch_size=10000,
                   test_epoch=1000)
    e = time.time()
    print(e-s, 's')
    
model_parameters = filter(lambda p: p.requires_grad, models[0].model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
