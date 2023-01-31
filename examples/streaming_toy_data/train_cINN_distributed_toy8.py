from time import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import os

from toy8 import generate

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    

def train_NLL(model, trainable_parameters, optimizer, train_loader, i_epoch=0, n_its_per_epoch = 8):
    model.train()

    l_tot = 0
    batch_idx = 0
    
    t_start = time()
    

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        #x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward step:
        z, log_j = model(x, c=y)
        nll = torch.mean(z**2) / 2 - torch.mean(log_j)/2

        # clip gradients
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.)
        l_tot += nll.data.item()
        nll.backward()

        #for p in model.parameters():
        #    if(not p.grad is None):
        #        p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()

    return model


def build_model(ndim_x = 2, ndim_cond = 8, no_coupling_blocks = 2, nh = 512, nl = 1):
    
    def subnet_fc(c_in, c_out):
        """
        Initialize subnetwork
        """
        layers=[nn.Linear(c_in, nh), nn.LeakyReLU()]

        for i in range(nl):
            layers.append(nn.Linear(nh, nh))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(nh,  c_out))
        mlp = nn.Sequential(*layers)

        for lin_layer in mlp:
            if(isinstance(lin_layer, nn.Linear)):
                nn.init.constant_(lin_layer.bias, 0)
                nn.init.xavier_uniform_(lin_layer.weight)

        nn.init.constant_(mlp[-1].bias, 0)
        nn.init.constant_(mlp[-1].weight, 0)
        return mlp

    nodes = [InputNode(ndim_x, name='input')]
    cond = Ff.ConditionNode(ndim_cond, name='condition') # color in one-hot encoding

    for k in range(no_coupling_blocks):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':2.0},
                          name=F'coupling_{k}', conditions=cond))
        nodes.append(Node(nodes[-1],
                          PermuteRandom,
                          {'seed':k},
                          name=F'permute_{k}'))

    model = Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
    
    return model


def main(rank, world_size):
    torch.cuda.set_device(rank)
    num_workers = 0 # constant
    
    setup(rank, world_size)
    
    pos, labels = generate(labels='all', tot_dataset_size=2**20) 
    pos = pos.to(rank)
    labels = labels.to(rank)
    batch_size = int(labels.sum(axis=0)[0])
    batch_size
    
    ####
    # prepare data
    ####
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos, labels),
        batch_size=batch_size, shuffle=True, drop_last=True)

    dataset = torch.utils.data.TensorDataset(pos, labels)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=False, num_workers=num_workers, sampler=sampler)
    
    ####
    # training hyperparameters
    ####
    ndim_x = 2
    ndim_cond = 8
    no_coupling_blocks = 2 
    nh = 512  #no hidden neurons
    nl = 1 #no hidden layers
    
    # create model and move it to GPU with id rank
    model = build_model().to(rank)
    print("EHLO from rank ", rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    
    n_epochs = 50
    n_its_per_epoch = 8

    lr = 1e-3

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9),
                                 eps=1e-6, weight_decay=l2_reg)
    
    ####
    # train cINN
    ####
    
    t_start = time()
    if(rank == 0):
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):
            train_NLL(model, trainable_parameters, optimizer, train_loader, i_epoch, n_its_per_epoch)
    else:
        for i_epoch in range(n_epochs):
            train_NLL(model, trainable_parameters, optimizer, train_loader, i_epoch, n_its_per_epoch)
        
    
    if(rank > 0):
        return

    print(f"\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")

    
    ####
    # evaluate model on master node
    ####
    N_samp = 40960

    x_samps = torch.cat([x for x,y in test_loader], dim=0)[:N_samp]
    y_samps = torch.cat([y for x,y in test_loader], dim=0)[:N_samp]
    c = np.where(y_samps.cpu())[1]
    #y_samps += y_noise_scale * torch.randn(N_samp, ndim_cond).to(rank)
    
    # sample posterior distribution 
    z = torch.randn(N_samp, ndim_x).to(rank)
    rev_x, _ = model(z, y_samps, rev=True)
    rev_x = rev_x.cpu().data.numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title('Predicted labels (Forwards Process)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('Generated Samples (Backwards Process)')
    fig.canvas.draw()
    
    axes[0].clear()
    axes[0].scatter(rev_x[:,0], rev_x[:,1], c=c, cmap='Set1', s=1., vmin=0, vmax=9)
    axes[0].axis('equal')
    axes[0].axis([-3,3,-3,3])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].clear()
    axes[1].scatter(rev_x[:,0], rev_x[:,1], c=c, cmap='Set1', s=1., vmin=0, vmax=9)
    axes[1].axis('equal')
    axes[1].axis([-3,3,-3,3])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.canvas.draw()
    fig.show()

    
    plt.savefig('foo.png', bbox_inches='tight')

    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(
        main,
        args=(world_size, ),
        nprocs=world_size
    )