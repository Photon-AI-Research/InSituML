#train_AE_khi_box_v2-2-1-8-rad-ex-random-v4-chamfers-Copy1.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from geomloss import SamplesLoss 
from utilities import *
# from networks import *

from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
from train_khi_AE_refactored.networks import ConvAutoencoder, VAE

# MAPPING_TO_NETWORK = {
#     "convAE": ConvAutoencoder,
#     "convAE1": ConvAutoencoder1,
#     "convAE2": ConvAutoencoder2,
#     "convAE3": ConvAutoencoder3,
#     "convAE4": ConvAutoencoder4,
#     "convAE5": ConvAutoencoder5,
#     "convAE6": ConvAutoencoder6,
#     "convAE7": ConvAutoencoder7,
#     "convAE8": ConvAutoencoder8,
#     "convAE9": ConvAutoencoder9,
#     "convAE10": ConvAutoencoder10,
#     "convAE11": ConvAutoencoder11,
#     "convAE12": ConvAutoencoder12,
#     "convAE16": ConvAutoencoder16,
#     "convAE17": ConvAutoencoder17,
#     "convAE18": ConvAutoencoder18,
#     "convAE19": ConvAutoencoder19,
#     "convAE20": ConvAutoencoder20,
#     "convAE21": ConvAutoencoder21,
#     "convAE22": ConvAutoencoder22,
#     "convAE23": ConvAutoencoder23,
#     "convAE24": ConvAutoencoder24,
#     "convAE25": ConvAutoencoder25,
#     }

 
def save_checkpoint(model, optimizer, path, last_loss, min_valid_loss, epoch, wandb_run_id):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_loss': last_loss.item(),
            'epoch': epoch,
            'min_valid_loss': min_valid_loss,
            'wandb_run_id': wandb_run_id,
        }

        torch.save(state, path + '/model_' + str(epoch))

def chamfersDist(a, b):
    d = torch.cdist(a, b, p=2)
    return torch.sum(torch.min(d, -1).values) + torch.sum(torch.min(d, -2).values)

def chamfersDist(a, b):
    # Ensure that a and b have the same batch size and the same number of dimensions
    assert a.size(0) == b.size(0) and a.size(2) == b.size(2), "Incompatible batch sizes or dimensions"

    # Compute pairwise distances for each point in each set within each batch
    # The resulting tensor 'd' has shape [batch_size, N_a, N_b]
    d = torch.cdist(a, b, p=2)

    # Calculate the Chamfer's distance for each batch
    # We take the minimum across the second dimension (N_a) and sum over the last dimension (N_b),
    # and vice versa, then sum these two results
    min_dist_a_to_b = torch.min(d, dim=2)[0].sum(dim=1)
    min_dist_b_to_a = torch.min(d, dim=1)[0].sum(dim=1)

    # Summing the distances for each batch to get a single value per batch
    chamfers_distance_per_batch = min_dist_a_to_b + min_dist_b_to_a

    return chamfers_distance_per_batch


# def filter_dims(phase_space, property_="positions"):
    
#     if property_ == "positions":
#         return phase_space[:,:,:3]
#     elif property_ == "momentum":
#         return phase_space[:,:,3:6]
#     elif property_ == "force":
#         return phase_space[:,:,6:]
#     else:
#         return phase_space
    
def filter_dims(phase_space, property_="positions"):
    
    if property_ == "positions":
        return phase_space[:,:,:3]
    elif property_ == "momentum":
        return phase_space[:,:,3:6]
    elif property_ == "force":
        return phase_space[:,:,6:]
    elif property_ == "momentum_force":
        return phase_space[:,:,3:]
    else:
        return phase_space
    
    
def create_momentum_density_plots(px, py, pz,
                                  px_pr, py_pr, pz_pr, chamfers_loss=None,emd_loss=None,
                                  bins=100, t=1000, gpu_box =0, path='',
                                  enable_wandb = False):
    
    # Specify the number of bins for each axis
    bins_px = np.linspace(min(px), max(px), bins)
    bins_py = np.linspace(min(py), max(py), bins)
    bins_pz = np.linspace(min(pz), max(pz), bins)
    
    loss_info = ''
    if chamfers_loss is not None:
        loss_info += '\nChamfers: {:.4f}'.format(chamfers_loss)
    if emd_loss is not None:
        loss_info += '\nEMD: {:.4f}'.format(emd_loss)
    
    # Create subplots for each plane
    plt.figure(figsize=(15, 10)) 
    
    # px-py Plane Ground Truth
    plt.subplot(231)
    plt.hist2d(px, py, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('py')
    plt.title('px-py Plane GT at t = {}, box = {}'.format(t,gpu_box))
    
    # px-pz Plane Ground Truth
    plt.subplot(232)
    plt.hist2d(px, pz, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('pz')
    plt.title('px-pz Plane GT at t = {}, box = {}'.format(t,gpu_box))
    
    # py-pz Plane Ground Truth
    plt.subplot(233)
    plt.hist2d(py, pz, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py')
    plt.ylabel('pz')
    plt.title('py-pz Plane GT at t = {}, box = {}'.format(t,gpu_box))
    
    # px-py Plane Prediction
    plt.subplot(234)
    plt.hist2d(px_pr, py_pr, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('py_pr')
    plt.title('px-py Plane Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # px-pz Plane Prediction
    plt.subplot(235)
    plt.hist2d(px_pr, pz_pr, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('pz_pr')
    plt.title('px-pz Plane Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # py-pz Plane Prediction
    plt.subplot(236)
    plt.hist2d(py_pr, pz_pr, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py_pr')
    plt.ylabel('pz_pr')
    plt.title('py-pz Plane Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    plt.tight_layout()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/momentum_density_plots_{}_{}.png'.format(t,gpu_box))
    
    if enable_wandb == True:
        # Log the overlapping histogram plot
        wandb.log({"Px vs Py vs Pz histograms (t={},box={})".format(t,gpu_box): wandb.Image(plt)})

        plt.close()
    else:
        plt.show()   


def create_force_density_plots(fx, fy, fz,
                               fx_pr, fy_pr, fz_pr, chamfers_loss=None,emd_loss=None,
                               bins=100, t=1000, gpu_box =0, path='',
                               enable_wandb = False):
    
    # Specify the number of bins for each axis
    bins_fx = np.linspace(min(fx), max(fx), bins)
    bins_fy = np.linspace(min(fy), max(fy), bins)
    bins_fz = np.linspace(min(fz), max(fz), bins)
    
    loss_info = ''
    if chamfers_loss is not None:
        loss_info += '\nChamfers: {:.4f}'.format(chamfers_loss)
    if emd_loss is not None:
        loss_info += '\nEMD: {:.4f}'.format(emd_loss)
        
    # Create subplots for each plane
    plt.figure(figsize=(15, 10))  # Adjust the figure size
    
    # fx-fy Plane Ground Truth
    plt.subplot(231)
    plt.hist2d(fx, fy, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fy')
    plt.title('fx-fy Plane GT at t = {}, box = {}'.format(t,gpu_box))
    
    # fx-fz Plane Ground Truth
    plt.subplot(232)
    plt.hist2d(fx, fz, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fz')
    plt.title('fx-fz Plane GT at t = {}, box = {}'.format(t,gpu_box))
    
    # fy-fz Plane Ground Truth
    plt.subplot(233)
    plt.hist2d(fy, fz, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy')
    plt.ylabel('fz')
    plt.title('fy-fz Plane GT at t = {}, box = {}'.format(t,gpu_box))
    
    # fx-fy Plane Prediction
    plt.subplot(234)
    plt.hist2d(fx_pr, fy_pr, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fy_pr')
    plt.title('fx-fy Plane Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fx-fz Plane Prediction
    plt.subplot(235)
    plt.hist2d(fx_pr, fz_pr, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fz_pr')
    plt.title('fx-fz Plane Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fy-fz Plane Prediction
    plt.subplot(236)
    plt.hist2d(fy_pr, fz_pr, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy_pr')
    plt.ylabel('fz_pr')
    plt.title('fy-fz Plane Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    plt.tight_layout()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/force_density_plots_{}_{}.png'.format(t,gpu_box))
    
    if enable_wandb == True:
        # Log the overlapping histogram plot
        wandb.log({"Fx vs Fy vs FZ histograms (t={},box={})".format(t,gpu_box): wandb.Image(plt)})

        plt.close()
    else:
        plt.show()  
        

# def validate_model(model, validation_dataset, device, config, emd_loss, create_position_density_plots, create_momentum_density_plots, create_force_density_plots):
#     model.eval()
#     val_loss_avg = []
#     with torch.no_grad():
#         for idx in range(len(validation_dataset)):
#             timestep_index, validation_boxes, phase_space, _ = validation_dataset[idx]
#             phase_space = filter_dims(phase_space, property_=config["property_"])
#             phase_space = phase_space.permute(0, 2, 1).to(device)
#             p_pr = model(phase_space)
#             print('p_pr', p_pr.shape)
#             val_loss = emd_loss(p_pr.transpose(2, 1).contiguous(), phase_space.transpose(2, 1).contiguous())
#             cd = chamfersDist(p_pr.transpose(2, 1), phase_space.transpose(2, 1))
#             val_loss_avg.append(val_loss[0].mean().item())
#             #config["t0"]
#             if timestep_index == 1990:
#                 for i, b_i in enumerate(validation_boxes):
#                     p_gt = p[i].transpose(0, 1).detach().cpu().numpy()
#                     pc_pr = p_pr[i].transpose(0, 1).detach().cpu().numpy()

#                     if config["property_"] == "positions":
#                         # Visualization code for positions
#                         x = p_gt[:, 0]  # X coordinates
#                         y = p_gt[:, 1]  # Y coordinates
#                         z = p_gt[:, 2]  # Z coordinates

#                         # # Assuming 'data' is your 10 million x 9 matrix
#                         x_pr = pc_pr[:, 0]  # X coordinates
#                         y_pr = pc_pr[:, 1]  # Y coordinates
#                         z_pr = pc_pr[:, 2]  # Z coordinates

#                         create_position_density_plots(x, y, z,
#                                                       x_pr, y_pr, z_pr,
#                                                       bins=100, t=timestep_index, gpu_box=b_i,
#                                                       chamfers_loss = cd[i].item(),
#                                                       emd_loss=val_loss[i].item(),
#                                                       enable_wandb = True
#                                                      )

#                     elif config["property_"] == "momentum":
#                         # Visualization code for momentum
#                         px = p_gt[:, 0]  # Px component of momentum
#                         py = p_gt[:, 1]  # Py component of momentum
#                         pz = p_gt[:, 2]  # Pz component of momentum

#                         px_pr = pc_pr[:, 0]  # Px component of momentum
#                         py_pr = pc_pr[:, 1]  # Py component of momentum
#                         pz_pr = pc_pr[:, 2]  # Pz component of momentum

#                         create_momentum_density_plots(px, py, pz,
#                                                       px_pr, py_pr, pz_pr,
#                                                       bins=100, t=timestep_index, gpu_box=b_i,
#                                                       chamfers_loss = cd[i].item(),
#                                                       emd_loss=val_loss[i].item(),
#                                                       enable_wandb = True
#                                                      )  

#                     else:
#                         # Visualization code for force
#                         fx = p_gt[:, 0]  # Fx component of force
#                         fy = p_gt[:, 1]  # Fy component of force
#                         fz = p_gt[:, 2]  # Fz component of force

#                         fx_pr = pc_pr[:, 0]  # Fx component of force
#                         fy_pr = pc_pr[:, 1]  # Fy component of force
#                         fz_pr = pc_pr[:, 2]  # Fz component of force

#                         create_force_density_plots(fx, fy, fz,
#                                                    fx_pr, fy_pr, fz_pr,
#                                                    bins=100, t=timestep_index, gpu_box=b_i,
#                                                    chamfers_loss = cd[i].item(),
#                                                    emd_loss=val_loss[i].item(),
#                                                    enable_wandb = True
#                                                   )

#     val_loss_overall_avg = sum(val_loss_avg) / len(val_loss_avg)
#     return val_loss_overall_avg


# def generate_validation(t_index,gpu_index,pathpattern1, config, device):
#     p_gt = np.load(pathpattern1.format(t_index),allow_pickle = True)
#     p_gt = p_gt[gpu_index]
#     p_gt = np.array(p_gt, dtype = np.float32)
#     p_gt = normalize_columns(p_gt)
#     p_gt = random_sample(p_gt, sample_size=config["particles_to_sample"])
#     p_gt = torch.from_numpy(p_gt).unsqueeze(0)
#     p_gt = filter_dims(p_gt, property_=config["property_"])
#     p_gt = p_gt.permute(0, 2, 1).to(device)
    
#     model.eval()
#     with torch.no_grad():
#         pc_pr = model(p_gt)
    
#     #print('pc_pr', pc_pr.shape)
#     emd = emd_loss(pc_pr.transpose(2, 1).contiguous(), p_gt.transpose(2, 1).contiguous())
#     #cd = chamfersDist(pc_pr.transpose(2, 1), p_gt.transpose(2, 1))
    
#     p_gt = p_gt.squeeze().transpose(1, 0).cpu().numpy()
#     pc_pr = pc_pr.squeeze().transpose(1, 0).cpu().numpy()
    
#     if config["property_"] == "positions":
#         # Visualization code for positions
#         x = p_gt[:, 0]  # X coordinates
#         y = p_gt[:, 1]  # Y coordinates
#         z = p_gt[:, 2]  # Z coordinates

#         # # Assuming 'data' is your 10 million x 9 matrix
#         x_pr = pc_pr[:, 0]  # X coordinates
#         y_pr = pc_pr[:, 1]  # Y coordinates
#         z_pr = pc_pr[:, 2]  # Z coordinates

#         create_position_density_plots(x, y, z,
#                                       x_pr, y_pr, z_pr,
#                                       bins=100, t=t_index, gpu_box=gpu_index,
#                                       #chamfers_loss = cd.item(),
#                                       emd_loss=emd.item(),
#                                       enable_wandb = True
#                                      )

#     elif config["property_"] == "momentum":
#         # Visualization code for momentum
#         px = p_gt[:, 0]  # Px component of momentum
#         py = p_gt[:, 1]  # Py component of momentum
#         pz = p_gt[:, 2]  # Pz component of momentum

#         px_pr = pc_pr[:, 0]  # Px component of momentum
#         py_pr = pc_pr[:, 1]  # Py component of momentum
#         pz_pr = pc_pr[:, 2]  # Pz component of momentum

#         create_momentum_density_plots(px, py, pz,
#                                       px_pr, py_pr, pz_pr,
#                                       bins=100, t=t_index, gpu_box=gpu_index,
#                                       #chamfers_loss = cd.item(),
#                                       emd_loss=emd.item(),
#                                       enable_wandb = True
#                                      )  
        
#     elif config["property_"] == "force":
#         # Visualization code for force
#         fx = p_gt[:, 0]  # Fx component of force
#         fy = p_gt[:, 1]  # Fy component of force
#         fz = p_gt[:, 2]  # Fz component of force

#         fx_pr = pc_pr[:, 0]  # Fx component of force
#         fy_pr = pc_pr[:, 1]  # Fy component of force
#         fz_pr = pc_pr[:, 2]  # Fz component of force

#         create_force_density_plots(fx, fy, fz,
#                                    fx_pr, fy_pr, fz_pr,
#                                    bins=100, t=t_index, gpu_box=gpu_index,
#                                    #chamfers_loss = cd.item(),
#                                    emd_loss=emd.item(),
#                                    enable_wandb = True
#                                   )

#     else:
#         # Visualization code for positions
#         x = p_gt[:, 0]  # X coordinates
#         y = p_gt[:, 1]  # Y coordinates
#         z = p_gt[:, 2]  # Z coordinates

#         # # Assuming 'data' is your 10 million x 9 matrix
#         x_pr = pc_pr[:, 0]  # X coordinates
#         y_pr = pc_pr[:, 1]  # Y coordinates
#         z_pr = pc_pr[:, 2]  # Z coordinates

#         # Visualization code for momentum
#         px = p_gt[:, 3]  # Px component of momentum
#         py = p_gt[:, 4]  # Py component of momentum
#         pz = p_gt[:, 5]  # Pz component of momentum

#         px_pr = pc_pr[:, 3]  # Px component of momentum
#         py_pr = pc_pr[:, 4]  # Py component of momentum
#         pz_pr = pc_pr[:, 5]  # Pz component of momentum

#         # Visualization code for force
#         fx = p_gt[:, 6]  # Fx component of force
#         fy = p_gt[:, 7]  # Fy component of force
#         fz = p_gt[:, 8]  # Fz component of force

#         fx_pr = pc_pr[:, 6]  # Fx component of force
#         fy_pr = pc_pr[:, 7]  # Fy component of force
#         fz_pr = pc_pr[:, 8]  # Fz component of force

#         create_position_density_plots(x, y, z,
#                               x_pr, y_pr, z_pr,
#                               bins=100, t=t_index,
#                               gpu_box=gpu_index,
#                               #chamfers_loss = cd.item(),
#                               emd_loss=emd.item(),
#                               enable_wandb = True
#                              )

#         create_momentum_density_plots(px, py, pz,
#                                       px_pr, py_pr, pz_pr,
#                                       bins=100, t=t_index,
#                                       gpu_box=gpu_index,
#                                       #chamfers_loss = cd.item(),
#                                       emd_loss=emd.item(),
#                                       enable_wandb = True
#                                      )

#         create_force_density_plots(fx, fy, fz,
#                                    fx_pr, fy_pr, fz_pr,
#                                    bins=100, t=t_index,
#                                    gpu_box=gpu_index,
#                                    #chamfers_loss = cd.item(),
#                                    emd_loss=emd.item(),
#                                    enable_wandb = True
#                                   )


def generate_validation(t_index,gpu_index,pathpattern1, config,normalizer, device):
    p_gt = np.load(pathpattern1.format(t_index),allow_pickle = True)
    p_gt = p_gt[gpu_index]
    p_gt = np.array(p_gt, dtype = np.float32)
    #p_gt = normalize_columns(p_gt,l.global_mean, l.global_std)
    #p_gt = normalize_columns(p_gt)
    p_gt = normalizer.normalize_data(p_gt, method=config["norm_method"])
    p_gt = random_sample(p_gt, sample_size=config["particles_to_sample"])
    p_gt = torch.from_numpy(p_gt).unsqueeze(0)
    p_gt = filter_dims(p_gt, property_=config["property_"])
    p_gt = p_gt.to(device)
    # p_gt = p_gt.permute(0, 2, 1).to(device)
    
    model.eval()
    with torch.no_grad():
        pc_pr = model.reconstruct_input(p_gt)
        # ,_,_
    # print('pc_pr', pc_pr.shape)
    # print('p_gt', p_gt.shape)
    emd = emd_loss(pc_pr.contiguous(), p_gt.contiguous())
    #cd = chamfersDist(pc_pr.transpose(2, 1), p_gt.transpose(2, 1))
    
    p_gt = p_gt.squeeze().cpu().numpy()
    pc_pr = pc_pr.squeeze().cpu().numpy()
    
    if config["property_"] == "positions":
        # Visualization code for positions
        x = p_gt[:, 0]  # X coordinates
        y = p_gt[:, 1]  # Y coordinates
        z = p_gt[:, 2]  # Z coordinates

        # # Assuming 'data' is your 10 million x 9 matrix
        x_pr = pc_pr[:, 0]  # X coordinates
        y_pr = pc_pr[:, 1]  # Y coordinates
        z_pr = pc_pr[:, 2]  # Z coordinates

        create_position_density_plots(x, y, z,
                                      x_pr, y_pr, z_pr,
                                      bins=100, t=t_index, gpu_box=gpu_index,
                                      #chamfers_loss = cd.item(),
                                      emd_loss=emd.item(),
                                      enable_wandb = True
                                     )

    elif config["property_"] == "momentum":
        # Visualization code for momentum
        px = p_gt[:, 0]  # Px component of momentum
        py = p_gt[:, 1]  # Py component of momentum
        pz = p_gt[:, 2]  # Pz component of momentum

        px_pr = pc_pr[:, 0]  # Px component of momentum
        py_pr = pc_pr[:, 1]  # Py component of momentum
        pz_pr = pc_pr[:, 2]  # Pz component of momentum

        create_momentum_density_plots(px, py, pz,
                                      px_pr, py_pr, pz_pr,
                                      bins=100, t=t_index, gpu_box=gpu_index,
                                      #chamfers_loss = cd.item(),
                                      emd_loss=emd.item(),
                                      enable_wandb = True
                                     )  
        
    elif config["property_"] == "force":
        # Visualization code for force
        fx = p_gt[:, 0]  # Fx component of force
        fy = p_gt[:, 1]  # Fy component of force
        fz = p_gt[:, 2]  # Fz component of force

        fx_pr = pc_pr[:, 0]  # Fx component of force
        fy_pr = pc_pr[:, 1]  # Fy component of force
        fz_pr = pc_pr[:, 2]  # Fz component of force

        create_force_density_plots(fx, fy, fz,
                                   fx_pr, fy_pr, fz_pr,
                                   bins=100, t=t_index, gpu_box=gpu_index,
                                   #chamfers_loss = cd.item(),
                                   emd_loss=emd.item(),
                                   enable_wandb = True
                                  )
    
    elif config["property_"] == "momentum_force":
        # Visualization code for momentum
        px = p_gt[:, 0]  # Px component of momentum
        py = p_gt[:, 1]  # Py component of momentum
        pz = p_gt[:, 2]  # Pz component of momentum

        px_pr = pc_pr[:, 0]  # Px component of momentum
        py_pr = pc_pr[:, 1]  # Py component of momentum
        pz_pr = pc_pr[:, 2]  # Pz component of momentum
        
        # Visualization code for force
        fx = p_gt[:, 3]  # Fx component of force
        fy = p_gt[:, 4]  # Fy component of force
        fz = p_gt[:, 5]  # Fz component of force

        fx_pr = pc_pr[:, 3]  # Fx component of force
        fy_pr = pc_pr[:, 4]  # Fy component of force
        fz_pr = pc_pr[:, 5]  # Fz component of force
        
        create_momentum_density_plots(px, py, pz,
                              px_pr, py_pr, pz_pr,
                              bins=100, t=t_index, gpu_box=gpu_index,
                              #chamfers_loss = cd.item(),
                              emd_loss=emd.item(),
                              enable_wandb = True
                             ) 
        
        create_force_density_plots(fx, fy, fz,
                           fx_pr, fy_pr, fz_pr,
                           bins=100, t=t_index, gpu_box=gpu_index,
                           #chamfers_loss = cd.item(),
                           emd_loss=emd.item(),
                           enable_wandb = True
                          )
        

    else:
        # Visualization code for positions
        x = p_gt[:, 0]  # X coordinates
        y = p_gt[:, 1]  # Y coordinates
        z = p_gt[:, 2]  # Z coordinates

        # # Assuming 'data' is your 10 million x 9 matrix
        x_pr = pc_pr[:, 0]  # X coordinates
        y_pr = pc_pr[:, 1]  # Y coordinates
        z_pr = pc_pr[:, 2]  # Z coordinates

        # Visualization code for momentum
        px = p_gt[:, 3]  # Px component of momentum
        py = p_gt[:, 4]  # Py component of momentum
        pz = p_gt[:, 5]  # Pz component of momentum

        px_pr = pc_pr[:, 3]  # Px component of momentum
        py_pr = pc_pr[:, 4]  # Py component of momentum
        pz_pr = pc_pr[:, 5]  # Pz component of momentum

        # Visualization code for force
        fx = p_gt[:, 6]  # Fx component of force
        fy = p_gt[:, 7]  # Fy component of force
        fz = p_gt[:, 8]  # Fz component of force

        fx_pr = pc_pr[:, 6]  # Fx component of force
        fy_pr = pc_pr[:, 7]  # Fy component of force
        fz_pr = pc_pr[:, 8]  # Fz component of force

        create_position_density_plots(x, y, z,
                              x_pr, y_pr, z_pr,
                              bins=100, t=t_index,
                              gpu_box=gpu_index,
                              #chamfers_loss = cd.item(),
                              emd_loss=emd.item(),
                              enable_wandb = True
                             )

        create_momentum_density_plots(px, py, pz,
                                      px_pr, py_pr, pz_pr,
                                      bins=100, t=t_index,
                                      gpu_box=gpu_index,
                                      #chamfers_loss = cd.item(),
                                      emd_loss=emd.item(),
                                      enable_wandb = True
                                     )

        create_force_density_plots(fx, fy, fz,
                                   fx_pr, fy_pr, fz_pr,
                                   bins=100, t=t_index,
                                   gpu_box=gpu_index,
                                   #chamfers_loss = cd.item(),
                                   emd_loss=emd.item(),
                                   enable_wandb = True
                                  )
        
def validate_model(model, valid_data_loader, property_, device):
    model.eval()
    val_loss_avg = []
    
    with torch.no_grad():
        for idx in range(len(valid_data_loader)):
            timestep_index, validation_boxes, p, _ = valid_data_loader[idx]
            p = filter_dims(p, property_)
            # p = p.permute(0, 2, 1).to(device)
            p = p.to(device)
            #print('p valid', p.shape)
            val_loss,x_reconst,kl_loss,p_pr,_ = model(p)
            
            # print('p_pr', p_pr.shape)
            # print('p', p.shape)
            
            wandb.config.update({'output points': p_pr.shape[2]})
            # print('p',p.shape)
            # print('p_pr',p_pr.shape)
            # print('idx', idx)
            # val_loss = emd_loss(p_pr.transpose(2, 1).contiguous(), p.transpose(2, 1).contiguous())
            
            val_loss_avg.append(val_loss.mean().item())
    val_loss_overall_avg = sum(val_loss_avg) / len(val_loss_avg)
    return val_loss_overall_avg

class TrainLoader:
    def __init__(self, 
                 normalisation,
                 pathpattern1="/bigdata/hplsim/aipp/Jeyhun/khi/particle_box/40_80_80_160_0_2/{}.npy",
                 pathpattern2="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex/{}.npy",
                 t0=0,
                 t1=100,
                 timebatchsize=20,
                 particlebatchsize=10240,
                 particles_to_sample=150000,
                 blacklist_box=None,
                 data_stats_path ='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_002/global_stats.npz'):
        self.normalisation = normalisation
        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        self.blacklist_box = blacklist_box
        
        # TODO check if all files are there
        
        self.t0 = t0
        self.t1 = t1
        
        self.timebatchsize = timebatchsize
        self.particlebatchsize = particlebatchsize
        self.particles_to_sample = particles_to_sample

        num_files = t1 - t0
        missing_files = [i for i in range(t0, t1) if not os.path.exists(pathpattern1.format(i))]
        num_missing = len(missing_files)
        all_files_exist = num_missing == 0

        if all_files_exist:
            print("All {} files from {} to {} exist in the directory.".format(num_files, t0, t1))
        else:
            print("{} files are missing out of {} in the directory.".format(num_missing, num_files))

    def __len__(self):
        return self.t1 - self.t0
        
    def __getitem__(self, idx):
        
        class Epoch:
            def __init__(self,
                         loader,
                         t0,
                         t1,
                         timebatchsize=20,
                         particlebatchsize=10240,
                         blacklist_box=None):
                self.perm = torch.randperm(len(loader))
                self.loader = loader
                self.t0 = t0
                self.t1 = t1
                self.timebatchsize = timebatchsize
                self.particlebatchsize = particlebatchsize
                self.blacklist_box = blacklist_box

            def __len__(self):
                return len(self.loader) // self.timebatchsize
        
            def __getitem__(self, timebatch):
                i = self.timebatchsize*timebatch
                bi = self.perm[i:i+self.timebatchsize]
                radiation = []
                particles = []
                for time in bi:
                    index = time + self.t0
                    
                    # Load particle data
                    p = np.load(self.loader.pathpattern1.format(index), allow_pickle = True)
                    # Load radiation data
                    r = torch.from_numpy(np.load(self.loader.pathpattern2.format(index)).astype(np.cfloat))
                    
                    # Exclude the blacklisted box if specified
                    if self.blacklist_box is not None:
                        p = np.delete(p, self.blacklist_box, axis=0)
                        r = np.delete(r, self.blacklist_box, axis=0)
                    
                    p = [self.loader.normalisation.normalize_data(element, method=config["norm_method"], timestep_index = index) for element in p]
                    p = np.array(p, dtype=object)
                    
                    # random sample N points from each box
                    p = [random_sample(element, sample_size=self.loader.particles_to_sample) for element in p]
                    p = torch.from_numpy(np.array(p, dtype = np.float32))
                    
                    # choose relevant directions
                    r = r[:, 1:, :]
                    r = r.view(r.shape[0], -1)
                    
                    # Compute the phase (angle) of the complex number in radians
                    phase = torch.angle(r)
                    
                    # Compute the amplitude (magnitude) of the complex number
                    amplitude = torch.abs(r)
                    r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

                    particles.append(p)
                    radiation.append(r)
                
                # concatenate particle and radiation data across randomly chosen timesteps
                particles = torch.cat(particles)
                radiation = torch.cat(radiation)
                
                class Timebatch:
                    def __init__(self, particles, radiation, batchsize):
                        self.batchsize = batchsize
                        self.particles = particles
                        self.radiation = radiation
                        
                        self.perm = torch.randperm(self.radiation.shape[0])
                        
                    def __len__(self):
                        return self.radiation.shape[0] // self.batchsize

                    def __getitem__(self, batch):
                        i = self.batchsize*batch
                        bi = self.perm[i:i+self.batchsize]
                    
                        return self.particles[bi], self.radiation[bi]
                
                return Timebatch(particles, radiation, self.particlebatchsize)
                    
        return Epoch(self, self.t0, self.t1, self.timebatchsize, self.particlebatchsize, self.blacklist_box)

# # ValidationDatast old
# class ValidationDataset:
#     def __init__(self, 
#                  pathpattern1,
#                  pathpattern2,
#                  validation_boxes,
#                  t0=0,
#                  t1=100):
#         self.pathpattern1 = pathpattern1
#         self.pathpattern2 = pathpattern2
#         self.validation_boxes = validation_boxes
#         self.t0 = t0
#         self.t1 = t1

#     def __len__(self):
#         return self.t1 - self.t0

#     def __getitem__(self, idx):

#         timestep_index = self.t0 + idx

#         # Load particle data for multiple validation boxes
#         p_loaded = np.load(self.pathpattern1.format(timestep_index), allow_pickle=True)
#         p = [p_loaded[box_index] for box_index in self.validation_boxes]
        
#         p = [normalize_columns(element) for element in p]
#         p = np.array(p, dtype=object)
        
#         p = [random_sample(element, sample_size=150000) for element in p]
#         p = torch.from_numpy(np.array(p, dtype = np.float32))

#         # Load radiation data for multiple validation boxes
#         r_loaded = torch.from_numpy(np.load(self.pathpattern2.format(timestep_index)).astype(np.cfloat))
#         r =  r_loaded[self.validation_boxes, 1:, :]
#         r = r.view(r.shape[0], -1)
        
#         # Compute the phase (angle) of the complex number in radians
#         phase = torch.angle(r)

#         # Compute the amplitude (magnitude) of the complex number
#         amplitude = torch.abs(r)
#         r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

#         return timestep_index, self.validation_boxes, p, r


class ValidationFixedBoxLoader:
    def __init__(self, 
                 pathpattern1,
                 pathpattern2,
                 validation_boxes,
                 normalisation,
                 t0=0,
                 t1=100,
                 particles_to_sample=150000,
                 select_timesteps=1,
                 data_stats_path ='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_002/global_stats.npz'
          ):
        self.normalisation = normalisation
        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        self.validation_boxes = validation_boxes
        self.t0 = t0
        self.t1 = t1
        self.particles_to_sample = particles_to_sample
        self.select_timesteps = select_timesteps
        
        # Extract global mean and standard deviation
#         global_mean_momentum = data_stats_path['mean_momentum']
#         global_std_momentum = data_stats_path['std_momentum']
#         global_mean_force = data_stats_path['mean_force']
#         global_std_force = data_stats_path['std_force']

#         # Combine the mean and standard deviation for momentum and force components
#         self.global_mean = np.concatenate((global_mean_momentum, global_mean_force))
#         self.global_std = np.concatenate((global_std_momentum, global_std_force))
        
    def __len__(self):
        self.perm =  torch.randperm((self.t1-self.t0))[:self.select_timesteps]
        return self.select_timesteps

    def __getitem__(self, idx):

        timestep_index = self.t0 + self.perm[idx]
     
        # Load particle data for the validation boxes
        p_loaded = np.load(self.pathpattern1.format(timestep_index), allow_pickle=True)
        p = [p_loaded[box_index] for box_index in self.validation_boxes]
        
        #p = [normalize_columns(element) for element in p]
        p = [self.normalisation.normalize_data(element, method=config["norm_method"], timestep_index = timestep_index) for element in p]
        p = np.array(p, dtype=object)
                
        p = [random_sample(element, 
                           sample_size=self.particles_to_sample) for element in p]
        
        p = torch.from_numpy(np.array(p, dtype = np.float32))        

        # Load radiation data for the validation boxes
        r_loaded = torch.from_numpy(np.load(self.pathpattern2.format(timestep_index)).astype(np.cfloat))
        r =  r_loaded[self.validation_boxes, 1:, :]
        r = r.view(r.shape[0], -1)
        
        # Compute the phase (angle) of the complex number in radians
        phase = torch.angle(r)

        # Compute the amplitude (magnitude) of the complex number
        amplitude = torch.abs(r)
        r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

        return timestep_index, self.validation_boxes, p, r

if __name__ == "__main__":

    hyperparameter_defaults = dict(
    t0 = 1900,
    t1 = 2001,
    timebatchsize = 4,
    particlebatchsize = 2,
    hidden_size = 1024,
    latent_space_dims = 544,
    dim_pool = 1,
    lr = 0.001,
    num_epochs = 20000,
    blacklist_boxes = None,
    val_boxes = [3],
    activation = 'relu',
    property_ = 'momentum_force',
    network = 'VAE',
    norm_method = 'mean_6d',
    particles_to_sample = 150000,
    weight_kl = 0.001,
    encoder_kwargs = {"ae_config":"non_deterministic",
                      "z_dim":544,
                      "input_dim":6,
                      "conv_layer_config":[16, 32, 64, 128, 256, 640],
                      "conv_add_bn": False, 
                      "fc_layer_config":[544]},
    decoder_kwargs = {"z_dim":544,
                      "input_dim":6,
                      "initial_conv3d_size":[16, 4, 4, 4],
                      "add_batch_normalisation":False,
                      "fc_layer_config":[1024]
                      },
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy"
    )
    
    
    included_hparams = ['network','encoder_kwargs', 'decoder_kwargs']

    # Generate a name for the run
    def generate_run_name(hparams, included_keys):
        name_parts = [f"{key}={hparams[key]}" for key in included_keys if key in hparams]
        return ",".join(name_parts)

    run_name = generate_run_name(hyperparameter_defaults, included_hparams)
    
    print('New session...')
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, project="khi_public_vae", name=run_name)

    start_epoch = 0
    min_valid_loss = np.inf
    
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    
#     tags_to_add = (config["encoder_kwargs"], config["decoder_kwargs"])
#     wandb.run.tags += tags_to_add
#     wandb.run.save()
    
    # tags_to_add = ("network_" + config.network, "property_" + config.property_)
    # wandb.run.tags += tags_to_add
    # wandb.run.save()
    
    pathpattern1 = config["pathpattern1"]
    pathpattern2 = config["pathpattern2"]
    
    # point_dim = 9 if config["property_"] == "all" else 3
    if config["property_"] == "all":
        point_dim = 9
    elif config["property_"] == "momentum_force":
        point_dim = 6
    else:
        point_dim = 3
    
    # Create an instance of the Normalizer class
    normalizer = Normalizer()
    
    l = TrainLoader(pathpattern1=pathpattern1,
                    pathpattern2=pathpattern2,
                    t0=config["t0"],
                    t1=config["t1"],
                    timebatchsize=config["timebatchsize"],
                    particlebatchsize=config["particlebatchsize"],
                    blacklist_box = config["blacklist_boxes"],
                    particles_to_sample = config["particles_to_sample"],
                    normalisation = normalizer)
    
    valid_data_loader = ValidationFixedBoxLoader(pathpattern1,
                                           pathpattern2,
                                           config["val_boxes"],
                                           t0=config["t0"],
                                           t1=config["t1"],
                                           particles_to_sample= config["particles_to_sample"],
                                                 normalisation = normalizer)
    
#     class Reshape(nn.Module):
#         def __init__(self, *args):
#             super().__init__()
#             self.shape = args

#         def forward(self, x):
#             return x.view(self.shape)
        
#     # Define the convolutional autoencoder class
#     class ConvAutoencoder(nn.Module):
#         def __init__(self):
#             super(ConvAutoencoder, self).__init__()

#             # Encoder
#             self.encoder = nn.Sequential(
#                 nn.Conv1d(point_dim, 16, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(16, 32, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(32, 64, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(64, 128, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(128, 256, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(256, 512, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(512, config["hidden_size"], kernel_size=1),
#                 nn.AdaptiveMaxPool1d(config["dim_pool"]), 
#                 nn.Flatten()
#             )

#             # Decoder
#             self.decoder = nn.Sequential(
#                 nn.Unflatten(1, (16,4,4,4)),
#                 nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2),
#                 nn.ReLU(),
#                 nn.ConvTranspose3d(8, point_dim, kernel_size=2, stride=2),
#                 nn.Flatten(2),
#             )

#         def forward(self, x):
#             x = self.encoder(x)
#             x = self.decoder(x)
#             return x
    
    # Initialize the convolutional autoencoder
    # model = ConvAutoencoder()
    
    # Initialize the convolutional autoencoder
    #model = MAPPING_TO_NETWORK[config["network"]](**config)
    # model = MAPPING_TO_NETWORK[config["network"]](config["hidden_size"],config["dim_pool"], point_dim)
    
    
#     encoder_kwargs = {"ae_config":"non_deterministic",
#                       "z_dim":config["latent_space_dims"],
#                       "input_dim":point_dim,
#                       "conv_layer_config":[16, 32, 64, 128, 256, 512],
#                       "conv_add_bn": False, 
#                       "fc_layer_config":[256]}

#     decoder_kwargs = {"z_dim":config["latent_space_dims"],
#                       "input_dim":point_dim,
#                       "initial_conv3d_size":[16, 4, 4, 4],
#                       "add_batch_normalisation":False}

    encoder_kwargs = config["encoder_kwargs"]
    decoder_kwargs = config["decoder_kwargs"]
    emd_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, verbose=True)

        
    model = VAE(encoder = Encoder, 
              encoder_kwargs = encoder_kwargs, 
              decoder = Conv3DDecoder, 
              z_dim=config["latent_space_dims"],
              decoder_kwargs = decoder_kwargs,
              loss_function = emd_loss,
              property_="momentum_force",
              particles_to_sample = config["particles_to_sample"],
              ae_config="non_deterministic",
              use_encoding_in_decoder=False,
              weight_kl = config["weight_kl"])
    
    
    print('model', model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    wandb.config.update({'total_params': total_params})
    


    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    
    directory ='/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/'+str(wandb.run.id)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    
    
    epoch = l[0]
    
    patience = 20
    slow_improvement_patience = 10
    no_improvement_count = 0
    slow_improvement_count = 0
    batch_tot_idx = 0

    start_time = time.time()
    for i_epoch in range(start_epoch, config["num_epochs"]):   
        model.train()
        loss_overall = []
        for timeBatchIndex in range(len(epoch)):
            loss_avg = []
            timebatch = epoch[timeBatchIndex]
            
            batch_idx = 0
            start_timebatch = time.time()
            for particleBatchIndex in range(len(timebatch)):
                batch_idx += 1
                batch_tot_idx +=1
                #print('batch_tot_idx', batch_tot_idx)
                optimizer.zero_grad()
                phase_space, _ = timebatch[particleBatchIndex]
                
                phase_space = filter_dims(phase_space, property_= config["property_"])
                
                #phase_space = phase_space.permute(0, 2, 1).to(device)
                phase_space = phase_space.to(device)
                
                loss,x_reconst,kl_loss,_,_ = model(phase_space)
                # loss = chamfersDist(output.transpose(2,1), phase_space.transpose(2,1))
                
                if i_epoch == 0 and particleBatchIndex == 0:
                    print('phase_space', phase_space.shape)
                    # print('output', output.shape)
                    
                # loss = emd_loss(output.transpose(2,1).contiguous(), phase_space.transpose(2,1).contiguous())

                loss = loss.mean()
                loss_avg.append(loss.item())
                
                if (batch_tot_idx)%500 == 0:
                    # Call the validation function
                    #val_loss_overall_avg = validate_model(model, validation_dataset, device, config, emd_loss, create_position_density_plots, create_momentum_density_plots, create_force_density_plots)
                    t_index = config["t1"] - 1
                    gpu_index = 3,19,12,17
                    generate_validation(t_index, gpu_index[0], pathpattern1, config,normalizer, device)
                    generate_validation(t_index, gpu_index[1], pathpattern1, config,normalizer, device)
                    generate_validation(t_index, gpu_index[2], pathpattern1, config,normalizer, device)
                    val_loss_overall_avg = validate_model(model, valid_data_loader, config["property_"], device)
    
                    # Log batch loss to wandb
                    wandb.log({
                        "Batch": batch_tot_idx,
                        "x_reconst": x_reconst,
                        "kl_loss": kl_loss,
                        "Train batch Loss": loss.item(),
                        "Validation Loss": val_loss_overall_avg
                    })
                
                loss.backward()
                optimizer.step()
                
            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch
            
            loss_timebatch_avg = sum(loss_avg)/len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print('i_epoch:{}, tb: {}, last timebatch loss: {}, avg_loss: {}, time: {}'.format(i_epoch, timeBatchIndex, loss.item(), loss_timebatch_avg, elapsed_timebatch))
    
        loss_overall_avg = sum(loss_overall)/len(loss_overall)
        
        # val_loss_overall_avg = validate_model(model, valid_data_loader, property_, device)
        
        if min_valid_loss > loss_overall_avg:     
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss_overall_avg:.6f}) \t Saving The Model')
            min_valid_loss = loss_overall_avg
            no_improvement_count = 0
            slow_improvement_count = 0
            # Saving State Dict
            torch.save(model.state_dict(), directory + '/best_model_', _use_new_zipfile_serialization=False)
        else:
            no_improvement_count += 1
            if loss_overall_avg - min_valid_loss <= 0.001:  # Adjust this threshold as needed
                slow_improvement_count += 1
        
        #model.train()
        scheduler.step()
        
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": i_epoch,
            "loss_timebatch_avg_loss": loss_timebatch_avg,
            "loss_overall_avg": loss_overall_avg,
            "min_train_loss": min_valid_loss,
        })
            
        
        # if no_improvement_count >= patience or slow_improvement_count >= slow_improvement_patience:
        #     break  # Stop training
        
    # Code or process to be measured goes here
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")
    
    # Plotting results
#     create_position_density_plots(x, y, z, x_pr, y_pr, z_pr, bins=100, t=t_index)

#     create_momentum_density_plots(px, py, pz, px_pr, py_pr, pz_pr, bins=100, t=t_index)

#     create_force_density_plots(fx, fy, fz, fx_pr, fy_pr, fz_pr, bins=100, t=t_index)










