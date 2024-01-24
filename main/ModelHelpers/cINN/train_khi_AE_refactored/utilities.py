import numpy as np
import torch
import matplotlib.pyplot as plt
from math import log, pi
import torch
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_gaussian(m, v):
    epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size())).to(device)
    z = m + torch.sqrt(v) * epsilon
    return z

def kl_normal(qm,qv,pm,pv):
    #checking how different is it from guassian distribution with
    # zero mean and 1 standard deviation. 
    # tensor shape (Batch,dim)
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def create_position_density_plots(x, y, z,
                                  x_pr, y_pr, z_pr,
                                  bins=100, t=1000, path='',
                                  wandb = None):
    
    # Specify the number of bins for each axis
    bins_x = np.linspace(min(x), max(x), bins)
    bins_y = np.linspace(min(y), max(y), bins)
    bins_z = np.linspace(min(z), max(z), bins)
    
    # Create subplots for each plane
    plt.figure(figsize=(15, 10))
    
    # XY Plane Ground Truth
    plt.subplot(231)
    plt.hist2d(x, y, bins=[bins_x, bins_y], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plane Ground Truth at t = {}'.format(t))
    
    # XZ Plane Ground Truth
    plt.subplot(232)
    plt.hist2d(x, z, bins=[bins_x, bins_z], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('XZ Plane Ground Truth at t = {}'.format(t))
    
    # YZ Plane Ground Truth
    plt.subplot(233)
    plt.hist2d(y, z, bins=[bins_y, bins_z], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('YZ Plane Ground Truth at t = {}'.format(t))
    
    # XY Plane Prediction
    plt.subplot(234)
    plt.hist2d(x_pr, y_pr, bins=[bins_x, bins_y], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plane Prediction at t = {}'.format(t))
    
    # XZ Plane Prediction
    plt.subplot(235)
    plt.hist2d(x_pr, z_pr, bins=[bins_x, bins_z], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('XZ Plane Prediction at t = {}'.format(t))
    
    # YZ Plane Prediction
    plt.subplot(236)
    plt.hist2d(y_pr, z_pr, bins=[bins_y, bins_z], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('YZ Plane Prediction at t = {}'.format(t))
    
    plt.tight_layout()  # Adjust subplot spacing
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/density_plots_{}.png'.format(t))
        
    if wandb is not None:
        # Log the overlapping histogram plot
        wandb.log({"XY vs XZ vs YZ histograms": wandb.Image(plt)})

        plt.close()
    else:    
        plt.show() 

def create_momentum_density_plots(px, py, pz,
                                  px_pr, py_pr, pz_pr,
                                  bins=100, t=1000, path='',
                                  wandb = None):
    
    # Specify the number of bins for each axis
    bins_px = np.linspace(min(px), max(px), bins)
    bins_py = np.linspace(min(py), max(py), bins)
    bins_pz = np.linspace(min(pz), max(pz), bins)
    
    # Create subplots for each plane
    plt.figure(figsize=(15, 10)) 
    
    # px-py Plane Ground Truth
    plt.subplot(231)
    plt.hist2d(px, py, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('py')
    plt.title('px-py Plane Ground Truth at t = {}'.format(t))
    
    # px-pz Plane Ground Truth
    plt.subplot(232)
    plt.hist2d(px, pz, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('pz')
    plt.title('px-pz Plane Ground Truth at t = {}'.format(t))
    
    # py-pz Plane Ground Truth
    plt.subplot(233)
    plt.hist2d(py, pz, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py')
    plt.ylabel('pz')
    plt.title('py-pz Plane Ground Truth at t = {}'.format(t))
    
    # px-py Plane Prediction
    plt.subplot(234)
    plt.hist2d(px_pr, py_pr, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('py_pr')
    plt.title('px-py Plane Prediction at t = {}'.format(t))
    
    # px-pz Plane Prediction
    plt.subplot(235)
    plt.hist2d(px_pr, pz_pr, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('pz_pr')
    plt.title('px-pz Plane Prediction at t = {}'.format(t))
    
    # py-pz Plane Prediction
    plt.subplot(236)
    plt.hist2d(py_pr, pz_pr, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py_pr')
    plt.ylabel('pz_pr')
    plt.title('py-pz Plane Prediction at t = {}'.format(t))
    
    plt.tight_layout()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/momentum_density_plots_{}.png'.format(t))
    
    if wandb is not None:
        # Log the overlapping histogram plot
        wandb.log({"Px vs Py vs Pz histograms": wandb.Image(plt)})

        plt.close()
    else:
        plt.show()   


def create_force_density_plots(fx, fy, fz,
                               fx_pr, fy_pr, fz_pr,
                               bins=100, t=1000, path='',
                               wandb = None):
    
    # Specify the number of bins for each axis
    bins_fx = np.linspace(min(fx), max(fx), bins)
    bins_fy = np.linspace(min(fy), max(fy), bins)
    bins_fz = np.linspace(min(fz), max(fz), bins)
    
    # Create subplots for each plane
    plt.figure(figsize=(15, 10))  # Adjust the figure size
    
    # fx-fy Plane Ground Truth
    plt.subplot(231)
    plt.hist2d(fx, fy, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fy')
    plt.title('fx-fy Plane Ground Truth at t = {}'.format(t))
    
    # fx-fz Plane Ground Truth
    plt.subplot(232)
    plt.hist2d(fx, fz, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fz')
    plt.title('fx-fz Plane Ground Truth at t = {}'.format(t))
    
    # fy-fz Plane Ground Truth
    plt.subplot(233)
    plt.hist2d(fy, fz, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy')
    plt.ylabel('fz')
    plt.title('fy-fz Plane Ground Truth at t = {}'.format(t))
    
    # fx-fy Plane Prediction
    plt.subplot(234)
    plt.hist2d(fx_pr, fy_pr, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fy_pr')
    plt.title('fx-fy Plane Prediction at t = {}'.format(t))
    
    # fx-fz Plane Prediction
    plt.subplot(235)
    plt.hist2d(fx_pr, fz_pr, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fz_pr')
    plt.title('fx-fz Plane Prediction at t = {}'.format(t))
    
    # fy-fz Plane Prediction
    plt.subplot(236)
    plt.hist2d(fy_pr, fz_pr, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy_pr')
    plt.ylabel('fz_pr')
    plt.title('fy-fz Plane Prediction at t = {}'.format(t))
    
    plt.tight_layout()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/force_density_plots_{}.png'.format(t))
    
    if wandb is not None:
        # Log the overlapping histogram plot
        wandb.log({"Fx vs Fy vs FZ histograms": wandb.Image(plt)})

        plt.close()
    else:
        plt.show()  
        
        
def random_sample(data, sample_size):
    # Check if the sample size is greater than the number of points in the data
    if sample_size > data.shape[0]:
        raise ValueError("Sample size exceeds the number of points in the data")

    # Randomly sample 'sample_size' points
    random_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    sampled_data = data[random_indices]

    return sampled_data

def normalize_columns(original_array):
    xyz_columns = original_array[:, :3]
    x_min, x_max = xyz_columns[:, 0].min(), xyz_columns[:, 0].max()
    y_min, y_max = xyz_columns[:, 1].min(), xyz_columns[:, 1].max()
    z_min, z_max = xyz_columns[:, 2].min(), xyz_columns[:, 2].max()

    xyz_columns[:, 0] = (xyz_columns[:, 0] - x_min) / (x_max - x_min)
    xyz_columns[:, 1] = (xyz_columns[:, 1] - y_min) / (y_max - y_min)
    xyz_columns[:, 2] = (xyz_columns[:, 2] - z_min) / (z_max - z_min)

    normalized_array = np.concatenate((xyz_columns, original_array[:, 3:]), axis=1)
    return normalized_array

def chamfersDist(a, b):
    d = torch.cdist(a, b, p=2)
    return torch.sum(torch.min(d, -1).values) + torch.sum(torch.min(d, -2).values)
