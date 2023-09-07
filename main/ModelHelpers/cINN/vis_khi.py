import os
import numpy as np
import torch
from model import model_MAF as model_MAF
import torch.optim as optim
import time
import wandb
import sys
import matplotlib.pyplot as plt


def create_density_plots(x,y,z, bins=200, t=1000, path =''):
    
    # Specify the number of bins for each axis
    bins_x = np.linspace(min(x), max(x), 200)  
    bins_y = np.linspace(min(y), max(y), 200)  
    bins_z = np.linspace(min(z), max(z), 200)  
    
    # Create subplots for each plane
    plt.figure(figsize=(15, 5))
    
    # XY Plane
    plt.subplot(131)
    plt.hist2d(x, y, bins=[bins_x, bins_y], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plane Density Plot at t = {}'.format(t))
    
    # XZ Plane
    plt.subplot(132)
    plt.hist2d(x, z, bins=[bins_x, bins_z], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('XZ Plane Density Plot at t = {}'.format(t))
    
    # YZ Plane
    plt.subplot(133)
    plt.hist2d(y, z, bins=[bins_y, bins_z], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('YZ Plane Density Plot at t = {}'.format(t))
    
    plt.tight_layout()
    
    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/density_plots_{}.png'.format(t))
    
    plt.show()

def create_force_scatter_plots(fx, fy, fz, t=1000, path =''):
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.scatter(fx, fy, s=1, alpha=0.5)
    plt.xlabel('Fx')
    plt.ylabel('Fy')
    plt.title('Fx vs. Fy at t={}'.format(t))
    
    plt.subplot(132)
    plt.scatter(fx, fz, s=1, alpha=0.5)
    plt.xlabel('Fx')
    plt.ylabel('Fz')
    plt.title('Fx vs. Fz at t={}'.format(t))
    
    plt.subplot(133)
    plt.scatter(fy, fz, s=1, alpha=0.5)
    plt.xlabel('Fy')
    plt.ylabel('Fz')
    plt.title('Fy vs. Fz at t={}'.format(t))
    
    plt.tight_layout()

    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/force_scatter_plots_{}.png'.format(t))
    
    plt.show()

def create_momentum_scatter_plots(px, py, pz, t=1000, path =''):
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.scatter(px, py, s=1, alpha=0.5)
    plt.xlabel('Px')
    plt.ylabel('Py')
    plt.title('Px vs. Py at t={}'.format(t))
    
    plt.subplot(132)
    plt.scatter(px, pz, s=1, alpha=0.5)
    plt.xlabel('Px')
    plt.ylabel('Pz')
    plt.title('Px vs. Pz at t={}'.format(t))
    
    plt.subplot(133)
    plt.scatter(py, pz, s=1, alpha=0.5)
    plt.xlabel('Py')
    plt.ylabel('Pz')
    plt.title('Py vs. Pz at t={}'.format(t))
    
    plt.tight_layout()
    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/momentum_scatter_plots_{}.png'.format(t))
    
    plt.show()

def create_momentum_histogram_plots(px, py, pz, bins=50, t=1000, path ='' ):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.hist(px, bins=bins, density=True, color='blue', alpha=0.7)
    plt.xlabel('Px')
    plt.ylabel('Frequency')
    plt.title('Histogram of Px at t={}'.format(t))

    plt.subplot(132)
    plt.hist(py, bins=bins, density=True, color='green', alpha=0.7)
    plt.xlabel('Py')
    plt.ylabel('Frequency')
    plt.title('Histogram of Py at t={}'.format(t))

    plt.subplot(133)
    plt.hist(pz, bins=bins, density=True, color='red', alpha=0.7)
    plt.xlabel('Pz')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pz at t={}'.format(t))

    plt.tight_layout()

    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/momentum_histogram_plots_{}.png'.format(t))
    
    plt.show()

def create_force_histogram_plots(fx, fy, fz, bins=50, t=1000, path ='' ):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.hist(fx, bins=bins, density=True, color='blue', alpha=0.7)
    plt.xlabel('Fx')
    plt.ylabel('Frequency')
    plt.title('Histogram of Fx at t={}'.format(t))

    plt.subplot(132)
    plt.hist(fy, bins=bins, density=True, color='green', alpha=0.7)
    plt.xlabel('Fy')
    plt.ylabel('Frequency')
    plt.title('Histogram of Fy at t={}'.format(t))

    plt.subplot(133)
    plt.hist(fz, bins=bins, density=True, color='red', alpha=0.7)
    plt.xlabel('Fz')
    plt.ylabel('Frequency')
    plt.title('Histogram of Fz at t={}'.format(t))

    plt.tight_layout()

    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/force_histogram_plots_{}.png'.format(t))
    
    plt.show()

def create_scatter_force_vs_xyz(x, y, z, fx, fy, fz, t=1000, path =''):
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(x, fx, s=1, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Fx')
    plt.title('Fx vs. x at t={}'.format(t))

    plt.subplot(132)
    plt.scatter(y, fy, s=1, alpha=0.5)
    plt.xlabel('y')
    plt.ylabel('Fy')
    plt.title('Fy vs. y at t={}'.format(t))

    plt.subplot(133)
    plt.scatter(z, fz, s=1, alpha=0.5)
    plt.xlabel('z')
    plt.ylabel('Fz')
    plt.title('Fz vs. z at t={}'.format(t))

    plt.tight_layout()

    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/scatter_force_vs_xyz_{}.png'.format(t))
    
    plt.show()

def create_scatter_momentum_vs_xyz(x, y, z, px, py, pz, t=1000, path =''):
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(x, px, s=1, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Px')
    plt.title('Px vs. x at t={}'.format(t))

    plt.subplot(132)
    plt.scatter(y, py, s=1, alpha=0.5)
    plt.xlabel('y')
    plt.ylabel('Py')
    plt.title('Py vs. y at t={}'.format(t))

    plt.subplot(133)
    plt.scatter(z, pz, s=1, alpha=0.5)
    plt.xlabel('z')
    plt.ylabel('Pz')
    plt.title('Pz vs. z at t={}'.format(t))

    plt.tight_layout()

    # Save the plots as image files
    plt.savefig('/home/rustam75/KHI/scatter_momentum_vs_xyz_{}.png'.format(t))
    
    plt.show()



file_template = "/bigdata/hplsim/aipp/Jeyhun/khi/particle/{}.npy"

for i in range(2001):
    if i % 100 == 0:
        print(i)
        file_path = file_template.format(i)
        data = np.load(file_path)
        print(data.shape)

        
        x = data[:, 0]  # X coordinates
        y = data[:, 1]  # Y coordinates
        z = data[:, 2]  # Z coordinates
        
        px = data[:, 3]  # Px component of momentum
        py = data[:, 4]  # Py component of momentum
        pz = data[:, 5]  # Pz component of momentum
        
        fx = data[:, 6]  # Fx component of force
        fy = data[:, 7]  # Fy component of force
        fz = data[:, 8]  # Fz component of force

        create_density_plots(x,y,z, bins=200, t=i)
        create_force_scatter_plots(fx, fy, fz, t=i)
        create_momentum_scatter_plots(px, py, pz, t=i)
        create_momentum_histogram_plots(px, py, pz, t=i)
        create_force_histogram_plots(fx, fy, fz, t=i)
        create_scatter_force_vs_xyz(x, y, z, fx, fy, fz, t=i)
        create_scatter_momentum_vs_xyz(x, y, z, px, py, pz, t=i)