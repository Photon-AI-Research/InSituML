import numpy as np
import torch
import matplotlib.pyplot as plt
from math import log, pi
import torch
import random
from collections import deque
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import inspect

class Normalizer:
    def __init__(self,
                 global_mean_std_file='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_002/global_stats.npz',
                 mean_std_file='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_002/global_stats_1900_2001.npz',
                 mean_std_per_time_file='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_002/global_stats_per_time.npz',
                 minmax_file='/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/minmax_002.npy'):
        
        self.global_mean_std_file = global_mean_std_file
        self.mean_std_file = mean_std_file
        self.mean_std_per_time_file = mean_std_per_time_file
        self.minmax_file = minmax_file

    def normalize_data(self, original_array, method='positions', *args, **kwargs):
        if method == 'positions':
            return self.normalize_positions(original_array)
        elif method == 'global_mean_6d':
            return self.normalize_global_mean_6d(original_array)
        elif method == 'mean_6d':
            return self.normalize_mean_6d(original_array)
        elif method == 'mean_per_time_6d':
            if 'timestep_index' in kwargs:
                return self.normalize_mean_per_time_6d(original_array, kwargs['timestep_index'])
            else:
                raise ValueError("timestep_index must be specified for mean_per_time_6d normalization.")
        elif method == 'min_max_6d':
            return self.normalize_min_max_6d(original_array)
        else:
            raise ValueError("Invalid normalization method specified.")


    def normalize_positions(self, original_array):
        xyz_columns = original_array[:, :3]
        mins = xyz_columns.min(axis=0)
        maxs = xyz_columns.max(axis=0)
        normalized_xyz_columns = (xyz_columns - mins) / (maxs - mins)
        normalized_array = np.concatenate((normalized_xyz_columns, original_array[:, 3:]), axis=1)
        return normalized_array
    
    
    def normalize_global_mean_6d(self, original_array):
        data_stats = np.load(self.global_mean_std_file)
        # Extract global mean and standard deviation
        global_mean = np.concatenate((data_stats['mean_momentum'], data_stats['mean_force']))
        global_std = np.concatenate((data_stats['std_momentum'], data_stats['std_force']))
        
        # Normalize x, y, z positions
        xyz_columns = original_array[:, :3]
        mins = xyz_columns.min(axis=0)
        maxs = xyz_columns.max(axis=0)
        xyz_columns_normalized = (xyz_columns - mins) / (maxs - mins)

        # Mean normalization for the rest of the dimensions (momentum and force components)
        other_columns_normalized = (original_array[:, 3:] - global_mean) / global_std
        
        # Combine the normalized columns into one array
        normalized_array = np.concatenate((xyz_columns_normalized, other_columns_normalized), axis=1)
        return normalized_array

    def normalize_mean_6d(self, original_array):
        data_stats = np.load(self.mean_std_file)
        # Extract global mean and standard deviation
        global_mean_momentum = data_stats['mean_momentum']
        global_mean_force = data_stats['mean_force']
        global_std_momentum = data_stats['std_momentum']
        global_std_force = data_stats['std_force']

        # Normalize x, y, z positions
        xyz_columns = original_array[:, :3]
        mins = xyz_columns.min(axis=0)
        maxs = xyz_columns.max(axis=0)
        xyz_columns_normalized = (xyz_columns - mins) / (maxs - mins)

        # Mean normalization for momentum dimensions (assuming they are next 3 columns)
        momentum_columns_normalized = (original_array[:, 3:6] - global_mean_momentum) / global_std_momentum

        # Mean normalization for force dimensions (assuming they are the last 3 columns)
        force_columns_normalized = (original_array[:, 6:9] - global_mean_force) / global_std_force

        # Combine the normalized columns into one array
        normalized_array = np.concatenate((xyz_columns_normalized, momentum_columns_normalized, force_columns_normalized), axis=1)
        return normalized_array


    def normalize_mean_per_time_6d(self, original_array, timestep_index):
        """
        Normalize the input array at the given timestep index using mean and standard deviation from the file.
        """
        data_stats = np.load(self.mean_std_per_time_file)
        # Extract mean and standard deviation for the given timestep
        global_mean = np.concatenate((data_stats['mean_momentum'][timestep_index], data_stats['mean_force'][timestep_index]))
        global_std = np.concatenate((data_stats['std_momentum'][timestep_index], data_stats['std_force'][timestep_index]))

        # Normalize x, y, z positions
        xyz_columns = original_array[:, :3]
        mins = xyz_columns.min(axis=0)
        maxs = xyz_columns.max(axis=0)
        xyz_columns_normalized = (xyz_columns - mins) / (maxs - mins)

        # Mean normalization for the rest of the dimensions (momentum and force components)
        other_columns_normalized = (original_array[:, 3:] - global_mean) / global_std

        # Combine the normalized columns into one array
        normalized_array = np.concatenate((xyz_columns_normalized, other_columns_normalized), axis=1)
        return normalized_array


    def normalize_min_max_6d(self, original_array):
        """
        Normalize the input array using min-max values from the given file.
        """
        # Normalize x, y, z columns between 0 and 1
        xyz_columns = original_array[:, :3]
        mins = xyz_columns.min(axis=0)
        maxs = xyz_columns.max(axis=0)
        xyz_columns_normalized = (xyz_columns - mins) / (maxs - mins)

        # Normalize the other 6 columns between -1 and 1
        other_columns = original_array[:, 3:]
        min_max = np.load(self.minmax_file)
        min_vals = min_max[0, 3:]
        max_vals = min_max[1, 3:]
        normalized_other_columns = 2 * (other_columns - min_vals) / (max_vals - min_vals) - 1

        # Concatenate normalized columns with x, y, z columns
        normalized_array = np.concatenate((xyz_columns_normalized, normalized_other_columns), axis=1)
        normalized_array = np.array(normalized_array, dtype = np.float32)
        
        return normalized_array
    

def inspect_and_select(base):
    
    def decorator(**all_input_pars):

        input_vals = {k:all_input_pars[k] for k, v in inspect.signature(base).parameters.items() 
                      if k in all_input_pars}
    
        return base(**input_vals)
    
    return decorator
    

def validate_model(model, valid_data_loader, property_, device):
    model.eval()
    val_loss_avg = []
    
    with torch.no_grad():
        for idx in range(len(valid_data_loader)):
            timestep_index, validation_boxes, p, _ = valid_data_loader[idx]
            p = filter_dims[property_](p)
            p = p.to(device)
            val_loss, p_pr = model(p)
            val_loss_avg.append(val_loss.mean().item())
    val_loss_overall_avg = sum(val_loss_avg) / (len(val_loss_avg) + 1e-8)
    return val_loss_overall_avg

# def filter_dims(phase_space, property_="positions"):
#
#     if property_ == "positions":
#         return phase_space[:,:,:3]
#     elif property_ == "momentum":
#         return phase_space[:,:,3:6]
#     elif property_ == "force":
#         return phase_space[:,:,6:]
#     elif property_ == "momentum_force":
#         return phase_space[:,:,3:]
#     elif property_ == "momentum_6":
#         return phase_space[:,:,:3]
#     elif property_ == "force_6":
#         return phase_space[:,:,3:6]
#     else:
#         return phase_space

filter_dims ={

    "positions" : lambda x:x[:,:,:3],
    "momentum" : lambda x:x[:,:,3:6],
    "force": lambda x:x[:,:,6:],
    "momentum_force": lambda x:x[:,:,3:],
    "momentum_6":lambda x:x[:,:,:3],
    "force_6":lambda x:x[:,:,3:6],
    "all":lambda x:x
    }
def save_visual_multi(*args, property_):
    property_run = ["positions", "momentum", "force"] if property_=="all" else ["momentum", "force"]

    deque(save_visual(*args, property_1, running6or9=property_) \
             for property_1 in property_run)
    
def save_visual(model, timebatch, wandb, timeInfo, info_image_path, 
                property_, running6or9=False):
    
    #avoiding turning on model.eval
    random_input, _ = timebatch[torch.randint(len(timebatch),(1,))[0]]
    #if model is being trained on all the dimensions, it changes the order filtering
    # and inference.
    if running6or9:
        random_input = filter_dims[running6or9](random_input)
        random_output = model.reconstruct_input(random_input.to(device))

        dims = "" if running6or9=="all" else "_6"

        random_input = filter_dims[property_+dims](random_input)
        random_output = filter_dims[property_+dims](random_output)
        
    else:
        random_input = filter_dims[property_](random_input)
        random_output = model.reconstruct_input(random_input.to(device))

    all_var_to_plot = random_input[0].transpose(1,0).tolist() + random_output[0].transpose(1,0).tolist()
    if property_ == "positions":
        create_position_density_plots(*all_var_to_plot, path=info_image_path,
                                      t=timeInfo, wandb=wandb)
    elif property_ == "momentum":
        create_momentum_density_plots(*all_var_to_plot, path=info_image_path,
                                      t=timeInfo, wandb=wandb)
    elif property_ == "force":
        create_force_density_plots(*all_var_to_plot, path=info_image_path,
                                   t=timeInfo,wandb=wandb)

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
    bins_px = np.linspace(-0.4, 0.4, bins)
    bins_py = np.linspace(-0.4, 0.4, bins)
    bins_pz = np.linspace(-0.2, 0.2, bins)
    
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
    bins_fx = np.linspace(-0.006, 0.006, bins)
    bins_fy = np.linspace(-0.006, 0.006, bins)
    bins_fz = np.linspace(-0.002, 0.002, bins)
    
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
