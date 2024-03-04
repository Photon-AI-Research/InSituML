import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import wandb

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
    

def create_position_density_plots(x, y, z,
                                  x_pr, y_pr, z_pr,
                                  bins=100, t=1000, path='',
                                  enable_wandb = False):
    
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
        
    if enable_wandb == True:
        # Log the overlapping histogram plot
        wandb.log({"XY vs XZ vs YZ histograms": wandb.Image(plt)})

        plt.close()
    else:    
        plt.show() 

def create_momentum_density_plots(px, py, pz,
                                  px_pr, py_pr, pz_pr,
                                  px_pr_ae, py_pr_ae, pz_pr_ae,
                                  chamfers_loss=None,emd_loss=None,
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
    plt.figure(figsize=(15, 15)) 
    
    # px-py Plane Ground Truth
    plt.subplot(331)
    plt.hist2d(px, py, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('py')
    plt.title('px-py GT at t = {}, box = {}'.format(t,gpu_box))
    
    # px-pz Plane Ground Truth
    plt.subplot(332)
    plt.hist2d(px, pz, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('pz')
    plt.title('px-pz GT at t = {}, box = {}'.format(t,gpu_box))
    
    # py-pz Plane Ground Truth
    plt.subplot(333)
    plt.hist2d(py, pz, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py')
    plt.ylabel('pz')
    plt.title('py-pz GT at t = {}, box = {}'.format(t,gpu_box))
    
    # px-py Plane Prediction
    plt.subplot(334)
    plt.hist2d(px_pr, py_pr, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('py_pr')
    plt.title('px-py MAF Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # px-pz Plane Prediction
    plt.subplot(335)
    plt.hist2d(px_pr, pz_pr, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('pz_pr')
    plt.title('px-pz MAF Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # py-pz Plane Prediction
    plt.subplot(336)
    plt.hist2d(py_pr, pz_pr, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py_pr')
    plt.ylabel('pz_pr')
    plt.title('py-pz MAF Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # px-py Plane Prediction
    plt.subplot(337)
    plt.hist2d(px_pr_ae, py_pr_ae, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('py_pr')
    plt.title('px-py AE Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # px-pz Plane Prediction
    plt.subplot(338)
    plt.hist2d(px_pr_ae, pz_pr_ae, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('pz_pr')
    plt.title('px-pz AE Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # py-pz Plane Prediction
    plt.subplot(339)
    plt.hist2d(py_pr_ae, pz_pr_ae, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py_pr')
    plt.ylabel('pz_pr')
    plt.title('py-pz AE Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
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
                               fx_pr, fy_pr, fz_pr,
                               fx_pr_ae, fy_pr_ae, fz_pr_ae,
                               chamfers_loss=None,emd_loss=None,
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
    plt.figure(figsize=(15, 15))  # Adjust the figure size
    
    # fx-fy Plane Ground Truth
    plt.subplot(331)
    plt.hist2d(fx, fy, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fy')
    plt.title('fx-fy GT at t = {}, box = {}'.format(t,gpu_box))
    
    # fx-fz Plane Ground Truth
    plt.subplot(332)
    plt.hist2d(fx, fz, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fz')
    plt.title('fx-fz GT at t = {}, box = {}'.format(t,gpu_box))
    
    # fy-fz Plane Ground Truth
    plt.subplot(333)
    plt.hist2d(fy, fz, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy')
    plt.ylabel('fz')
    plt.title('fy-fz GT at t = {}, box = {}'.format(t,gpu_box))
    
    # fx-fy Plane Prediction
    plt.subplot(334)
    plt.hist2d(fx_pr, fy_pr, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fy_pr')
    plt.title('fx-fy MAF Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fx-fz Plane Prediction
    plt.subplot(335)
    plt.hist2d(fx_pr, fz_pr, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fz_pr')
    plt.title('fx-fz MAF Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fy-fz Plane Prediction
    plt.subplot(336)
    plt.hist2d(fy_pr, fz_pr, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy_pr')
    plt.ylabel('fz_pr')
    plt.title('fy-fz MAF Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fx-fy Plane Prediction
    plt.subplot(337)
    plt.hist2d(fx_pr_ae, fy_pr_ae, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fy_pr')
    plt.title('fx-fy AE Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fx-fz Plane Prediction
    plt.subplot(338)
    plt.hist2d(fx_pr_ae, fz_pr_ae, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fz_pr')
    plt.title('fx-fz AE Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
    # fy-fz Plane Prediction
    plt.subplot(339)
    plt.hist2d(fy_pr_ae, fz_pr_ae, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy_pr')
    plt.ylabel('fz_pr')
    plt.title('fy-fz AE Pred at t = {}, box = {}{}'.format(t,gpu_box, loss_info))
    
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


def smooth_data(data, window_size=5):
    """Smooth data using a moving average."""
    return uniform_filter1d(data, size=window_size, mode='nearest')

def plot_radiation(ground_truth_intensity, predicted_intensity=None, t=1000, gpu_box =0, path='',
                                  enable_wandb = False):
    """
    Plot radiation intensity against frequency.

    Parameters:
    - intensity: A tensor of radiation spectra values.

    """
    
    frequency = np.load("/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/omega.npy")
    ground_truth_intensity = ground_truth_intensity.cpu().numpy()
    ground_truth_smoothed = smooth_data(ground_truth_intensity)
    # print('ground_truth_intensity', ground_truth_intensity.max())
    plt.figure(figsize=(10, 6))
    
    # Plot smoothed ground truth intensity
    # plt.plot(frequency, ground_truth_smoothed, label='GT Radiation Intensity (Smoothed)', color='blue', linewidth=2)
    
    # Plot raw ground truth intensity with lower opacity
    plt.plot(ground_truth_intensity, label='GT Radiation Intensity (Raw)', color='blue', linewidth=2)
    # plt.plot(frequency, ground_truth_intensity, label='GT Radiation Intensity (Raw)', color='blue', linewidth=1, alpha=0.3)
    
    
    # Plot predicted intensity if provided
    if predicted_intensity is not None:
        predicted_intensity = predicted_intensity.cpu().numpy()
        predicted_smoothed = smooth_data(predicted_intensity)
        # print('predicted_intensity', predicted_intensity.max())
        
        # Plot smoothed predicted intensity
        plt.plot(predicted_smoothed, label='Predicted Radiation Intensity (Smoothed)', linestyle='--', color='red', marker='o', markersize=5, zorder=1)
        
        # Plot raw predicted intensity with lower opacity
        plt.plot(predicted_intensity, label='Predicted Radiation Intensity (Raw)', linestyle='--', color='red', alpha=0.3, zorder=0, markersize=3)
    
    
    plt.xlabel('Frequency')
    plt.ylabel('Intensity (log scale)')
    plt.title('Radiation Intensity vs. Frequency t = {}, box = {}'.format(t,gpu_box))
    plt.legend()
    plt.grid(True)
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/radiation_plots_{}_{}.png'.format(t,gpu_box))
    
    if enable_wandb == True:
        # Log the overlapping histogram plot
        wandb.log({"Radiation (t={},box={})".format(t,gpu_box): wandb.Image(plt)})
            
        plt.close()
    else:    
        plt.show() 

        
def plot_1d_histograms(p_gt, p_pr, bins=30, alpha=0.5, t=1000, gpu_box =0, path='',
                                  enable_wandb = False):
    """
    Plot 1D histograms for two PyTorch tensors.

    Parameters:
    - p_gt: PyTorch tensor, ground truth values.
    - p_pr: PyTorch tensor, predicted values.
    - bins: int, number of bins for the histogram.
    - alpha: float, transparency level for the histogram bars.
    """
    # Ensure tensors are on CPU and convert to numpy for plotting
    p_gt_np = p_gt.cpu().detach().numpy()
    p_pr_np = p_pr.cpu().detach().numpy()

    # Create the histograms
    plt.figure(figsize=(10, 6))
    plt.hist(p_gt_np, bins=bins, alpha=alpha, label='Ground Truth', density=True)
    plt.hist(p_pr_np, bins=bins, alpha=alpha, label='Prediction', density=True)
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('1D Histograms of GT and Pred of latent space t = {}, box = {}'.format(t,gpu_box))
    plt.legend()
    
        # Save the plots as image files
    if path:
        plt.savefig(path + '/histogram_plots_{}_{}.png'.format(t,gpu_box))
    
    if enable_wandb == True:
        # Log the overlapping histogram plot
        wandb.log({"1D Histograms of Ground Truth and Prediction (t={},box={})".format(t,gpu_box): wandb.Image(plt)})
            
        plt.close()
    else:    
        plt.show() 

        
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

    
