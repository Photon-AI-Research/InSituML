import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Losses
def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def fit(input, target):
    return torch.mean((input - target)**2)
    
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
                                  bins=100, t=1000, path='',
                                  enable_wandb = False):
    
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
    
    if enable_wandb == True:
        # Log the overlapping histogram plot
        wandb.log({"Px vs Py vs Pz histograms": wandb.Image(plt)})

        plt.close()
    else:
        plt.show()   


def create_force_density_plots(fx, fy, fz,
                               fx_pr, fy_pr, fz_pr,
                               bins=100, t=1000, path='',
                               enable_wandb = False):
    
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
    
    if enable_wandb == True:
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


def smooth_data(data, window_size=5):
    """Smooth data using a moving average."""
    return uniform_filter1d(data, size=window_size, mode='nearest')

def plot_radiation(ground_truth_intensity, predicted_intensity=None, frequency_range=512, t=1000, gpu_box=0, path='',
                   enable_wandb=False):
    """
    Plot radiation intensity against frequency and compute MSE and relative MSE
    between ground truth and prediction. Compatible with both NumPy arrays and PyTorch tensors.

    Parameters:
    - ground_truth_intensity: A tensor or array of ground truth radiation spectra values.
    - predicted_intensity: A tensor or array of predicted radiation spectra values (optional).
    - t: Time step for the title (default=1000).
    - gpu_box: Identifier for the GPU box (default=0).
    - path: Path to save the plot (optional).
    - enable_wandb: Enable logging to Weights & Biases (default=False).
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    def to_numpy(data):
        """Convert PyTorch tensor to NumPy array if necessary."""
        if 'torch' in str(type(data)):
            return data.cpu().numpy()
        return data

    # Load frequency data
    frequency = np.load("/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/omega.npy")[:frequency_range]
    
    # Ensure ground_truth_intensity and predicted_intensity are NumPy arrays
    ground_truth_intensity = to_numpy(ground_truth_intensity)[:frequency_range]
    ground_truth_smoothed = smooth_data(ground_truth_intensity)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequency, ground_truth_intensity, label='GT Radiation Intensity (Raw)', color='blue', linewidth=2)
    
    mse, rel_mse = 0, 0  # Initialize MSE and Relative MSE
    
    if predicted_intensity is not None:
        predicted_intensity = to_numpy(predicted_intensity)[:frequency_range]
        predicted_smoothed = smooth_data(predicted_intensity)
        
        plt.plot(frequency, predicted_smoothed, label='Predicted Radiation Intensity (Smoothed)', linestyle='--', color='red', marker='o', markersize=5, zorder=1)
        plt.plot(frequency, predicted_intensity, label='Predicted Radiation Intensity (Raw)', linestyle='--', color='red', alpha=0.3, zorder=0, markersize=3)
        
        # Compute MSE
        mse = np.mean((ground_truth_intensity - predicted_intensity) ** 2)
        
        # Compute Relative MSE
        rel_mse = mse / np.mean(ground_truth_intensity ** 2)
    
    # Update plot title with MSE and Relative MSE if prediction is provided
    if predicted_intensity is not None:
        plt.title(f'Radiation Intensity vs. Frequency t = {t}, box = {gpu_box}\nMSE = {mse:.2e}, Relative MSE = {rel_mse:.2e}')
    else:
        plt.title(f'Radiation Intensity vs. Frequency t = {t}, box = {gpu_box}')
    
    plt.xlabel('Frequency')
    plt.ylabel('Intensity (log scale)')
    plt.legend()
    plt.grid(True)
    
    if path:
        plt.savefig(f'{path}/radiation_plots_{t}_{gpu_box}.png')
    
    if enable_wandb:
        import wandb
        wandb.log({"Radiation (t={},box={})".format(t, gpu_box): wandb.Image(plt)})
        plt.close()
    else:
        plt.show()
        
def save_checkpoint(model, optimizer, path, last_loss, epoch, min_valid_loss=None, wandb_run_id=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'last_loss': last_loss.item(),
        'epoch': epoch,
    }
    
    if min_valid_loss is not None:
        state['min_valid_loss'] = min_valid_loss

    if wandb_run_id is not None:
        state['wandb_run_id'] = wandb_run_id

    torch.save(state, path + '/model_' + str(epoch))
    
        
def load_checkpoint(path_to_checkpoint, model, optimizer):
    # Load the saved file
    checkpoint = torch.load(path_to_checkpoint)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    last_loss = checkpoint['last_loss']
    epoch = checkpoint['epoch']
    
    min_valid_loss = checkpoint.get('min_valid_loss', None)
    wandb_run_id = checkpoint.get('wandb_run_id', None)

    return model, optimizer, last_loss, min_valid_loss, epoch, wandb_run_id

def save_checkpoint_conditionally(model, optimizer, path, epoch, last_loss, min_valid_loss=None, wandb_run_id=None):
    checkpoint_filename = f'model_{epoch}'
    checkpoint_path = os.path.join(path, checkpoint_filename)

    # Check if the checkpoint for this epoch already exists
    if not os.path.exists(checkpoint_path):

        save_checkpoint(model, optimizer, path, last_loss, min_valid_loss, epoch, wandb_run_id)
        print(f"Checkpoint for epoch {epoch} saved.")
    else:
        print(f"Checkpoint for epoch {epoch} already exists. Skipping save.")
