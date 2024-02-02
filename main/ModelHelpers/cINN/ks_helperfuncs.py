"""
Functions for normalization, sampling, plotting, and saving checkpoints
"""
import matplotlib.pyplot as plt
from torch import no_grad, save, from_numpy, angle, abs, cat
from torch import float32 as torchfloat32
from numpy.random import choice
from numpy import linspace, array, load, cfloat, concatenate, float32

def normalize_columns(original_array):
    xyz_columns = original_array[:, :3]
    x_min, x_max = xyz_columns[:, 0].min(), xyz_columns[:, 0].max()
    y_min, y_max = xyz_columns[:, 1].min(), xyz_columns[:, 1].max()
    z_min, z_max = xyz_columns[:, 2].min(), xyz_columns[:, 2].max()

    xyz_columns[:, 0] = (xyz_columns[:, 0] - x_min) / (x_max - x_min)
    xyz_columns[:, 1] = (xyz_columns[:, 1] - y_min) / (y_max - y_min)
    xyz_columns[:, 2] = (xyz_columns[:, 2] - z_min) / (z_max - z_min)

    normalized_array = concatenate((xyz_columns, original_array[:, 3:]), axis=1)
    return normalized_array

def denormalize_columns(normalized_array, gt):
    
    x_min, x_max = gt[:, 0].min(), gt[:, 0].max()
    y_min, y_max = gt[:, 1].min(), gt[:, 1].max()
    z_min, z_max = gt[:, 2].min(), gt[:, 2].max()

    xyz_columns = normalized_array[:, :3]

    xyz_columns[:, 0] = xyz_columns[:, 0] * (x_max - x_min) + x_min
    xyz_columns[:, 1] = xyz_columns[:, 1] * (y_max - y_min) + y_min
    xyz_columns[:, 2] = xyz_columns[:, 2] * (z_max - z_min) + z_min

    denormalized_array = concatenate((xyz_columns, normalized_array[:, 3:]), axis=1)
    return denormalized_array

def sample_pointcloud(model, num_samples, cond):
    model.model.eval()
    with no_grad():
        pc_pr = (model.model.sample(num_samples, cond))
        
    return pc_pr

def random_sample(data, sample_size):
    
    # Check if the sample size is greater than the number of points in the data
    if sample_size > data.shape[0]:
        raise ValueError("Sample size exceeds the number of points in the data")

    random_indices = choice(data.shape[0], sample_size, replace=False)
    sampled_data = data[random_indices]

    return sampled_data

def save_checkpoint(model, optimizer, path, last_loss, min_valid_loss, epoch, wandb_run_id):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_loss': last_loss.item(),
            'epoch': epoch,
            'min_valid_loss': min_valid_loss,
            'wandb_run_id': wandb_run_id,
        }

        save(state, path + '/model_' + str(epoch))
        
        
def create_position_density_plots(x, y, z,
                                  x_pr, y_pr, z_pr,
                                  bins=100, t=1000, path=''):
    
    # Specify the number of bins for each axis
    bins_x = linspace(min(x), max(x), bins)
    bins_y = linspace(min(y), max(y), bins)
    bins_z = linspace(min(z), max(z), bins)
    
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
    
    plt.tight_layout()
    plt.show()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/density_plots_{}.png'.format(t))

def create_momentum_density_plots(px, py, pz,
                                  px_pr, py_pr, pz_pr,
                                  bins=100, t=1000, path=''):
    
    # Specify the number of bins for each axis
    bins_px = linspace(min(px), max(px), bins)
    bins_py = linspace(min(py), max(py), bins)
    bins_pz = linspace(min(pz), max(pz), bins)
    
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
    plt.show()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/momentum_density_plots_{}.png'.format(t))



def create_force_density_plots(fx, fy, fz,
                               fx_pr, fy_pr, fz_pr,
                               bins=100, t=1000, path=''):
    
    # Specify the number of bins for each axis
    bins_fx = linspace(min(fx), max(fx), bins)
    bins_fy = linspace(min(fy), max(fy), bins)
    bins_fz = linspace(min(fz), max(fz), bins)
    
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
    plt.show()
    
    # Save the plots as image files
    if path:
        plt.savefig(path + '/force_density_plots_{}.png'.format(t))
    

def inference(gpu_index,t_index):

    p_gt = load(hyperparameter_defaults["pathpattern1"].format(t_index),allow_pickle = True)

    p_gt = [random_sample(element, sample_size=10000) for element in p_gt]
    p_gt = array(p_gt, dtype = float32)

    p_rad = from_numpy(load(hyperparameter_defaults["pathpattern2"].format(t_index)).astype(cfloat))

    p_rad_x = p_rad[gpu_index,0,:]
    p_rad_y = p_rad[gpu_index,1,:]
    p_rad_z = p_rad[gpu_index,2,:]

    p_rad = p_rad[:, 1:, :]
    p_rad = p_rad.view(p_rad.shape[0], -1)
    p_rad = p_rad.unsqueeze(1)

    p_rad = p_rad[gpu_index,:,:]
    p_gt = p_gt[gpu_index,:,:]

    # Compute the phase (angle) of the complex number
    phase = angle(p_rad)

    # Compute the amplitude (magnitude) of the complex number
    amplitude = abs(p_rad)
    p_rad = cat((amplitude, phase), dim=1).to(torchfloat32)

    num_samples = 1
    cond = p_rad.cuda()

    pc_pr =  sample_pointcloud(model, num_samples, cond)

    pc_pr = pc_pr.squeeze().cpu().numpy()

    pc_pr = pc_pr.reshape(10000,9)

    pc_pr = denormalize_columns(pc_pr, p_gt)

    x = p_gt[:, 0]  # X coordinates
    y = p_gt[:, 1]  # Y coordinates
    z = p_gt[:, 2]  # Z coordinates

    px = p_gt[:, 3]  # Px component of momentum
    py = p_gt[:, 4]  # Py component of momentum
    pz = p_gt[:, 5]  # Pz component of momentum

    fx = p_gt[:, 6]  # Fx component of force
    fy = p_gt[:, 7]  # Fy component of force
    fz = p_gt[:, 8]  # Fz component of force


    x_pr = pc_pr[:, 0]  # X coordinates
    y_pr = pc_pr[:, 1]  # Y coordinates
    z_pr = pc_pr[:, 2]  # Z coordinates

    px_pr = pc_pr[:, 3]  # Px component of momentum
    py_pr = pc_pr[:, 4]  # Py component of momentum
    pz_pr = pc_pr[:, 5]  # Pz component of momentum

    fx_pr = pc_pr[:, 6]  # Fx component of force
    fy_pr = pc_pr[:, 7]  # Fy component of force
    fz_pr = pc_pr[:, 8]  # Fz component of force


    create_position_density_plots(x, y, z, x_pr, y_pr, z_pr, bins=100, t=t_index)

    create_momentum_density_plots(px, py, pz, px_pr, py_pr, pz_pr, bins=100, t=t_index)

    create_force_density_plots(fx, fy, fz, fx_pr, fy_pr, fz_pr, bins=100, t=t_index)
      
