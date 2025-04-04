import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
import inspect
import wandb


def inspect_and_select(base):

    def decorator(**all_input_pars):

        input_vals = {k: all_input_pars[k]
                      for k, _ in inspect.signature(base).parameters.items()
                      if k in all_input_pars}

        return base(**input_vals)

    return decorator


def sample_gaussian(m, v, device):
    epsilon = torch.normal(torch.zeros(m.size()),
                           torch.ones(m.size())).to(device)
    z = m + torch.sqrt(v) * epsilon
    return z


def kl_normal(qm, qv, pm, pv):
    # checking how different is it from guassian distribution with
    # zero mean and 1 standard deviation.
    # tensor shape (Batch,dim)
    element_wise = 0.5 * (torch.log(pv) -
                          torch.log(qv) +
                          qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


# Losses
def MMD_multiscale(x, y):

    if dist.is_initialized():
        worldShape = [*x.shape]
        worldShape[0] *= dist.get_world_size()
        xall = torch.zeros(worldShape, dtype=x.dtype, device=x.device)
        xh = dist.all_gather_into_tensor(xall, x, async_op=True)
        yall = torch.zeros(worldShape, dtype=x.dtype, device=x.device)
        yh = dist.all_gather_into_tensor(yall, y, async_op=True)

        xh.wait()
        xx = torch.mm(xall, xall.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        dxx = rx.t() + rx - 2.0 * xx

        yh.wait()
        yy, zz = torch.mm(yall, yall.t()), torch.mm(xall, yall.t())
        ry = yy.diag().unsqueeze(0).expand_as(yy)
    else:
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx

    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros(xx.shape).to(x.device),
        torch.zeros(xx.shape).to(x.device),
        torch.zeros(xx.shape).to(x.device),
    )

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx) ** -1
        YY += a**2 * (a**2 + dyy) ** -1
        XY += a**2 * (a**2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY)


def fit(input, target):
    return torch.mean((input - target) ** 2)


def create_position_density_plots(
    x,
    y,
    z,
    x_pr,
    y_pr,
    z_pr,
    bins=100,
    t=1000,
    path="",
    enable_wandb=False,
    wandb=None,
):

    # Specify the number of bins for each axis
    bins_x = np.linspace(min(x), max(x), bins)
    bins_y = np.linspace(min(y), max(y), bins)
    bins_z = np.linspace(min(z), max(z), bins)

    # Create subplots for each plane
    plt.figure(figsize=(15, 10))

    # XY Plane Ground Truth
    plt.subplot(231)
    plt.hist2d(x, y, bins=[bins_x, bins_y], cmap="Blues")
    plt.colorbar(label="Density")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY Plane Ground Truth at t = {}".format(t))

    # XZ Plane Ground Truth
    plt.subplot(232)
    plt.hist2d(x, z, bins=[bins_x, bins_z], cmap="Greens")
    plt.colorbar(label="Density")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("XZ Plane Ground Truth at t = {}".format(t))

    # YZ Plane Ground Truth
    plt.subplot(233)
    plt.hist2d(y, z, bins=[bins_y, bins_z], cmap="Reds")
    plt.colorbar(label="Density")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("YZ Plane Ground Truth at t = {}".format(t))

    # XY Plane Prediction
    plt.subplot(234)
    plt.hist2d(x_pr, y_pr, bins=[bins_x, bins_y], cmap="Blues")
    plt.colorbar(label="Density")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY Plane Prediction at t = {}".format(t))

    # XZ Plane Prediction
    plt.subplot(235)
    plt.hist2d(x_pr, z_pr, bins=[bins_x, bins_z], cmap="Greens")
    plt.colorbar(label="Density")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("XZ Plane Prediction at t = {}".format(t))

    # YZ Plane Prediction
    plt.subplot(236)
    plt.hist2d(y_pr, z_pr, bins=[bins_y, bins_z], cmap="Reds")
    plt.colorbar(label="Density")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("YZ Plane Prediction at t = {}".format(t))

    plt.tight_layout()  # Adjust subplot spacing

    # Save the plots as image files
    if path:
        plt.savefig(path + "/density_plots_{}.png".format(t))

    if enable_wandb:
        # Log the overlapping histogram plot
        wandb.log({"XY vs XZ vs YZ histograms": wandb.Image(plt)})

        plt.close()
    else:
        plt.show()


def create_momentum_density_plots(
    px,
    py,
    pz,
    px_pr,
    py_pr,
    pz_pr,
    px_pr_ae,
    py_pr_ae,
    pz_pr_ae,
    chamfers_loss_inn=None,
    chamfers_loss_vae=None,
    emd_loss_inn=None,
    emd_loss_vae=None,
    bins=100, t=1000,
    gpu_box=0,
    path='',
    show_plot=True,
    enable_wandb=False,
    max_min_dict=None
):

    # Specify the number of bins for each axis
    # bins_px = np.linspace(min(px), max(px), bins)
    # bins_py = np.linspace(min(py), max(py), bins)
    # bins_pz = np.linspace(min(pz), max(pz), bins)
    bins_px = np.linspace(max_min_dict["px"][0], max_min_dict["px"][1], bins)
    bins_py = np.linspace(max_min_dict["py"][0], max_min_dict["py"][1], bins)
    bins_pz = np.linspace(max_min_dict["pz"][0], max_min_dict["pz"][1], bins)

    loss_info_inn = ''
    loss_info_vae = ''

    if chamfers_loss_vae is not None:
        loss_info_vae += '\nChamfers: {:.6f}'.format(chamfers_loss_vae)
    if chamfers_loss_inn is not None:
        loss_info_inn += '\nChamfers: {:.6f}'.format(chamfers_loss_inn)

    if emd_loss_inn is not None:
        loss_info_inn += '\nEMD: {:.6f}'.format(emd_loss_inn)
    if emd_loss_vae is not None:
        loss_info_vae += '\nEMD: {:.6f}'.format(emd_loss_vae)

    # Create subplots for each plane
    plt.figure(figsize=(15, 15))

    # px-py Plane Ground Truth
    plt.subplot(331)
    plt.hist2d(px, py, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('py')
    plt.title('px-py GT at t = {}, box = {}'.format(t, gpu_box))

    # px-pz Plane Ground Truth
    plt.subplot(332)
    plt.hist2d(px, pz, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px')
    plt.ylabel('pz')
    plt.title('px-pz GT at t = {}, box = {}'.format(t, gpu_box))

    # py-pz Plane Ground Truth
    plt.subplot(333)
    plt.hist2d(py, pz, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py')
    plt.ylabel('pz')
    plt.title('py-pz GT at t = {}, box = {}'.format(t, gpu_box))

    # px-py Plane Prediction
    plt.subplot(334)
    plt.hist2d(px_pr, py_pr, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('py_pr')
    plt.title('px-py INN Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_inn))

    # px-pz Plane Prediction
    plt.subplot(335)
    plt.hist2d(px_pr, pz_pr, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('pz_pr')
    plt.title('px-pz INN Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_inn))

    # py-pz Plane Prediction
    plt.subplot(336)
    plt.hist2d(py_pr, pz_pr, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py_pr')
    plt.ylabel('pz_pr')
    plt.title('py-pz INN Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_inn))

    # px-py Plane Prediction
    plt.subplot(337)
    plt.hist2d(px_pr_ae, py_pr_ae, bins=[bins_px, bins_py], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('py_pr')
    plt.title('px-py VAE Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_vae))

    # px-pz Plane Prediction
    plt.subplot(338)
    plt.hist2d(px_pr_ae, pz_pr_ae, bins=[bins_px, bins_pz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('px_pr')
    plt.ylabel('pz_pr')
    plt.title('px-pz VAE Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_vae))

    # py-pz Plane Prediction
    plt.subplot(339)
    plt.hist2d(py_pr_ae, pz_pr_ae, bins=[bins_py, bins_pz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('py_pr')
    plt.ylabel('pz_pr')
    plt.title('py-pz VAE Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_vae))

    plt.tight_layout()

    # Save the plots as image files
    if path:

        plt.savefig(path + '/momentum_density_plots_{}_{}.png'.format(t, gpu_box))

    if enable_wandb:
        # Log the overlapping histogram plot
        wandb.log({"Px vs Py vs Pz histograms (t={},box={})".format(t, gpu_box): wandb.Image(plt)})
        plt.close()
    elif show_plot:
        plt.show()
    else:
        plt.close()


def create_force_density_plots(
    fx,
    fy,
    fz,
    fx_pr,
    fy_pr,
    fz_pr,
    fx_pr_ae,
    fy_pr_ae,
    fz_pr_ae,
    chamfers_loss_inn=None,
    chamfers_loss_vae=None,
    emd_loss_inn=None,
    emd_loss_vae=None,
    bins=100,
    t=1000,
    gpu_box=0,
    path='',
    show_plot=True,
    enable_wandb=False,
    max_min_dict=None
):

    # Specify the number of bins for each axis
    # bins_fx = np.linspace(min(fx), max(fx), bins)
    # bins_fy = np.linspace(min(fy), max(fy), bins)
    # bins_fz = np.linspace(min(fz), max(fz), bins)
    bins_fx = np.linspace(max_min_dict["fx"][0], max_min_dict["fx"][1], bins)
    bins_fy = np.linspace(max_min_dict["fy"][0], max_min_dict["fy"][1], bins)
    bins_fz = np.linspace(max_min_dict["fz"][0], max_min_dict["fz"][1], bins)

    loss_info_inn = ''
    loss_info_vae = ''

    if chamfers_loss_vae is not None:
        loss_info_vae += '\nChamfers: {:.6f}'.format(chamfers_loss_vae)
    if chamfers_loss_inn is not None:
        loss_info_inn += '\nChamfers: {:.6f}'.format(chamfers_loss_inn)

    if emd_loss_inn is not None:
        loss_info_inn += '\nEMD: {:.6f}'.format(emd_loss_inn)
    if emd_loss_vae is not None:
        loss_info_vae += '\nEMD: {:.6f}'.format(emd_loss_vae)

    # Create subplots for each plane
    plt.figure(figsize=(15, 15))  # Adjust the figure size

    # fx-fy Plane Ground Truth
    plt.subplot(331)
    plt.hist2d(fx, fy, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fy')
    plt.title('fx-fy GT at t = {}, box = {}'.format(t, gpu_box))

    # fx-fz Plane Ground Truth
    plt.subplot(332)
    plt.hist2d(fx, fz, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx')
    plt.ylabel('fz')
    plt.title('fx-fz GT at t = {}, box = {}'.format(t, gpu_box))

    # fy-fz Plane Ground Truth
    plt.subplot(333)
    plt.hist2d(fy, fz, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy')
    plt.ylabel('fz')
    plt.title('fy-fz GT at t = {}, box = {}'.format(t, gpu_box))

    # fx-fy Plane Prediction
    plt.subplot(334)
    plt.hist2d(fx_pr, fy_pr, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fy_pr')
    plt.title('fx-fy INN Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_inn))

    # fx-fz Plane Prediction
    plt.subplot(335)
    plt.hist2d(fx_pr, fz_pr, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fz_pr')
    plt.title('fx-fz INN Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_inn))

    # fy-fz Plane Prediction
    plt.subplot(336)
    plt.hist2d(fy_pr, fz_pr, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy_pr')
    plt.ylabel('fz_pr')
    plt.title('fy-fz INN Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_inn))

    # fx-fy Plane Prediction
    plt.subplot(337)
    plt.hist2d(fx_pr_ae, fy_pr_ae, bins=[bins_fx, bins_fy], cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fy_pr')
    plt.title('fx-fy VAE Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_vae))

    # fx-fz Plane Prediction
    plt.subplot(338)
    plt.hist2d(fx_pr_ae, fz_pr_ae, bins=[bins_fx, bins_fz], cmap='Greens')
    plt.colorbar(label='Density')
    plt.xlabel('fx_pr')
    plt.ylabel('fz_pr')
    plt.title('fx-fz VAE Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_vae))

    # fy-fz Plane Prediction
    plt.subplot(339)
    plt.hist2d(fy_pr_ae, fz_pr_ae, bins=[bins_fy, bins_fz], cmap='Reds')
    plt.colorbar(label='Density')
    plt.xlabel('fy_pr')
    plt.ylabel('fz_pr')
    plt.title('fy-fz VAE Pred at t = {}, box = {}{}'.format(t, gpu_box, loss_info_vae))

    plt.tight_layout()

    # Save the plots as image files
    if path:

        plt.savefig(path + '/force_density_plots_{}_{}.png'.format(t, gpu_box))

    if enable_wandb:
        # Log the overlapping histogram plot
        wandb.log({"Fx vs Fy vs FZ histograms (t={},box={})".format(t, gpu_box): wandb.Image(plt)})

        plt.close()
    elif show_plot:
        plt.show()
    else:

        plt.close()


def random_sample(data, sample_size):
    # Check if the sample size is greater than the number of points in the data
    if sample_size > data.shape[0]:
        raise ValueError(
            "Sample size exceeds the number of points in the data"
        )

    # Randomly sample 'sample_size' points
    random_indices = np.random.choice(
        data.shape[0], sample_size, replace=False
    )
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

    normalized_array = np.concatenate(
        (xyz_columns, original_array[:, 3:]), axis=1
    )
    return normalized_array


def chamfersDist(a, b):
    d = torch.cdist(a, b, p=2)
    return torch.sum(torch.min(d, -1).values) + torch.sum(
        torch.min(d, -2).values
    )


def smooth_data(data, window_size=5):
    """Smooth data using a moving average."""
    return uniform_filter1d(data, size=window_size, mode="nearest")


def plot_radiation(
    ground_truth_intensity,
    predicted_intensity=None,
    frequency_range=512,
    t=1000,
    gpu_box=0,
    path='',
    show_plot=True,
    enable_wandb=False
):
    """
    Plot radiation intensity against frequency and
      compute MSE and relative MSE
    between ground truth and prediction.
    Compatible with both NumPy arrays and PyTorch tensors.

    Parameters:
    - ground_truth_intensity:
      A tensor or array of ground truth radiation spectra values.
    - predicted_intensity:
      A tensor or array of predicted radiation spectra values (optional).
    - t: Time step for the title (default=1000).
    - gpu_box: Identifier for the GPU box (default=0).
    - path: Path to save the plot (optional).
    - enable_wandb: Enable logging to Weights & Biases (default=False).
    """

    def to_numpy(data):
        """Convert PyTorch tensor to NumPy array if necessary."""
        if "torch" in str(type(data)):
            return data.cpu().numpy()
        return data

    # Load frequency data
    frequency = np.load(
        "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/omega.npy")[:frequency_range]

    # Ensure ground_truth_intensity and predicted_intensity are NumPy arrays
    ground_truth_intensity = to_numpy(ground_truth_intensity)[:frequency_range]

    plt.figure(figsize=(10, 6))
    plt.plot(
        frequency,
        ground_truth_intensity,
        label="GT Radiation Intensity (Raw)",
        color="blue",
        linewidth=2,
    )

    mse, rel_mse = 0, 0  # Initialize MSE and Relative MSE

    if predicted_intensity is not None:
        predicted_intensity = to_numpy(predicted_intensity)[:frequency_range]
        predicted_smoothed = smooth_data(predicted_intensity)

        plt.plot(
            frequency,
            predicted_smoothed,
            label='Predicted Radiation Intensity (Smoothed)',
            linestyle='--',
            color='red',
            marker='o',
            markersize=5,
            zorder=1
        )
        plt.plot(
            frequency,
            predicted_intensity,
            label='Predicted Radiation Intensity (Raw)',
            linestyle='--',
            color='red',
            alpha=0.3, zorder=0,
            markersize=3
        )
        plt.xscale('log')
        # Compute MSE
        mse = np.mean((ground_truth_intensity - predicted_intensity) ** 2)

        # Compute Relative MSE
        rel_mse = mse / np.mean(ground_truth_intensity**2)

    # Update plot title with MSE and Relative MSE if prediction is provided
    if predicted_intensity is not None:
        plt.title(
            f"Radiation Intensity vs. Frequency t = {t}," +
            f" box = {gpu_box}\nMSE = {mse:.2e}, Relative MSE = {rel_mse:.2e}"
        )
    else:
        plt.title(
            f"Radiation Intensity vs. Frequency t = {t}, box = {gpu_box}"
        )

    plt.xlabel("Frequency")
    plt.ylabel("Intensity (log scale)")
    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(f"{path}/radiation_plots_{t}_{gpu_box}.png")

    if enable_wandb:
        wandb.log(
            {"Radiation (t={},box={})".format(t, gpu_box): wandb.Image(plt)})
        plt.close()
    elif show_plot:
        plt.show()
    else:
        plt.close()


def save_checkpoint(
    model,
    optimizer,
    prefix,
    last_loss,
    iteration,
    min_valid_loss=None,
    wandb_run_id=None,
):
    print("save_checkpoint rank:", dist.get_rank())
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return  # only one rank should save

    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    state = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "last_loss": last_loss,
        "iteration": iteration,
    }

    if min_valid_loss is not None:
        state["min_valid_loss"] = min_valid_loss

    if wandb_run_id is not None:
        state["wandb_run_id"] = wandb_run_id

    torch.save(state, prefix + "model_" + str(iteration))


def load_checkpoint(
    path_to_checkpoint, model, optimizer=None, map_location=None
):
    # Load the saved file
    try:
        checkpoint = torch.load(path_to_checkpoint, map_location=map_location)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint file at {path_to_checkpoint} was not found."
        )

    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    last_loss = checkpoint.get("last_loss", None)
    iteration = checkpoint.get("iteration", None)
    min_valid_loss = checkpoint.get("min_valid_loss", None)
    wandb_run_id = checkpoint.get("wandb_run_id", None)

    return model, optimizer, last_loss, min_valid_loss, iteration, wandb_run_id


def save_checkpoint_conditionally(
    model,
    optimizer,
    prefix,
    iteration,
    last_loss,
    min_valid_loss=None,
    wandb_run_id=None,
):
    checkpoint_path = prefix + "model_" + str(iteration)
    checkpoint_dirname = os.path.dirname(checkpoint_path)
    if checkpoint_dirname and not os.path.exists(checkpoint_dirname):
        os.mkdir(checkpoint_dirname)

    # Check if the checkpoint for this iteration already exists
    if not os.path.exists(checkpoint_path):

        save_checkpoint(
            model,
            optimizer,
            prefix,
            last_loss,
            iteration,
            min_valid_loss,
            wandb_run_id,
        )
        print(f"Checkpoint for iteration {iteration} saved.")
    else:
        print(
            f"Checkpoint for iteration {iteration} already exists." +
            " Skipping save."
        )


def normalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Normalize point from a set of points with vmin(minimum) and vmax(maximum)
    to be in a range [a, b]
    '''

    # Extract the first three columns
    first_three_col = point[:, :3]

    # Perform operations on the first three columns
    modified_first_three_col = a + (first_three_col - vmin) * (b - a) / (vmax - vmin)

    # Combine the modified columns with the unchanged columns
    result_array = torch.cat((modified_first_three_col, point[:, 3:]), dim=1).to(point.dtype)

    return result_array


def denormalize_point(point_normalized, vmin, vmax, a=0., b=1.):
    '''
    Denormalize point back to the original range using vmin(minimum) and vmax(maximum).
    '''

    # Convert the input to PyTorch tensors
    # point_normalized = torch.tensor(point_normalized)
    vmin = torch.tensor(vmin)
    vmax = torch.tensor(vmax)

    # Extract the first three columns
    first_three_col_normalized = point_normalized[:, :3]

    # Perform operations on the first three columns to denormalize them
    denormalized_first_three_col = vmin + (first_three_col_normalized - a) * (vmax - vmin) / (b - a)

    # Combine the denormalized columns with the unchanged columns
    result_array = torch.cat((denormalized_first_three_col, point_normalized[:, 3:]), dim=1).to(point_normalized.dtype)

    return result_array


def filter_dims(phase_space, property_="positions"):

    if property_ == "positions":
        return phase_space[:, :, :3]
    elif property_ == "momentum":
        return phase_space[:, :, 3:6]
    elif property_ == "force":
        return phase_space[:, :, 6:]
    elif property_ == "momentum_force":
        return phase_space[:, :, 3:]
    else:
        return phase_space


def normalize_mean_6d(original_array, mean_std_file):
    # Check if mean_std_file is a dictionary
    if isinstance(mean_std_file, dict):
        data_stats = mean_std_file
        global_mean_momentum = data_stats['momentum_mean']
        global_mean_force = data_stats['force_mean']
        global_std_momentum = data_stats['momentum_std']
        global_std_force = data_stats['force_std']
    else:
        # Ensure mean_std_file is a path-like object
        if not isinstance(mean_std_file, (str)):
            raise TypeError("mean_std_file must be a string or a dictionary")

        # Check if mean_std_file is a valid file path
        if not os.path.isfile(mean_std_file):
            raise FileNotFoundError("Provided file path does not exist: " + str(mean_std_file))

        data_stats = np.load(mean_std_file)
        global_mean_momentum = data_stats['mean_momentum']
        global_mean_force = data_stats['mean_force']
        global_std_momentum = data_stats['std_momentum']
        global_std_force = data_stats['std_force']

    is_torch_tensor = torch.is_tensor(original_array)

    if is_torch_tensor:
        original_array = original_array.float()
        global_mean_momentum = torch.tensor(
            global_mean_momentum,
            dtype=original_array.dtype,
            device=original_array.device
        )
        global_mean_force = torch.tensor(
            global_mean_force,
            dtype=original_array.dtype,
            device=original_array.device
        )
        global_std_momentum = torch.tensor(
            global_std_momentum,
            dtype=original_array.dtype,
            device=original_array.device
        )
        global_std_force = torch.tensor(
            global_std_force,
            dtype=original_array.dtype,
            device=original_array.device)

        # Normalize x, y, z positions
        xyz_columns = original_array[:, :3]
        mins = torch.min(xyz_columns, dim=0).values
        maxs = torch.max(xyz_columns, dim=0).values
        xyz_columns_normalized = (xyz_columns - mins) / (maxs - mins)

        # Mean normalization for momentum dimensions
        momentum_columns_normalized = (original_array[:, 3:6] - global_mean_momentum) / global_std_momentum

        # Mean normalization for force dimensions
        force_columns_normalized = (original_array[:, 6:9] - global_mean_force) / global_std_force

        # Combine the normalized columns into one array
        normalized_array = torch.cat(
            (
                xyz_columns_normalized,
                momentum_columns_normalized,
                force_columns_normalized
            ),
            dim=1
        )
    else:
        # Normalize x, y, z positions
        xyz_columns = original_array[:, :3]
        mins = np.min(xyz_columns, axis=0)
        maxs = np.max(xyz_columns, axis=0)
        xyz_columns_normalized = (xyz_columns - mins) / (maxs - mins)

        # Mean normalization for momentum dimensions
        momentum_columns_normalized = (original_array[:, 3:6] - global_mean_momentum) / global_std_momentum

        # Mean normalization for force dimensions
        force_columns_normalized = (original_array[:, 6:9] - global_mean_force) / global_std_force

        # Combine the normalized columns into one array
        normalized_array = np.concatenate(
            (
                xyz_columns_normalized,
                momentum_columns_normalized,
                force_columns_normalized
            ),
            axis=1
        )

    return normalized_array


def denormalize_mean_6d(normalized_array, mean_std_file):

    # Check if mean_std_file is a dictionary
    if isinstance(mean_std_file, dict):
        data_stats = mean_std_file
        global_mean_momentum = data_stats['momentum_mean']
        global_mean_force = data_stats['force_mean']
        global_std_momentum = data_stats['momentum_std']
        global_std_force = data_stats['force_std']
    else:
        # Ensure mean_std_file is a path-like object
        if not isinstance(mean_std_file, (str)):
            raise TypeError("mean_std_file must be a string or a dictionary")

        # Check if mean_std_file is a valid file path
        if not os.path.isfile(mean_std_file):
            raise FileNotFoundError("Provided file path does not exist: " + str(mean_std_file))

        data_stats = np.load(mean_std_file)
        global_mean_momentum = data_stats['mean_momentum']
        global_mean_force = data_stats['mean_force']
        global_std_momentum = data_stats['std_momentum']
        global_std_force = data_stats['std_force']

    is_torch_tensor = torch.is_tensor(normalized_array)

    if is_torch_tensor:
        # Convert numpy arrays to PyTorch tensors if input is a tensor
        global_mean_momentum = torch.tensor(
            global_mean_momentum,
            dtype=normalized_array.dtype,
            device=normalized_array.device
        )
        global_mean_force = torch.tensor(
            global_mean_force,
            dtype=normalized_array.dtype,
            device=normalized_array.device
        )
        global_std_momentum = torch.tensor(
            global_std_momentum,
            dtype=normalized_array.dtype,
            device=normalized_array.device
        )
        global_std_force = torch.tensor(
            global_std_force,
            dtype=normalized_array.dtype,
            device=normalized_array.device
        )

    # Denormalize momentum dimensions
    momentum_columns_denormalized = normalized_array[:, :3] * global_std_momentum + global_mean_momentum

    # Denormalize force dimensions
    force_columns_denormalized = normalized_array[:, 3:6] * global_std_force + global_mean_force

    if is_torch_tensor:
        denormalized_array = torch.cat((momentum_columns_denormalized, force_columns_denormalized), dim=1)
    else:
        denormalized_array = np.concatenate((momentum_columns_denormalized, force_columns_denormalized), axis=1)

    return denormalized_array


def plot_losses_histogram(emd_losses,
                          histogram_bins=20,
                          histogram_alpha=0.5,
                          plot_title='Histogram of INN-VAE reconstruction losses with Cluster Means',
                          x_title='EMD Losses',
                          t=1000,
                          gpu_box=0,
                          loss_type=None,
                          cluster_means=None,
                          flow_type=None,
                          save_path=None,
                          show_plot=True):

    # Create a histogram for visualization
    plt.hist(emd_losses, bins=histogram_bins, alpha=histogram_alpha, label='Histogram of EMD losses')

    # Plot cluster means if provided
    if cluster_means is not None:
        plt.scatter(cluster_means, [0] * len(cluster_means), color='red', zorder=5, label='Cluster Means')

    plt.title(plot_title)
    plt.xlabel(x_title)
    plt.legend()

    if save_path:
        plt.savefig(save_path + '/Loss_histograms_{}_{}_{}_{}.png'.format(flow_type, loss_type, t, gpu_box))
        plt.close()
    elif show_plot:
        plt.show()
    else:
        plt.close()


def generate_momentum_force_radiation_plots(p_gt,
                                            pc_pr,
                                            pc_pr_ae,
                                            r,
                                            rad_pred,
                                            t_index,
                                            gpu_index,
                                            emd_losses_inn_mean,
                                            emd_losses_vae_mean,
                                            plot_directory_path,
                                            chamfers_losses_inn_mean=None,
                                            chamfers_losses_vae_mean=None,
                                            show_plot=False,
                                            max_min_dict=None):

    # Momentum components extraction
    px, py, pz = p_gt[:, 0], p_gt[:, 1], p_gt[:, 2]
    px_pr, py_pr, pz_pr = pc_pr[:, 0], pc_pr[:, 1], pc_pr[:, 2]
    px_pr_ae, py_pr_ae, pz_pr_ae = pc_pr_ae[:, 0], pc_pr_ae[:, 1], pc_pr_ae[:, 2]

    # Force components extraction
    fx, fy, fz = p_gt[:, 3], p_gt[:, 4], p_gt[:, 5]
    fx_pr, fy_pr, fz_pr = pc_pr[:, 3], pc_pr[:, 4], pc_pr[:, 5]
    fx_pr_ae, fy_pr_ae, fz_pr_ae = pc_pr_ae[:, 3], pc_pr_ae[:, 4], pc_pr_ae[:, 5]

    # Call to create momentum density plots
    create_momentum_density_plots(
        px,
        py,
        pz,
        px_pr,
        py_pr,
        pz_pr,
        px_pr_ae,
        py_pr_ae,
        pz_pr_ae,
        bins=100,
        t=t_index,
        gpu_box=gpu_index,
        emd_loss_inn=emd_losses_inn_mean.item(),
        emd_loss_vae=emd_losses_vae_mean.item(),
        path=plot_directory_path,
        show_plot=show_plot,
        # chamfers_loss=chamfers_loss,
        chamfers_loss_inn=chamfers_losses_inn_mean,
        chamfers_loss_vae=chamfers_losses_vae_mean,
        max_min_dict=max_min_dict
    )

    # Call to create force density plots
    create_force_density_plots(
        fx,
        fy,
        fz,
        fx_pr,
        fy_pr,
        fz_pr,
        fx_pr_ae,
        fy_pr_ae,
        fz_pr_ae,
        bins=100,
        t=t_index,
        gpu_box=gpu_index,
        emd_loss_inn=emd_losses_inn_mean.item(),
        emd_loss_vae=emd_losses_vae_mean.item(),
        path=plot_directory_path,
        show_plot=show_plot,
        chamfers_loss_inn=chamfers_losses_inn_mean,
        chamfers_loss_vae=chamfers_losses_vae_mean,
        max_min_dict=max_min_dict
    )

    # Call to plot radiation
    plot_radiation(
        ground_truth_intensity=r,
        predicted_intensity=rad_pred,
        t=t_index,
        gpu_box=gpu_index,
        path=plot_directory_path,
        show_plot=show_plot
    )
