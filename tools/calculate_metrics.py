import os
import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from queue import Queue
from inSituML.utilities import (
    fit,
    MMD_multiscale,
    load_checkpoint,
    normalize_mean_6d,
    random_sample,
    filter_dims,
    denormalize_mean_6d,
    generate_momentum_force_radiation_plots,
    plot_losses_histogram
)
from sklearn.cluster import KMeans
from scipy.special import erfinv
import argparse
from inSituML.ks_models import INNModel
from inSituML.encoder_decoder import Encoder
from inSituML.encoder_decoder import Conv3DDecoder
from inSituML.loss_functions import EarthMoversLoss, ChamfersLoss
from inSituML.networks import VAE
from inSituML.ks_producer_openPMD_streaming import StreamLoader
from inSituML.ks_transform_policies import AbsoluteSquare, BoxesAttributesParticles


def main():

    parser = argparse.ArgumentParser(description="Update config settings for the script.")

    parser.add_argument(
        "--sim",
        type=str,
        help="Simulation to evaluate on.",
        default="014"
    )
    parser.add_argument(
        "--N_samples",
        type=int,
        help="Number model passes for loss calculation.",
        default=5
    )
    parser.add_argument(
        "--eval_timesteps",
        nargs='+',
        type=int,
        help="Timesteps to evaluate on.",
        default=[900, 950, 1000]
    )
    parser.add_argument(
        "--generate_plots",
        action=argparse.BooleanOptionalAction,
        help="Whether to generate all plots",
        default=True
    )
    parser.add_argument(
        "--generate_best_box_plot",
        action=argparse.BooleanOptionalAction,
        help="Whether to generate best box plot",
        default=True
    )
    parser.add_argument(
        "--plot_directory_path",
        type=str,
        help="Directory for saving plots.",
        default="metrics/"
    )
    parser.add_argument(
        "--load_model_checkpoint",
        type=str,
        help="Load model checkpoint",
        default="model_24211"
    )
    parser.add_argument(
        "--particle_pathpattern",
        type=str,
        help="Specify partilce path",
        default="/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
                "04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/"
                "8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1645/simOutput/openPMD/simData_%T.bp5"
    )
    parser.add_argument(
        "--radiation_pathpattern",
        type=str,
        help="Specify radiation path",
        default="/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
                "04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/"
                "8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1645/simOutput/streamedRadiation/ts_{}.npy"
    )

    args = parser.parse_args()

    openPMDBuffer = Queue(8)
    normalization_values = dict(
        momentum_mean=1.2091940752668797e-08,
        momentum_std=0.11923234769525472,
        force_mean=-2.7682006649827533e-09,
        force_std=7.705477610810592e-05
    )

    streamLoader_config = dict(
        sim_t0=900,
        t0=900,
        t1=1001,
        streaming_config=None,
        # particle_pathpattern = "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
        # "24-nodes_full-picongpu-data/04-01_1013/simOutput/openPMD/simData_%T.bp5",
        # radiation_pathpattern = "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
        # "24-nodes_full-picongpu-data/04-01_1013/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5",
        particle_pathpattern=(
            "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
            "04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/"
            "8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1645/simOutput/openPMD/simData_%T.bp5"
        ),
        radiation_pathpattern=(
            "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
            "04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/"
            "8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1645/simOutput/streamedRadiation/ts_{}.npy"
        ),
        amplitude_direction=0,  # choose single direction along which the radiation signal
                                # is observed, max: N_observer-1, where N_observer is
                                # defined in PIConGPU's radiation plugin
        phase_space_variables=[
            "momentum", "force"
        ],  # allowed are "position", "momentum", and "force". If "force" is set,
            # "momentum" needs to be set too.
        number_particles_per_gpu=30000,
        verbose=False,
        # offline training params
        num_epochs=.01,  # .0625
        normalization=normalization_values
    )

    config = dict(
        y_noise_scale=1e-1,
        zeros_noise_scale=5e-2,
        lambd_predict=3.,
        lambd_latent=300.,
        lambd_rev=400.,
        t0=900,
        t1=1001,
        ndim_tot=544,
        ndim_x=544,
        ndim_y=512,
        ndim_z=32,
        num_coupling_layers=4,
        latent_space_dims=544,
        hidden_size=256,
        dim_pool=1,
        property_='momentum_force',
        particles_to_sample=150000,
        activation='gelu',
        load_model=None,  # '24k0zbm4/best_model_',
        load_model_checkpoint='/bigdata/hplsim/scratch/kelling/chamfers/slurm-6923925/model_24211',
        lambd_AE=1.0,
        lambd_IM=0.001,
        weight_kl=0.001,
        rad_eps=1e-9,
        network='INN_VAE',
        sim="014",
        y_borders={
            "002": [96, 160, 352, 416],
            "003": [96, 160, 352, 416],
            "004": [96, 160, 352, 416],
            "007": [32, 96, 160, 224],
            "008": [64, 192, 320, 448],
            "009": [48, 72, 168, 192],
            "014": [32, 96, 160, 224],
            "015": [48, 72, 168, 192],
            "016": [32, 96, 160, 224],
        },
        eval_timesteps=[900, 950, 1000],
        N_samples=5,
        N_best_samples=10,
        fraction = 0.8,
        generate_plots=False,
        generate_best_box_plot=True,
        radiation_transformation=True,
        plot_directory_path='metrics/',
        mean_std_file_path=normalization_values,
        streamLoader_config=streamLoader_config,
    )

    # Update config with values from command-line arguments
    config["sim"] = args.sim
    config["N_samples"] = args.N_samples
    config["eval_timesteps"] = args.eval_timesteps
    config["generate_best_box_plot"] = args.generate_best_box_plot
    config["plot_directory_path"] = args.plot_directory_path
    config["load_model_checkpoint"] = args.load_model_checkpoint
    config["generate_plots"] = args.generate_plots
    config["streamLoader_config"]["particle_pathpattern"] = args.particle_pathpattern
    config["streamLoader_config"]["radiation_pathpattern"] = args.radiation_pathpattern

    config["model_checkpoint"] = os.path.basename(config["load_model_checkpoint"])

    # Check if the file ends with .npy
    if config["streamLoader_config"]["radiation_pathpattern"].endswith('.npy'):
        print("File is a .npy file. Setting includeRadiation = False in StreamLoader.")
        config["includeRadiation"] = False
    else:
        config["includeRadiation"] = True

    timeBatchLoader = StreamLoader(
        openPMDBuffer,
        config["streamLoader_config"],
        BoxesAttributesParticles(),
        AbsoluteSquare(),
        includeRadiation=config["includeRadiation"],
        includeMetadata=True
    )

    timeBatchLoader.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    emd_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, verbose=True, backend='auto')
    chamfers = ChamfersLoss()

    if config["property_"] == "all":
        point_dim = 9
    elif config["property_"] == "momentum_force":
        point_dim = 6
    else:
        point_dim = 3

    data = []
    while True:
        data.append(openPMDBuffer.get())
        if data[-1] is None:
            print("No more data to consume. Exiting.")
            break

    # filter data corresponding to evaluation timesteps
    data = [item for item in data if item is not None and item[2] in config["eval_timesteps"]]

    # denormalised data
    denorm_data = [
        denormalize_mean_6d(
            item[0].permute(0, 2, 1).reshape(-1, item[0].shape[1]),
            config["mean_std_file_path"]) for item in data
    ]

    N = len(data)  # Number of timesteps

    sum_values = torch.zeros(point_dim)
    total_count = 0

    for i in range(N):
        sum_values += denorm_data[i].sum(dim=0)
        total_count += denorm_data[i].size(0)

    # Compute the global mean
    mean_values = sum_values / total_count

    sum_squared_diff = torch.zeros(point_dim)

    for i in range(N):
        sum_squared_diff += ((denorm_data[i] - mean_values) ** 2).sum(dim=0)

    # Compute the global standard deviation
    std_values = torch.sqrt(sum_squared_diff / total_count)

    denorm_mean_std_dict = {
        "px": (mean_values[0].item(), std_values[0].item()),
        "py": (mean_values[1].item(), std_values[1].item()),
        "pz": (mean_values[2].item(), std_values[2].item()),
        "fx": (mean_values[3].item(), std_values[3].item()),
        "fy": (mean_values[4].item(), std_values[4].item()),
        "fz": (mean_values[5].item(), std_values[5].item())
    }

    def get_bounds(mean_std_dict, fraction=0.95):
        bounds_dict = {}

        for param, (mean, stddev) in mean_std_dict.items():

            # Compute n using the inverse error function
            n = np.sqrt(2) * erfinv(fraction)

            # Calculate bounds based on mean and standard deviation
            lower_bound = mean - n * stddev
            upper_bound = mean + n * stddev

            bounds_dict[param] = (lower_bound, upper_bound)

        return bounds_dict

    # bounds for denormalised plots
    config["denorm_max_min_dict"] = get_bounds(denorm_mean_std_dict, config["fraction"])

    def normalize(value, mean, std):
        return (value - mean) / std

    # Normalized bounds dictionary
    max_min_dict = {}

    for key, (min_val, max_val) in config["denorm_max_min_dict"].items():
        if key.startswith('p'):
            mean = config["mean_std_file_path"]['momentum_mean']
            std = config["mean_std_file_path"]['momentum_std']
        else:
            mean = config["mean_std_file_path"]['force_mean']
            std = config["mean_std_file_path"]['force_std']

        normalized_min = normalize(min_val, mean, std)
        normalized_max = normalize(max_val, mean, std)
        max_min_dict[key] = (normalized_min, normalized_max)

    # bounds for normalised plots
    config["max_min_dict"] = max_min_dict

    class ModelFinal(nn.Module):
        def __init__(
            self,
            base_network,
            inner_model,
            loss_function_IM=None,
            weight_AE=1.0,
            weight_IM=1.0,
        ):
            super().__init__()

            self.base_network = base_network
            self.inner_model = inner_model
            self.loss_function_IM = loss_function_IM
            self.weight_AE = weight_AE
            self.weight_IM = weight_IM

        def forward(self, x, y):

            loss_AE, loss_ae_reconst, kl_loss, _, encoded = self.base_network(x)

            # Check if the inner model is an instance of INNModel
            if isinstance(self.inner_model, INNModel):
                # Use the compute_losses function of INNModel
                loss_IM, l_fit, l_latent, l_rev = self.inner_model.compute_losses(encoded, y)
                total_loss = loss_AE*self.weight_AE + loss_IM*self.weight_IM

                losses = {
                    'total_loss': total_loss,
                    'loss_AE': loss_AE*self.weight_AE,
                    'loss_IM': loss_IM*self.weight_IM,
                    'loss_ae_reconst': loss_ae_reconst,
                    'kl_loss': kl_loss,
                    'l_fit': l_fit,
                    'l_latent': l_latent,
                    'l_rev': l_rev,
                        }

                return losses
            else:
                # For other types of models, such as MAF
                loss_IM = self.inner_model(inputs=encoded, context=y)
                total_loss = loss_AE*self.weight_AE + loss_IM * self.weight_IM

                losses = {
                    'total_loss': total_loss,
                    'loss_AE': loss_AE*self.weight_AE,
                    'loss_IM': loss_IM*self.weight_IM,
                    'loss_ae_reconst': loss_ae_reconst,
                    'kl_loss': kl_loss
                        }

                return losses

        def reconstruct(self, x, y, num_samples=1):

            if isinstance(self.inner_model, INNModel):
                lat_z_pred = self.inner_model(x, y, rev=True)
                y = self.base_network.decoder(lat_z_pred)
            else:
                lat_z_pred = self.inner_model.sample_pointcloud(num_samples=num_samples, cond=y)
                y = self.base_network.decoder(lat_z_pred)

            return y, lat_z_pred

    inner_model = INNModel(
        ndim_tot=config["ndim_tot"],
        ndim_x=config["ndim_x"],
        ndim_y=config["ndim_y"],
        ndim_z=config["ndim_z"],
        loss_fit=fit,
        loss_latent=MMD_multiscale,
        loss_backward=MMD_multiscale,
        lambd_predict=config["lambd_predict"],
        lambd_latent=config["lambd_latent"],
        lambd_rev=config["lambd_rev"],
        zeros_noise_scale=config["zeros_noise_scale"],
        y_noise_scale=config["y_noise_scale"],
        hidden_size=config["hidden_size"],
        activation=config["activation"],
        num_coupling_layers=config["num_coupling_layers"],
        device=device,
    )

    encoder_kwargs = {"ae_config": "non_deterministic",
                      "z_dim": config["latent_space_dims"],
                      "input_dim": point_dim,
                      "conv_layer_config": [16, 32, 64, 128, 256, 608],
                      "conv_add_bn": False,
                      "fc_layer_config": [544]}

    decoder_kwargs = {"z_dim": config["latent_space_dims"],
                      "input_dim": point_dim,
                      "initial_conv3d_size": [16, 4, 4, 4],
                      "add_batch_normalisation": False,
                      "fc_layer_config": [1024]}

    VAE_obj = VAE(
        encoder=Encoder,
        encoder_kwargs=encoder_kwargs,
        decoder=Conv3DDecoder,
        z_dim=config["latent_space_dims"],
        decoder_kwargs=decoder_kwargs,
        loss_function=emd_loss,
        property_="momentum_force",
        particles_to_sample=config["particles_to_sample"],
        ae_config="non_deterministic",
        use_encoding_in_decoder=False,
        weight_kl=config["weight_kl"],
        device=device,
    )

    model = ModelFinal(VAE_obj,
                       inner_model,
                       EarthMoversLoss(),
                       weight_AE=config["lambd_AE"],
                       weight_IM=config["lambd_IM"])

    if config["load_model"] is not None:
        original_state_dict = torch.load(config["load_model"], map_location=device)
        model.load_state_dict(original_state_dict)
        print('Loaded pre-trained model successfully')

    elif config["load_model_checkpoint"] is not None:
        model, _, _, _, _, _ = load_checkpoint(
            config["load_model_checkpoint"],
            model,
            map_location=device
        )
        print('Loaded model checkpoint successfully')
    else:
        pass  # run with random init

    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    def particle_transformation(p_gt, normalise=True):

        if normalise:
            p_gt = [
                normalize_mean_6d(
                    element,
                    mean_std_file=config["mean_std_file_path"]
                )
                for element in p_gt
            ]

        p_gt = [random_sample(element, sample_size=config["particles_to_sample"]) for element in p_gt]
        p_gt = torch.from_numpy(np.array(p_gt, dtype=np.float32))

        p_gt = filter_dims(p_gt, property_=config["property_"])

        return p_gt

    def ensure_path_exists(path):
        """Ensure that a directory exists. If it doesn't, create it."""
        if not os.path.exists(path):
            os.makedirs(path)

    def radiation_transformation(r):

        # ampliudes in each direction
        amp_x = torch.abs(r[:, 0, :]).to(torch.float32)
        amp_y = torch.abs(r[:, 1, :]).to(torch.float32)
        amp_z = torch.abs(r[:, 2, :]).to(torch.float32)

        # spectra
        r = amp_x**2 + amp_y**2 + amp_z**2

        # log transformation
        r = torch.log(r+config["rad_eps"])

        return r

    def cluster_and_extract_lowest_mean_data_points_and_mean(emd_losses_inn_tensor):

        # Use KMeans to cluster the data into two clusters
        kmeans = KMeans(n_clusters=2, random_state=0).fit(emd_losses_inn_tensor.reshape(-1, 1))

        # Compute the mean of each cluster
        cluster_means = kmeans.cluster_centers_.flatten()
        cluster_labels = kmeans.labels_

        # Identify the cluster with the lowest mean
        lowest_mean_index = np.argmin(cluster_means)
        lowest_mean = cluster_means[lowest_mean_index]

        # Extract the data points belonging to the cluster with the lowest mean
        data_points_lowest_mean_cluster = emd_losses_inn_tensor[cluster_labels == lowest_mean_index]

        return data_points_lowest_mean_cluster, lowest_mean, cluster_means

    def evaluate(
        p_gt,
        r,
        model,
        t_index,
        gpu_index,
        device,
        N_samples,
        generate_plots=False,
        generate_best_box_plot=False,
        flow_type='flow'
    ):

        plot_directory_path = os.path.join(
            config["plot_directory_path"],
            config["model_checkpoint"],
            config["sim"],
            flow_type,
            f"data_gpu_{gpu_index}_tindex_{t_index}")

        ensure_path_exists(plot_directory_path)

        p_gt = p_gt.permute(0, 2, 1)
        p_gt = p_gt[gpu_index]  # normalised p_gt torch.Size([150000, 6])
        # unnormalised
        p_gt_og = denormalize_mean_6d(
            normalized_array=p_gt,
            mean_std_file=config["mean_std_file_path"]
        )
        r = r[gpu_index]  # radiation

        cond = r.reshape(1, -1).to(device)

        p_gt_clone = p_gt.clone()
        p_gt_clone = p_gt_clone.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            emd_losses_inn = []
            emd_losses_vae = []

            chamfers_losses_inn = []
            chamfers_losses_vae = []

            fwd_losses_inn = []
            latent_losses = []
            emd_loss_inn_sam_min = float('inf')
            for ii in range(N_samples):
                pc_pr, lat_z_pred = model.reconstruct(p_gt_clone, cond)
                pc_pr_denorm = denormalize_mean_6d(
                    normalized_array=pc_pr.squeeze(),
                    mean_std_file=config["mean_std_file_path"]
                )
                # pc_pr_denorm torch.Size([4096, 6])
                # p_gt_og torch.Size([150000, 6])
                emd_loss_inn_sam = emd_loss(pc_pr_denorm.unsqueeze(0), p_gt_og.unsqueeze(0).to(device))
                emd_losses_inn.append(emd_loss_inn_sam)

                chamfers_loss_inn_sam = chamfers(pc_pr_denorm.unsqueeze(0), p_gt_og.unsqueeze(0).to(device))
                chamfers_losses_inn.append(chamfers_loss_inn_sam)

                _, _, _, pc_pr_ae, z_encoded = model.base_network(p_gt_clone)

                pc_pr_ae_denorm = denormalize_mean_6d(
                    pc_pr_ae.squeeze(),
                    mean_std_file=config["mean_std_file_path"]
                )
                emd_loss_vae_sam = emd_loss(pc_pr_ae_denorm.unsqueeze(0), p_gt_og.unsqueeze(0).to(device))
                emd_losses_vae.append(emd_loss_vae_sam)

                chamfers_loss_vae_sam = chamfers(pc_pr_ae_denorm.unsqueeze(0), p_gt_og.unsqueeze(0).to(device))
                chamfers_losses_vae.append(chamfers_loss_vae_sam)

                # latent loss
                latent_loss_sam = fit(z_encoded.to(device), lat_z_pred.to(device))
                latent_losses.append(latent_loss_sam)

                # with torch.no_grad():
                output = model.inner_model(z_encoded)
                rad_pred = output[:, config["ndim_z"]:].squeeze()
                l_fit_sam = fit(r.to(device), rad_pred.to(device))
                fwd_losses_inn.append(l_fit_sam)

                # find the lowest emd inn value
                if emd_loss_inn_sam < emd_loss_inn_sam_min and generate_best_box_plot:
                    emd_loss_inn_sam_min = emd_loss_inn_sam
                    norm_path = plot_directory_path + '/normalised_best_box'
                    denorm_path = plot_directory_path + '/denormalised_best_box'

                    ensure_path_exists(norm_path)
                    ensure_path_exists(denorm_path)

                    # normalised plots
                    emd_loss_inn_sam_norm = emd_loss(pc_pr.contiguous(), p_gt_clone.contiguous())
                    emd_loss_vae_sam_norm = emd_loss(pc_pr_ae.contiguous(), p_gt_clone.contiguous())
                    chamfers_loss_inn_sam_norm = chamfers(pc_pr, p_gt_clone)
                    chamfers_loss_vae_sam_norm = chamfers(pc_pr_ae, p_gt_clone)

                    generate_momentum_force_radiation_plots(
                        p_gt=p_gt.cpu().numpy(),
                        pc_pr=pc_pr.squeeze().cpu().numpy(),
                        pc_pr_ae=pc_pr_ae.squeeze().cpu().numpy(),
                        r=r,
                        rad_pred=rad_pred,
                        t_index=t_index,
                        gpu_index=gpu_index,
                        emd_losses_inn_mean=emd_loss_inn_sam_norm,
                        emd_losses_vae_mean=emd_loss_vae_sam_norm,
                        chamfers_losses_inn_mean=chamfers_loss_inn_sam_norm.item(),
                        chamfers_losses_vae_mean=chamfers_loss_vae_sam_norm.item(),
                        # chamfers_loss=chamfers_loss_vae_sam_norm.item(),
                        plot_directory_path=norm_path,
                        show_plot=False,
                        max_min_dict=config["max_min_dict"]
                    )

                    # denormalised plots
                    generate_momentum_force_radiation_plots(
                        p_gt=p_gt_og.cpu().numpy(),
                        pc_pr=pc_pr_denorm.squeeze().cpu().numpy(),
                        pc_pr_ae=pc_pr_ae_denorm.squeeze().cpu().numpy(),
                        r=r,
                        rad_pred=rad_pred,
                        t_index=t_index,
                        gpu_index=gpu_index,
                        emd_losses_inn_mean=emd_loss_inn_sam,
                        emd_losses_vae_mean=emd_loss_vae_sam,
                        chamfers_losses_inn_mean=chamfers_loss_inn_sam_norm.item(),
                        chamfers_losses_vae_mean=chamfers_loss_vae_sam_norm.item(),
                        # chamfers_loss=chamfers_loss_vae_sam.item(),
                        plot_directory_path=denorm_path,
                        show_plot=False,
                        max_min_dict=config["denorm_max_min_dict"]
                    )

            emd_losses_inn_tensor = torch.tensor(emd_losses_inn)
            chamfers_losses_inn_tensor = torch.tensor(chamfers_losses_inn)
            if flow_type == 'right_flow' or flow_type == 'left_flow':

                (
                    data_points_lowest_mean_cluster_emd_inn,
                    lowest_mean_emd_inn,
                    cluster_means_emd_inn
                ) = cluster_and_extract_lowest_mean_data_points_and_mean(
                    emd_losses_inn_tensor
                )

                (
                    data_points_lowest_mean_cluster_chamfers_inn,
                    lowest_mean_chamfers_inn,
                    cluster_means_chamfers_inn
                ) = cluster_and_extract_lowest_mean_data_points_and_mean(
                    chamfers_losses_inn_tensor
                )

                if data_points_lowest_mean_cluster_emd_inn.shape[0] < 2:
                    emd_losses_inn_mean = emd_losses_inn_tensor.mean()
                    emd_losses_inn_std = emd_losses_inn_tensor.std()
                else:
                    # Calculate the standard deviation of the cluster with the lowest mean
                    std_lowest_mean_cluster_emd_inn = data_points_lowest_mean_cluster_emd_inn.std()

                    emd_losses_inn_mean = torch.tensor(lowest_mean_emd_inn)
                    emd_losses_inn_std = std_lowest_mean_cluster_emd_inn

                if data_points_lowest_mean_cluster_chamfers_inn.shape[0] < 2:
                    chamfers_losses_inn_mean = chamfers_losses_inn_tensor.mean()
                    chamfers_losses_inn_std = chamfers_losses_inn_tensor.std()
                else:
                    # Calculate the standard deviation of the cluster with the lowest mean
                    std_lowest_mean_cluster_chamfers_inn = data_points_lowest_mean_cluster_chamfers_inn.std()

                    chamfers_losses_inn_mean = torch.tensor(lowest_mean_chamfers_inn)
                    chamfers_losses_inn_std = std_lowest_mean_cluster_chamfers_inn

                if generate_plots:
                    # hist_path = plot_directory_path + flow_type
                    plot_losses_histogram(
                        emd_losses_inn_tensor,
                        histogram_bins=20,
                        histogram_alpha=0.5,
                        plot_title=(
                            'Histogram of INN-VAE reconstruction losses with Cluster Means'
                        ),
                        x_title='EMD Losses',
                        t=t_index,
                        gpu_box=gpu_index,
                        cluster_means=cluster_means_emd_inn,
                        flow_type=flow_type,
                        loss_type='emd',
                        save_path=plot_directory_path,
                        show_plot=False
                    )

            else:  # Vortex type boxes
                emd_losses_inn_mean = emd_losses_inn_tensor.mean()
                emd_losses_inn_std = emd_losses_inn_tensor.std()

                chamfers_losses_inn_mean = chamfers_losses_inn_tensor.mean()
                chamfers_losses_inn_std = chamfers_losses_inn_tensor.std()

                if generate_plots:
                    # hist_path = plot_directory_path + flow_type
                    plot_losses_histogram(
                        emd_losses_inn_tensor,
                        histogram_bins=20,
                        histogram_alpha=0.5,
                        plot_title='Histogram of INN-VAE reconstruction losses',
                        x_title='EMD Losses',
                        t=t_index,
                        gpu_box=gpu_index,
                        cluster_means=None,
                        flow_type=flow_type,
                        loss_type='emd',
                        save_path=plot_directory_path,
                        show_plot=False,
                    )

            emd_losses_vae_tensor = torch.tensor(emd_losses_vae)
            emd_losses_vae_mean = emd_losses_vae_tensor.mean()
            emd_losses_vae_std = emd_losses_vae_tensor.std()

            chamfers_losses_vae_tensor = torch.tensor(chamfers_losses_vae)
            chamfers_losses_vae_mean = chamfers_losses_vae_tensor.mean()
            chamfers_losses_vae_std = chamfers_losses_vae_tensor.std()

            fwd_losses_inn_tensor = torch.tensor(fwd_losses_inn)
            fwd_losses_inn_mean = fwd_losses_inn_tensor.mean()
            fwd_losses_inn_std = fwd_losses_inn_tensor.std()

            latent_losses_tensor = torch.tensor(latent_losses)
            latent_losses_mean = latent_losses_tensor.mean()
            latent_losses_std = latent_losses_tensor.std()

            if generate_plots:
                # hist_path = plot_directory_path + flow_type
                plot_losses_histogram(
                    latent_losses_tensor,
                    histogram_bins=20,
                    histogram_alpha=0.5,
                    plot_title='Histogram of latent losses',
                    x_title='L2 Losses',
                    t=t_index,
                    gpu_box=gpu_index,
                    cluster_means=None,
                    flow_type=flow_type,
                    loss_type='latent',
                    save_path=plot_directory_path,
                    show_plot=False
                )

        p_gt = p_gt_og.cpu().numpy()
        pc_pr = pc_pr_denorm.squeeze().cpu().numpy()
        pc_pr_ae = pc_pr_ae_denorm.squeeze().cpu().numpy()

        if generate_plots:

            generate_momentum_force_radiation_plots(
                p_gt=p_gt,
                pc_pr=pc_pr,
                pc_pr_ae=pc_pr_ae,
                r=r,
                rad_pred=rad_pred,
                t_index=t_index,
                gpu_index=gpu_index,
                emd_losses_inn_mean=emd_losses_inn_mean,
                emd_losses_vae_mean=emd_losses_vae_mean,
                chamfers_losses_inn_mean=chamfers_losses_inn_mean.item(),
                chamfers_losses_vae_mean=chamfers_losses_vae_mean.item(),
                plot_directory_path=plot_directory_path,
                show_plot=False,
                max_min_dict=config["denorm_max_min_dict"],
            )

        results = {
            'emd_losses_inn_mean': emd_losses_inn_mean,
            'emd_losses_inn_std': emd_losses_inn_std,
            'emd_losses_vae_mean': emd_losses_vae_mean,
            'emd_losses_vae_std': emd_losses_vae_std,
            'fwd_losses_inn_mean': fwd_losses_inn_mean,
            'fwd_losses_inn_std': fwd_losses_inn_std,
            'latent_losses_mean': latent_losses_mean,
            'latent_losses_std': latent_losses_std,
            'chamfers_losses_inn_mean': chamfers_losses_inn_mean,
            'chamfers_losses_inn_std': chamfers_losses_inn_std,
            'chamfers_losses_vae_mean': chamfers_losses_vae_mean,
            'chamfers_losses_vae_std': chamfers_losses_vae_std,
        }

        return results

    def evaluate_boxes(model, ii, t_index, config, device, N_samples, boxes, flow_type='vortex'):
        # Initialize lists to store results
        boxes_emd_losses_inn_mean = []
        boxes_emd_losses_inn_std = []
        boxes_emd_losses_vae_mean = []
        boxes_emd_losses_vae_std = []
        boxes_fwd_losses_inn_mean = []
        boxes_fwd_losses_inn_std = []
        boxes_latent_losses_mean = []
        boxes_latent_losses_std = []
        boxes_chamfers_losses_inn_mean = []
        boxes_chamfers_losses_inn_std = []
        boxes_chamfers_losses_vae_mean = []
        boxes_chamfers_losses_vae_std = []

        p_gt = data[ii][0]

        if config["streamLoader_config"]["radiation_pathpattern"].endswith('.npy'):
            rad_t_index = config["streamLoader_config"]["t0"]-config["streamLoader_config"]["sim_t0"]+1
            r = np.load(config["streamLoader_config"]["radiation_pathpattern"].format(rad_t_index))
            r = torch.from_numpy(r).squeeze()
        else:
            r = data[ii][1]

        # Iterate over each GPU index
        for gpu_index in boxes:
            print('gpu_index', gpu_index)
            # Call the evaluate function
            results = evaluate(
                p_gt,
                r,
                model,
                t_index,
                gpu_index,
                device,
                N_samples,
                generate_plots=config["generate_plots"],
                generate_best_box_plot=False,
                flow_type=flow_type
            )

            # Unpack results
            emd_losses_inn_mean = results['emd_losses_inn_mean']
            emd_losses_inn_std = results['emd_losses_inn_std']

            emd_losses_vae_mean = results['emd_losses_vae_mean']
            emd_losses_vae_std = results['emd_losses_vae_std']

            fwd_losses_inn_mean = results['fwd_losses_inn_mean']
            fwd_losses_inn_std = results['fwd_losses_inn_std']

            latent_losses_mean = results['latent_losses_mean']
            latent_losses_std = results['latent_losses_std']

            chamfers_losses_inn_mean = results['chamfers_losses_inn_mean']
            chamfers_losses_inn_std = results['chamfers_losses_inn_std']

            chamfers_losses_vae_mean = results['chamfers_losses_vae_mean']
            chamfers_losses_vae_std = results['chamfers_losses_vae_std']

            # Append results to lists
            boxes_emd_losses_inn_mean.append(emd_losses_inn_mean.numpy())
            boxes_emd_losses_inn_std.append(emd_losses_inn_std.numpy())

            boxes_emd_losses_vae_mean.append(emd_losses_vae_mean.numpy())
            boxes_emd_losses_vae_std.append(emd_losses_vae_std.numpy())

            boxes_fwd_losses_inn_mean.append(fwd_losses_inn_mean.numpy())
            boxes_fwd_losses_inn_std.append(fwd_losses_inn_std.numpy())

            boxes_latent_losses_mean.append(latent_losses_mean.numpy())
            boxes_latent_losses_std.append(latent_losses_std.numpy())

            boxes_chamfers_losses_inn_mean.append(chamfers_losses_inn_mean.numpy())
            boxes_chamfers_losses_inn_std.append(chamfers_losses_inn_std.numpy())

            boxes_chamfers_losses_vae_mean.append(chamfers_losses_vae_mean.numpy())
            boxes_chamfers_losses_vae_std.append(chamfers_losses_vae_std.numpy())

        # Find the minimum value
        min_box_emd_loss_inn = min(boxes_emd_losses_inn_mean)

        # Find the index of the minimum value
        min_emd_loss_inn_index = boxes_emd_losses_inn_mean.index(min_box_emd_loss_inn)

        print('box with min emd loss:', boxes[min_emd_loss_inn_index])
        # print('min emd loss inn:',min_box_emd_loss_inn)

        # generate plots for the best box again
        best_results = evaluate(
            p_gt,
            r,
            model,
            t_index,
            boxes[min_emd_loss_inn_index],
            device,
            N_samples=config["N_best_samples"],
            generate_plots=config["generate_plots"],
            generate_best_box_plot=config["generate_best_box_plot"],
            flow_type=flow_type
        )

        # min_box_emd_loss_inn, _, _, _, _, _,_,_ = best_results
        min_box_emd_loss_inn_mean = best_results['emd_losses_inn_mean']
        min_box_emd_loss_inn_std = best_results['emd_losses_inn_std']
        print('min emd loss inn mean:', min_box_emd_loss_inn_mean)
        print('min emd loss inn std:', min_box_emd_loss_inn_mean)

        # Calculate overall means and standard deviations
        overall_means_and_stds = {
            'min_box_emd_loss_inn_mean': min_box_emd_loss_inn_mean,
            'min_box_emd_loss_inn_std': min_box_emd_loss_inn_std,
            'min_emd_loss_inn_index': boxes[min_emd_loss_inn_index],
            'emd_losses_inn_mean_all': sum(boxes_emd_losses_inn_mean) / len(boxes_emd_losses_inn_mean),
            'emd_losses_inn_std_all': sum(boxes_emd_losses_inn_std) / len(boxes_emd_losses_inn_std),
            'emd_losses_vae_mean_all': sum(boxes_emd_losses_vae_mean) / len(boxes_emd_losses_vae_mean),
            'emd_losses_vae_std_all': sum(boxes_emd_losses_vae_std) / len(boxes_emd_losses_vae_std),
            'fwd_losses_inn_mean_all': sum(boxes_fwd_losses_inn_mean) / len(boxes_fwd_losses_inn_mean),
            'fwd_losses_inn_std_all': sum(boxes_fwd_losses_inn_std) / len(boxes_fwd_losses_inn_std),
            'latent_losses_inn_mean_all': sum(boxes_latent_losses_mean) / len(boxes_latent_losses_mean),
            'latent_losses_inn_std_all': sum(boxes_latent_losses_std) / len(boxes_latent_losses_std),
            'chamfers_losses_inn_mean_all': sum(boxes_chamfers_losses_inn_mean) / len(boxes_chamfers_losses_inn_mean),
            'chamfers_losses_inn_std_all': sum(boxes_chamfers_losses_inn_std) / len(boxes_chamfers_losses_inn_std),
            'chamfers_losses_vae_mean_all': sum(boxes_chamfers_losses_vae_mean) / len(boxes_chamfers_losses_vae_mean),
            'chamfers_losses_vae_std_all': sum(boxes_chamfers_losses_vae_std) / len(boxes_chamfers_losses_vae_std),
        }

        return overall_means_and_stds

    def evaluate_across_timesteps(timesteps, config, model):
        lower_vortex_boxes_data_across_timesteps = []
        upper_vortex_boxes_data_across_timesteps = []
        left_flow_boxes_data_across_timesteps = []
        right_flow_boxes_data_across_timesteps = []

        for ii, t_index in enumerate(timesteps):
            print('t_index:', t_index)
            left_flow_boxes_data = evaluate_boxes(
                model,
                ii,
                t_index,
                config,
                config["device"],
                config["N_samples"],
                boxes=config["left_flow_boxes"],
                flow_type='left_flow'
            )
            right_flow_boxes_data = evaluate_boxes(
                model,
                ii,
                t_index,
                config,
                config["device"],
                config["N_samples"],
                boxes=config["right_flow_boxes"],
                flow_type='right_flow'
            )
            lower_vortex_boxes_data = evaluate_boxes(
                model,
                ii,
                t_index,
                config,
                config["device"],
                config["N_samples"],
                boxes=config["lower_vortex_boxes"],
                flow_type='lower_vortex'
            )
            upper_vortex_boxes_data = evaluate_boxes(
                model,
                ii,
                t_index,
                config,
                config["device"],
                config["N_samples"],
                boxes=config["upper_vortex_boxes"],
                flow_type='upper_vortex'
            )

            lower_vortex_boxes_data_across_timesteps.append(lower_vortex_boxes_data)
            upper_vortex_boxes_data_across_timesteps.append(upper_vortex_boxes_data)
            left_flow_boxes_data_across_timesteps.append(left_flow_boxes_data)
            right_flow_boxes_data_across_timesteps.append(right_flow_boxes_data)

        results_dict = {
            'timesteps': timesteps,
            'lower_vortex_boxes_data_t': lower_vortex_boxes_data_across_timesteps,
            'upper_vortex_boxes_data_t': upper_vortex_boxes_data_across_timesteps,
            'left_flow_boxes_data_t': left_flow_boxes_data_across_timesteps,
            'right_flow_boxes_data_t': right_flow_boxes_data_across_timesteps,
        }

        return results_dict

    def calculate_all_metrics(config, model):

        y_lower_min, y_lower_max, y_upper_min, y_upper_max = config["y_borders"].get(config["sim"], [32, 96, 160, 224])

        # particle data at t0
        p_gt_all = data[0][0]
        gpu_offset = data[0][3]
        gpu_extent = data[0][4]

        # Permute the dimensions to [gpu_box, num_of_particles, particles_dim]
        p_gt_all = p_gt_all.permute(0, 2, 1)

        lower_vortex_boxes = []
        upper_vortex_boxes = []
        right_flow_boxes = []
        left_flow_boxes = []
        for box_id, p_box in enumerate(p_gt_all):
            y_box_min = int(gpu_offset['y'][box_id])
            y_box_max = int(y_box_min + gpu_extent['y'][box_id])
            if y_box_min >= y_lower_min and y_box_max <= y_lower_max:
                lower_vortex_boxes.append(box_id)
            elif y_box_min >= y_upper_min and y_box_max <= y_upper_max:
                upper_vortex_boxes.append(box_id)
            else:
                if p_box[:, 0].mean() > 0:
                    right_flow_boxes.append(box_id)
                else:
                    left_flow_boxes.append(box_id)

        config['lower_vortex_boxes'] = lower_vortex_boxes
        config['upper_vortex_boxes'] = upper_vortex_boxes
        config['right_flow_boxes'] = right_flow_boxes
        config['left_flow_boxes'] = left_flow_boxes

        print('lower_vortex_boxes', lower_vortex_boxes)
        print('upper_vortex_boxes', upper_vortex_boxes)
        print('right_flow_boxes', right_flow_boxes)
        print('left_flow_boxes', left_flow_boxes)

        results_dict = evaluate_across_timesteps(config["eval_timesteps"], config, model)
        metrics_path = os.path.join(
            config["plot_directory_path"],
            config["model_checkpoint"],
            config["sim"],
            'results.npz',
        )

        np.savez(metrics_path, **results_dict)
        print('Done')

        return results_dict

    results_dict = calculate_all_metrics(config, model)

    return results_dict


if __name__ == "__main__":
    results_dict = main()
    print(results_dict)
