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
from sklearn.cluster import KMeans
import argparse

from ks_models import PC_MAF, INNModel

import FrEIA
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
from train_khi_AE_refactored.networks import ConvAutoencoder, VAE
        
def main():
    
    config = dict(
    l2_reg = 2e-5,
    y_noise_scale = 1e-1,
    zeros_noise_scale = 5e-2,
    lambd_predict = 3.,
    lambd_latent = 300.,
    lambd_rev = 400.,
    t0 = 900,
    t1 = 1001,
    timebatchsize = 4,
    particlebatchsize = 2,
    ndim_tot = 544,
    ndim_x = 544,
    ndim_y = 512,
    ndim_z = 32,
    num_coupling_layers = 4,
    latent_space_dims = 544,
    hidden_size = 256,
    dim_pool = 1,
    lr = 0.00001,
    num_epochs = 5,
    blacklist_boxes = None,
    val_boxes = [3,12,61,51],
    property_ = 'momentum_force',
    #network = 'convAE17',
    norm_method = 'mean_6d',
    particles_to_sample = 150000,
    activation = 'gelu',
    load_model = None, #'24k0zbm4/best_model_', 
    load_model_checkpoint = 'model_24211', #'model_6058',
    grad_clamp = 5.00,
    lambd_AE = 1.0,
    lambd_IM = 0.001,
    weight_kl = 0.001,
    rad_eps = 1e-9,
    network = 'INN_VAE',
    sim = "014",
    freeze_ae_weights = False,
    y_borders = {
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
    eval_timesteps = [900,950,1000],
    N_samples = 5,
    generate_plots = True,
    plot_directory_path = 'metrics/',
    #model_filepath_pattern = '/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/{}', 
    model_filepath_pattern = '/bigdata/hplsim/scratch/kelling/chamfers/slurm-6923925/{}', 
    mean_std_file_path = '/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/mean_std_{}/global_stats_{}_{}.npz',
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_{}/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_{}/{}.npy",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    emd_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, verbose=True, backend='auto')

    if config["property_"] == "all":
        point_dim = 9
    elif config["property_"] == "momentum_force":
        point_dim = 6
    else:
        point_dim = 3

    class ModelFinal(nn.Module):
            def __init__(self,
                        base_network,
                        inner_model,
                        loss_function_IM = None,
                        weight_AE=1.0,
                        weight_IM=1.0):
                super().__init__()

                self.base_network = base_network
                self.inner_model = inner_model
                self.loss_function_IM = loss_function_IM
                self.weight_AE = weight_AE
                self.weight_IM = weight_IM

            def forward(self, x, y):

                loss_AE,loss_ae_reconst,kl_loss, _, encoded = self.base_network(x)

                # Check if the inner model is an instance of INNModel
                if isinstance(self.inner_model, INNModel):
                    # Use the compute_losses function of INNModel
                    loss_IM, l_fit,l_latent,l_rev = self.inner_model.compute_losses(encoded, y)
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

            def reconstruct(self,x, y, num_samples = 1):

                if isinstance(self.inner_model, INNModel):
                    lat_z_pred = self.inner_model(x, y, rev = True)
                    y = self.base_network.decoder(lat_z_pred)
                else:
                    lat_z_pred = self.inner_model.sample_pointcloud(num_samples = num_samples, cond=y)
                    y = self.base_network.decoder(lat_z_pred)

                return y, lat_z_pred

    inner_model = INNModel(ndim_tot=config["ndim_tot"],
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
                    device = device)


    encoder_kwargs = {"ae_config":"non_deterministic",
                      "z_dim":config["latent_space_dims"],
                      "input_dim":point_dim,
                      "conv_layer_config":[16, 32, 64, 128, 256, 608],
                      "conv_add_bn": False, 
                      "fc_layer_config":[544]}

    decoder_kwargs = {"z_dim":config["latent_space_dims"],
                      "input_dim":point_dim,
                      "initial_conv3d_size":[16, 4, 4, 4],
                      "add_batch_normalisation":False,
                      "fc_layer_config":[1024]}


    VAE_obj = VAE(encoder = Encoder, 
              encoder_kwargs = encoder_kwargs, 
              decoder = Conv3DDecoder, 
              z_dim=config["latent_space_dims"],
              decoder_kwargs = decoder_kwargs,
              loss_function = emd_loss,
              property_="momentum_force",
              particles_to_sample = config["particles_to_sample"],
              ae_config="non_deterministic",
              use_encoding_in_decoder=False,
              weight_kl = config["weight_kl"],
              device = device)


    model = ModelFinal(VAE_obj,
                       inner_model,
                       EarthMoversLoss(),
                       weight_AE=config["lambd_AE"],
                       weight_IM=config["lambd_IM"])



    filepath = config["model_filepath_pattern"]


    if config["load_model"] is not None:
        original_state_dict = torch.load(filepath.format(config["load_model"]), map_location=device)
        # updated_state_dict = {key.replace('VAE.', 'base_network.'): value for key, value in original_state_dict.items()}
        model.load_state_dict(original_state_dict)
        print('Loaded pre-trained model successfully')

    elif config["load_model_checkpoint"] is not None:
        model, _, _, _, _, _ = load_checkpoint(filepath.format(config["load_model_checkpoint"]), model,map_location=device)
        print('Loaded model checkpoint successfully')
    else:
        pass # run with random init


    model.to(device)

    for param in model.parameters():
        param.requires_grad = False


    def particle_transformation(p_gt,normalise=True):

        if normalise==True:
            p_gt = [normalize_mean_6d(element, mean_std_file=config["mean_std_file_path"].format(config["sim"],config["t0"], config["t1"])) for element in p_gt]

        p_gt = [random_sample(element, sample_size=config["particles_to_sample"]) for element in p_gt]
        p_gt = torch.from_numpy(np.array(p_gt, dtype = np.float32))

        p_gt = filter_dims(p_gt, property_=config["property_"])

        return p_gt


    def radiation_transformation(r):

        # ampliudes in each direction
        amp_x = torch.abs(r[:, 0, :]).to(torch.float32)
        amp_y = torch.abs(r[:, 1, :]).to(torch.float32)
        amp_z = torch.abs(r[:, 2, :]).to(torch.float32)

        #spectra
        r = amp_x**2 + amp_y**2 + amp_z**2

        #log transformation
        r = torch.log(r+config["rad_eps"])

        return r

    def evaluate(p_gt,
                 p_gt_og,
                 r, 
                 model,
                 t_index,
                 gpu_index,
                 device,
                 N_samples,
                 generate_plots = False,
                 flow_type = 'flow'):

        # filename = 
        # plot_directory_path = config["plot_directory_path"]+config["load_model_checkpoint"]+'/'+config["sim"]+f"/data_gpu_{gpu_index}_tindex_{t_index}/"

        plot_directory_path = os.path.join(config["plot_directory_path"], 
                                       config["load_model_checkpoint"], 
                                       config["sim"], 
                                       flow_type,
                                       f"data_gpu_{gpu_index}_tindex_{t_index}")

        if not os.path.exists(plot_directory_path):
            os.makedirs(plot_directory_path)
        
        p_gt = p_gt[gpu_index]
        p_gt_og = p_gt_og[gpu_index]
        r = r[gpu_index]


        cond = r.reshape(1,-1).to(device)

        p_gt_clone = p_gt.clone()
        p_gt_clone = p_gt_clone.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            emd_losses_inn = []
            emd_losses_vae = []
            fwd_losses_inn = []
            latent_losses = []
            for _ in range(N_samples):
                pc_pr, lat_z_pred = model.reconstruct(p_gt_clone, cond)
                # print('pc_pr',pc_pr.shape)
                pc_pr_denorm = denormalize_mean_6d(normalized_array=pc_pr.squeeze(), mean_std_file=config["mean_std_file_path"].format(config["sim"],config["t0"], config["t1"]))
                # print('pc_pr_denorm',pc_pr_denorm.shape)
                emd_loss_inn_sam = emd_loss(pc_pr_denorm.unsqueeze(0), p_gt_og.unsqueeze(0).to(device))
                emd_losses_inn.append(emd_loss_inn_sam)


                _,_,_, pc_pr_ae, z_encoded = model.base_network(p_gt_clone)

                # print('pc_pr_ae',pc_pr_ae.shape)
                pc_pr_ae_denorm  = denormalize_mean_6d(pc_pr_ae.squeeze(), mean_std_file = config["mean_std_file_path"].format(config["sim"],config["t0"],config["t1"]))
                # print('pc_pr_ae_denorm',pc_pr_ae_denorm.shape)
                emd_loss_vae_sam = emd_loss(pc_pr_ae_denorm.unsqueeze(0), p_gt_og.unsqueeze(0).to(device))
                emd_losses_vae.append(emd_loss_vae_sam)

                # latent loss
                latent_loss_sam = fit(z_encoded.to(device),lat_z_pred.to(device))
                latent_losses.append(latent_loss_sam)

                # with torch.no_grad():
                output = model.inner_model(z_encoded)
                rad_pred = output[:, config["ndim_z"]:].squeeze()
                l_fit_sam = fit(r.to(device),rad_pred.to(device))
                fwd_losses_inn.append(l_fit_sam)


            emd_losses_inn_tensor = torch.tensor(emd_losses_inn)
            if flow_type == 'right_flow' or flow_type == 'left_flow':
                ####added
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

                if data_points_lowest_mean_cluster.shape[0]<2:
                    emd_losses_inn_mean = emd_losses_inn_tensor.mean()
                    emd_losses_inn_std = emd_losses_inn_tensor.std()
                else:
                    # Calculate the standard deviation of the cluster with the lowest mean
                    std_lowest_mean_cluster = data_points_lowest_mean_cluster.std()

                    emd_losses_inn_mean = torch.tensor(lowest_mean)
                    emd_losses_inn_std = std_lowest_mean_cluster

                if generate_plots == True:
                    # hist_path = plot_directory_path + flow_type
                    plot_losses_histogram(emd_losses_inn_tensor,
                                  histogram_bins=20,
                                  histogram_alpha=0.5, 
                                  plot_title='Histogram of INN-VAE reconstruction losses with Cluster Means',
                                  x_title='EMD Losses',
                                  t=t_index, gpu_box =gpu_index,
                                  cluster_means=cluster_means,
                                  flow_type = flow_type,
                                  loss_type = 'emd',
                                  save_path=plot_directory_path,
                                  show_plot=False)

            else:
                emd_losses_inn_mean = emd_losses_inn_tensor.mean()
                emd_losses_inn_std = emd_losses_inn_tensor.std()
                if generate_plots == True:
                    # hist_path = plot_directory_path + flow_type
                    plot_losses_histogram(emd_losses_inn_tensor,
                      histogram_bins=20,
                      histogram_alpha=0.5, 
                      plot_title='Histogram of INN-VAE reconstruction losses',
                      x_title='EMD Losses',
                      t=t_index, gpu_box =gpu_index,
                      cluster_means=None,
                      flow_type = flow_type,
                      loss_type = 'emd',
                      save_path=plot_directory_path,
                      show_plot=False)


            emd_losses_vae_tensor = torch.tensor(emd_losses_vae)
            emd_losses_vae_mean = emd_losses_vae_tensor.mean()
            emd_losses_vae_std = emd_losses_vae_tensor.std()

            fwd_losses_inn_tensor = torch.tensor(fwd_losses_inn)
            fwd_losses_inn_mean = fwd_losses_inn_tensor.mean()
            fwd_losses_inn_std = fwd_losses_inn_tensor.std()


            latent_losses_tensor = torch.tensor(latent_losses)
            latent_losses_mean = latent_losses_tensor.mean()
            latent_losses_std = latent_losses_tensor.std()

            if generate_plots == True:
                # hist_path = plot_directory_path + flow_type
                plot_losses_histogram(latent_losses_tensor,
                      histogram_bins=20,
                      histogram_alpha=0.5, 
                      plot_title='Histogram of latent losses',
                      x_title='L2 Losses',
                      t=t_index, gpu_box =gpu_index,
                      cluster_means=None,
                      flow_type = flow_type,
                      loss_type = 'latent',
                      save_path=plot_directory_path,
                      show_plot=False)

            #denormalise radiation
            # rad_pred = torch.exp(rad_pred) - config["rad_eps"]
            if generate_plots == True:
                plot_radiation(ground_truth_intensity = r, 
                               predicted_intensity = rad_pred,
                               t = t_index,
                               gpu_box = gpu_index,
                               path = plot_directory_path,
                               show_plot = False,
                               )


        p_gt= p_gt_og.cpu().numpy()
        pc_pr= pc_pr_denorm.squeeze().cpu().numpy()
        pc_pr_ae= pc_pr_ae_denorm.squeeze().cpu().numpy()

        # Visualization code for momentum
        px = p_gt[:, 0]  # Px component of momentum
        py = p_gt[:, 1]  # Py component of momentum
        pz = p_gt[:, 2]  # Pz component of momentum

        px_pr = pc_pr[:, 0]  # Px component of momentum
        py_pr = pc_pr[:, 1]  # Py component of momentum
        pz_pr = pc_pr[:, 2]  # Pz component of momentum

        px_pr_ae = pc_pr_ae[:, 0]  # Px component of momentum
        py_pr_ae = pc_pr_ae[:, 1]  # Py component of momentum
        pz_pr_ae = pc_pr_ae[:, 2]  # Pz component of momentum

        # Visualization code for force
        fx = p_gt[:, 3]  # Fx component of force
        fy = p_gt[:, 4]  # Fy component of force
        fz = p_gt[:, 5]  # Fz component of force

        fx_pr = pc_pr[:, 3]  # Fx component of force
        fy_pr = pc_pr[:, 4]  # Fy component of force
        fz_pr = pc_pr[:, 5]  # Fz component of force

        fx_pr_ae = pc_pr_ae[:, 3]  # Fx component of force
        fy_pr_ae = pc_pr_ae[:, 4]  # Fy component of force
        fz_pr_ae = pc_pr_ae[:, 5]  # Fz component of force

        if generate_plots == True:
            create_momentum_density_plots(px, py, pz,
                                  px_pr, py_pr, pz_pr,
                                  px_pr_ae, py_pr_ae, pz_pr_ae,     
                                  bins=100, t=t_index, gpu_box=gpu_index,
                                  #chamfers_loss = cd.item(),
                                  emd_loss_inn=emd_losses_inn_mean.item(),
                                  emd_loss_vae=emd_losses_vae_mean.item(),
                                  path = plot_directory_path,
                                  show_plot = False,
                                  #enable_wandb = enable_wandb,
                                 ) 

            create_force_density_plots(fx, fy, fz,
                               fx_pr, fy_pr, fz_pr,
                               fx_pr_ae, fy_pr_ae, fz_pr_ae,     
                               bins=100, t=t_index, gpu_box=gpu_index,
                               #chamfers_loss = cd.item(),
                               emd_loss_inn=emd_losses_inn_mean.item(),
                               emd_loss_vae=emd_losses_vae_mean.item(),
                               path = plot_directory_path,
                               show_plot = False,
                               #enable_wandb = enable_wandb,
                              )
            
        return emd_losses_inn_mean,emd_losses_inn_std,emd_losses_vae_mean,emd_losses_vae_std,fwd_losses_inn_mean,fwd_losses_inn_std,latent_losses_mean,latent_losses_std


    def evaluate_boxes(model, t_index, config, device,N_samples, boxes, flow_type='vortex'):
        # Initialize lists to store results
        boxes_emd_losses_inn_mean = []
        boxes_emd_losses_inn_std = []
        boxes_emd_losses_vae_mean = []
        boxes_emd_losses_vae_std = []
        boxes_fwd_losses_inn_mean = []
        boxes_fwd_losses_inn_std = []

        # pathpattern1 = config["pathpattern1"]
        # pathpattern2 = config["pathpattern2"]

        # raw p_gt
        p_gt_all = np.load(config["pathpattern1"].format(config["sim"],t_index),allow_pickle = True)

        # normalised transformed p_gt
        p_gt = particle_transformation(p_gt_all,normalise=True)

        # unnormalised transformed p_gt
        p_gt_og = particle_transformation(p_gt_all,normalise=False)

        # Load radiation data
        r = torch.from_numpy(np.load(config["pathpattern2"].format(config["sim"],t_index)).astype(np.cfloat))

        r = radiation_transformation(r)

        # Iterate over each GPU index
        for gpu_index in boxes:
            print('gpu_index', gpu_index)
            # Call the evaluate function
            results = evaluate(p_gt, p_gt_og, r, model, t_index,gpu_index, device, N_samples, generate_plots=config["generate_plots"], flow_type=flow_type)

            # Unpack results
            emd_losses_inn_mean, emd_losses_inn_std, emd_losses_vae_mean, emd_losses_vae_std, fwd_losses_inn_mean, fwd_losses_inn_std,latent_losses_mean,latent_losses_std = results

            # Append results to lists
            boxes_emd_losses_inn_mean.append(emd_losses_inn_mean.numpy())
            boxes_emd_losses_inn_std.append(emd_losses_inn_std.numpy())
            boxes_emd_losses_vae_mean.append(emd_losses_vae_mean.numpy())
            boxes_emd_losses_vae_std.append(emd_losses_vae_std.numpy())
            boxes_fwd_losses_inn_mean.append(fwd_losses_inn_mean.numpy())
            boxes_fwd_losses_inn_std.append(fwd_losses_inn_std.numpy())
        
        # Find the minimum value
        min_box_emd_loss_inn = min(boxes_emd_losses_inn_mean)

        # Find the index of the minimum value
        min_index_emd_loss_inn = boxes_emd_losses_inn_mean.index(min_box_emd_loss_inn)
        
        print('box with min emd loss:',boxes[min_index_emd_loss_inn])
        print('min emd loss inn:',min_box_emd_loss_inn)
        
        # Calculate overall means and standard deviations
        overall_means_and_stds = {
            'min_box_emd_loss_inn': min_box_emd_loss_inn,
            'min_index_emd_loss_inn': boxes[min_index_emd_loss_inn],
            'emd_losses_inn_mean_all': sum(boxes_emd_losses_inn_mean) / len(boxes_emd_losses_inn_mean),
            'emd_losses_inn_std_all': sum(boxes_emd_losses_inn_std) / len(boxes_emd_losses_inn_std),
            'emd_losses_vae_mean_all': sum(boxes_emd_losses_vae_mean) / len(boxes_emd_losses_vae_mean),
            'emd_losses_vae_std_all': sum(boxes_emd_losses_vae_std) / len(boxes_emd_losses_vae_std),
            'fwd_losses_inn_mean_all': sum(boxes_fwd_losses_inn_mean) / len(boxes_fwd_losses_inn_mean),
            'fwd_losses_inn_std_all': sum(boxes_fwd_losses_inn_std) / len(boxes_fwd_losses_inn_std)
        }

        return overall_means_and_stds

    def evaluate_across_timesteps(timesteps, config, model):
        lower_vortex_boxes_data_across_timesteps = []
        upper_vortex_boxes_data_across_timesteps = []
        left_flow_boxes_data_across_timesteps = []
        right_flow_boxes_data_across_timesteps = []

        for t_index in timesteps:
            print('t_index:', t_index)
            left_flow_boxes_data = evaluate_boxes(model, t_index, config, config["device"], config["N_samples"], boxes = config["left_flow_boxes"], flow_type='left_flow')
            right_flow_boxes_data = evaluate_boxes(model, t_index, config, config["device"], config["N_samples"], boxes = config["right_flow_boxes"], flow_type='right_flow')
            lower_vortex_boxes_data = evaluate_boxes(model, t_index, config, config["device"], config["N_samples"], boxes = config["lower_vortex_boxes"], flow_type='lower_vortex')
            upper_vortex_boxes_data = evaluate_boxes(model, t_index, config, config["device"], config["N_samples"], boxes = config["upper_vortex_boxes"], flow_type='upper_vortex')

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

        y_lower_min, y_lower_max, y_upper_min, y_upper_max = config["y_borders"].get(config["sim"])

        p_gt_all = np.load(config["pathpattern1"].format(config["sim"],config["t0"]),allow_pickle = True)

        lower_vortex_boxes = []
        upper_vortex_boxes = []
        right_flow_boxes = []
        left_flow_boxes = []
        for box_id, p_box in enumerate(p_gt_all):
            # print('p_box', p_box.shape)
            if np.all((p_box[:,1] > y_lower_min) & (p_box[:,1] < y_lower_max)):
                lower_vortex_boxes.append(box_id)
            elif np.all((p_box[:,1] > y_upper_min) & (p_box[:,1] < y_upper_max)):
                upper_vortex_boxes.append(box_id)
            else:
                if p_box[:,3].mean()>0:
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
        metrics_path = os.path.join(config["plot_directory_path"], 
                                   config["load_model_checkpoint"], 
                                   config["sim"],
                                   'results.npz')

        np.savez(metrics_path, **results_dict)
        print('Done')

        return results_dict
    
    results_dict = calculate_all_metrics(config, model)
    
    return results_dict

if __name__ == "__main__":
    results_dict = main()
    print(results_dict)
