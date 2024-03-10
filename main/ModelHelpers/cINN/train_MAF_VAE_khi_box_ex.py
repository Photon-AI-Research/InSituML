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
from networks import *
from model import model_MAF as model_MAF
from data_loaders import TrainLoader, ValidationFixedBoxLoader

from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
from train_khi_AE_refactored.networks import ConvAutoencoder, VAE



def generate_plots_maf(model, t_index,gpu_index,pathpattern1,pathpattern2, config,normalizer, device, epoch, pbatch, plot_every):
    
    p_gt = np.load(pathpattern1.format(t_index),allow_pickle = True)
    #gt = p_gt.copy()

    p_gt = [normalizer.normalize_data(element, method=config["norm_method"], timestep_index = t_index) for element in p_gt]

    p_gt = [random_sample(element, sample_size=config["particles_to_sample"]) for element in p_gt]
    # p_gt = np.array(p_gt, dtype = np.float32)
    p_gt = torch.from_numpy(np.array(p_gt, dtype = np.float32))

    p_gt = filter_dims(p_gt, property_=config["property_"])
    p_gt = p_gt[gpu_index]

    # Load radiation data
    r = torch.from_numpy(np.load(pathpattern2.format(t_index)).astype(np.cfloat))

    # ampliudes in each direction
    amp_x = torch.abs(r[:, 0, :]).to(torch.float32)
    amp_y = torch.abs(r[:, 1, :]).to(torch.float32)
    amp_z = torch.abs(r[:, 2, :]).to(torch.float32)

    #spectra
    r = amp_x**2 + amp_y**2 + amp_z**2

    r = r[gpu_index]

    #log transformation
    # r = torch.log(r+1)
    r = torch.log(r+config["rad_eps"])
        
    if epoch == 0 and pbatch+1 == plot_every:
        # print('pbatch', pbatch)
        plot_radiation(r, t=t_index, gpu_box= gpu_index,
                                  enable_wandb = True)
    
    cond = r.reshape(1,-1).to(device)

    p_gt_clone = p_gt.clone()
    p_gt_clone = p_gt_clone.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pc_pr, lat_z_pred = model.reconstruct_maf(1, cond)
        _, pc_pr_ae, z_encoded = model.VAE(p_gt_clone)
    
#     z_encoded = z_encoded.squeeze()  
#     lat_z_pred = lat_z_pred.squeeze()

#     contains_nan_lat_z_pred = torch.isnan(z_encoded).any()
#     contains_nan_lat_z_pred= torch.isnan(lat_z_pred).any()

#     # Only call the plotting function if there are no NaN values in either tensor
#     if not contains_nan_lat_z_pred and not contains_nan_lat_z_pred:
#         plot_1d_histograms(z_encoded, lat_z_pred, bins=50, t=t_index, gpu_box=gpu_index,
#                            enable_wandb=True)
#     else:
#         print("Cannot plot due to NaN values in the tensor(s).")
    
        
    #emd = emd_loss(pc_pr_decoded.contiguous(), p_gt.to(device).contiguous())

    p_gt= p_gt.cpu().numpy()
    pc_pr= pc_pr.squeeze().cpu().numpy()
    pc_pr_ae= pc_pr_ae.squeeze().cpu().numpy()
    
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

    create_momentum_density_plots(px, py, pz,
                          px_pr, py_pr, pz_pr,
                          px_pr_ae, py_pr_ae, pz_pr_ae,     
                          bins=100, t=t_index, gpu_box=gpu_index,
                          #chamfers_loss = cd.item(),
                          #emd_loss=emd.item(),
                          enable_wandb = True
                         ) 

    create_force_density_plots(fx, fy, fz,
                       fx_pr, fy_pr, fz_pr,
                       fx_pr_ae, fy_pr_ae, fz_pr_ae,     
                       bins=100, t=t_index, gpu_box=gpu_index,
                       #chamfers_loss = cd.item(),
                       #emd_loss=emd.item(),
                       enable_wandb = True
                      )
    
def validate_model(model, valid_data_loader, property_, device):
    model.eval()
    val_loss_avg = []
    
    with torch.no_grad():
        for idx in range(len(valid_data_loader)):
            timestep_index, validation_boxes, p, r = valid_data_loader[idx]
            p = filter_dims(p, property_)
            p = p.to(device)
            r = r.to(device)
            val_loss,_,_ = model(x=p,y=r)
            val_loss_avg.append(val_loss.mean().item())
    val_loss_overall_avg = sum(val_loss_avg) / len(val_loss_avg)
    return val_loss_overall_avg
        
if __name__ == "__main__":

    hyperparameter_defaults = dict(
    t0 = 1980,
    t1 = 2001,
    timebatchsize = 4,
    particlebatchsize = 2,
    hidden_size = 1024,
    dim_pool = 1,
    lr = 0.001,
    num_epochs = 20000,
    blacklist_boxes = None,
    val_boxes = [3,12,61,51],
    property_ = 'momentum_force',
    #network = 'convAE17',
    norm_method = 'mean_6d',
    particles_to_sample = 150000,
    dim_input = 1024,
    dim_condition = 512,
    num_coupling_layers = 2,
    hidden_size_maf = 512,
    latent_space_dims = 1024,
    num_blocks_mat = 2,
    activation = 'gelu',
    #load_ae_model = '1oux6p2o',
    grad_clamp = 5.00,
    weight_AE = 1.0,
    weight_IM = 0.00001,
    rad_eps = 1e-9,
    network = 'MAF_VAE',
    pathpattern1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
    pathpattern2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
    pathpattern_valid1 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_003/{}.npy",
    pathpattern_valid2 = "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_003/{}.npy"
    )

    # Specify which hyperparameters to include in the run name
    included_hparams = ['network', 'lr', 'hidden_size_maf','weight_AE','weight_IM']

    # Generate a name for the run
    def generate_run_name(hparams, included_keys):
        name_parts = [f"{key}={hparams[key]}" for key in included_keys if key in hparams]
        return ",".join(name_parts)

    run_name = generate_run_name(hyperparameter_defaults, included_hparams)
    
    print('New session...')
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, project="khi_public", name=run_name)
    start_epoch = 0
    min_valid_loss = np.inf

    # Access all hyperparameter values through wandb.config
    config = wandb.config

    pathpattern1 = config["pathpattern1"]
    pathpattern2 = config["pathpattern2"]
    pathpattern_valid1 = config["pathpattern_valid1"]
    pathpattern_valid2 = config["pathpattern_valid2"]

    # point_dim = 9 if config["property_"] == "all" else 3
    if config["property_"] == "all":
        point_dim = 9
    elif config["property_"] == "momentum_force":
        point_dim = 6
    else:
        point_dim = 3

    wandb.config.update({'point_dim': point_dim})

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
                    normalisation = normalizer,
                    norm_method = config["norm_method"])

    valid_data_loader = ValidationFixedBoxLoader(pathpattern1=pathpattern_valid1,
                                           pathpattern2=pathpattern_valid2,
                                           validation_boxes = config["val_boxes"],
                                           t0=config["t0"],
                                           t1=config["t1"],
                                           particles_to_sample= config["particles_to_sample"],
                                            load_radiation = True,
                                            normalisation = normalizer,
                                            norm_method = config["norm_method"])


    
    
    class ModelFinal(nn.Module):
        def __init__(self, 
                     VAE,
                     inner_model,
                     loss_function_IM = None,
                     weight_AE=1.0,
                     weight_IM=1.0):
            super().__init__()

            self.VAE = VAE
            self.inner_model = inner_model
            self.loss_function_IM = loss_function_IM
            self.weight_AE = weight_AE
            self.weight_IM = weight_IM

        def forward(self, x, y):

            loss_AE, _, encoded = self.VAE(x)
            loss_IM = self.inner_model(inputs=encoded, context=y)*self.weight_IM

            return loss_AE*self.weight_AE + loss_IM, loss_AE*self.weight_AE, loss_IM
        
        def reconstruct_vae_input(self, x):
            y = self.VAE.reconstruct_input(x)
            return y
        
        def reconstruct_maf(self, num_samples, cond):
            
            lat_z_pred = self.inner_model.sample_pointcloud(num_samples, cond).squeeze(0)
            
            y = self.VAE.decoder(lat_z_pred)
            
            return y, lat_z_pred

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    encoder_kwargs = {"ae_config":"non_deterministic",
                      "z_dim":config["latent_space_dims"],
                      "input_dim":point_dim,
                      "conv_layer_config":[16, 32, 64, 128, 256, 512],
                      "conv_add_bn": False, 
                      "fc_layer_config":[256]}

    decoder_kwargs = {"z_dim":config["latent_space_dims"],
                      "input_dim":point_dim,
                      "initial_conv3d_size":[16, 4, 4, 4],
                      "add_batch_normalisation":False}

    inner_model = (model_MAF.PC_MAF(dim_condition=config["dim_condition"],
                               dim_input=config["dim_input"],
                               num_coupling_layers=config["num_coupling_layers"],
                               hidden_size=config["hidden_size"],
                               device=device,
                               num_blocks_mat = config["num_blocks_mat"],
                               activation = config["activation"]
                             ))

    VAE = VAE(encoder = Encoder, 
              encoder_kwargs = encoder_kwargs, 
              decoder = Conv3DDecoder, 
              z_dim=config["latent_space_dims"],
              decoder_kwargs = decoder_kwargs,
              loss_function = EarthMoversLoss(),
              property_="momentum_force",
              particles_to_sample = config["particles_to_sample"],
              ae_config="non_deterministic",
              use_encoding_in_decoder=False)

    model = ModelFinal(VAE, inner_model, EarthMoversLoss())
    

    print(model)

    model.to(device)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    wandb.config.update({'total_params': total_params})



    # Set up loss function and optimizer
    # criterion = nn.MSELoss()
    # emd_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, verbose=True)
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
    plot_every = 500

    start_time = time.time()
    for i_epoch in range(start_epoch, config["num_epochs"]):
        print('i_epoch', i_epoch)
        model.train()
        loss_overall = []
        for timeBatchIndex in range(len(epoch)):
            loss_avg = []
            timebatch = epoch[timeBatchIndex]

            #batch_idx = 0
            start_timebatch = time.time()
            for particleBatchIndex in range(len(timebatch)):
                #batch_idx += 1
                batch_tot_idx +=1

                optimizer.zero_grad()
                phase_space, radiation = timebatch[particleBatchIndex]

                phase_space = filter_dims(phase_space, property_= config["property_"])

                # phase_space = phase_space.permute(0, 2, 1).to(device)
                phase_space = phase_space.to(device)
                
                loss, loss_AE, loss_IM = model(x=phase_space,y=radiation)
                
                loss_avg.append(loss.item())

                if (batch_tot_idx)%plot_every== 0:

                    t_index = config["t1"] - 1
                    gpu_index = 17,25,19
                    for gpu in gpu_index:
                        generate_plots_maf(model,t_index,gpu,pathpattern1,pathpattern2, config,normalizer, device, i_epoch,particleBatchIndex,plot_every)

                    val_loss_overall_avg = validate_model(model, valid_data_loader, config["property_"], device)
                    # Log batch loss to wandb
                    wandb.log({
                        "Batch": batch_tot_idx,
                        "loss_AE": loss_AE,
                        "loss_IM": loss_IM,
                        "Train batch Loss": loss.item(),
                        "Validation Loss": val_loss_overall_avg
                    })

                loss.backward()
                
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-config["grad_clamp"], config["grad_clamp"])
                        
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