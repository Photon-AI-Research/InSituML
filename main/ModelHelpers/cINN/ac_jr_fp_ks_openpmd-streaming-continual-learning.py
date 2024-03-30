"""
Main file/module to train ML model from PIConGPU openPMD data using streaming and threads.
"""
import time

from ac_train_batch_buffer import TrainBatchBuffer
from ac_consumer_trainer import ModelTrainer
from threading import Thread
import torch
import numpy as np
from time import sleep
from random import random
from queue import Queue

import sys
import os
from torch import optim
import torch.nn as nn

from utilities import MMD_multiscale, fit, load_checkpoint
from ks_models import PC_MAF, INNModel

from train_khi_AE_refactored.args_transform import MAPPING_TO_LOSS
from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Encoder
from train_khi_AE_refactored.encoder_decoder import Conv3DDecoder, MLPDecoder
from train_khi_AE_refactored.loss_functions import EarthMoversLoss
from train_khi_AE_refactored.networks import VAE, ConvAutoencoder
from wandb_logger import WandbLogger
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from dummy_openpmd_producer import DummyOpenPMDProducer

import pathlib
import importlib.util
import sys

print("Done importing modules.")


def main():

    parser = argparse.ArgumentParser(
        description="""For running openPMDproduction based trainings.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    script_path = str(pathlib.Path(__file__).parent.resolve())

    parser.add_argument('--runner',
                    type=str,
                    default=None,
                    help="Which runner type is in use: srun (frontier), mpirun, torchrun. Overrides seting in io_config.")

    parser.add_argument('--type_streamer',
                    type=str,
                    default=None,
                    help="Which type of streamer to produce: streaming, offline, dummy. Overrides seting in io_config.")

    parser.add_argument('--io_config',
                    type=str,
                    default=script_path + '/io_config.py',
                    help="IO/streaming/data/paths -related config")

    parser.add_argument('--model_config',
                    type=str,
                    default=script_path + '/model_config.py',
                    help="model config")

    args = parser.parse_args()

    ##TODO: if this file is to be used via import, the config paths or configs
    ## must be allowed to com from somewhere else than CLI args
    spec = importlib.util.spec_from_file_location("io_config", args.io_config)
    #global io_config
    io_config = importlib.util.module_from_spec(spec)
    sys.modules["io_config"] = io_config
    spec.loader.exec_module(io_config)

    spec = importlib.util.spec_from_file_location("model_config", args.model_config)
    #global model_config
    model_config = importlib.util.module_from_spec(spec)
    sys.modules["model_config"] = model_config
    spec.loader.exec_module(model_config)

    if args.runner is None:
        args.runner = io_config.runner

    if args.type_streamer is None:
        args.type_streamer = io_config.type_streamer

    if not "training_bs" in io_config.trainBatchBuffer_config:
        io_config.trainBatchBuffer_config["training_bs"] = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    openPMDBuffer = Queue(io_config.openPMD_queue_size) ## Buffer shared between openPMD data loader and model

    # nomraliztion values loaded from model_config, because they are related to the pre-trained model
    streamLoader_config = io_config.streamLoader_config
    streamLoader_config["normalization"] = model_config.normalization_values

    config = model_config.config

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        print("[WW] WORLD_SIZE not defined in env, falling back to SLURM_NTASKS.", file=sys.stderr)
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        raise RuntimeError("cannot determine WORLD_SIZE")

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


    VAE_encoder_kwargs = {"ae_config":"non_deterministic",
                    "z_dim":model_config.latent_space_dims,
                    "input_dim":io_config.ps_dims,
                    "conv_layer_config":[16, 32, 64, 128, 256, 608],
                    "conv_add_bn": False,
                    "fc_layer_config":[544]}

    VAE_decoder_kwargs = {"z_dim":model_config.latent_space_dims,
                    "input_dim":io_config.ps_dims,
                    "initial_conv3d_size":[16, 4, 4, 4],
                    "add_batch_normalisation":False,
                        "fc_layer_config":[1024]}
    def load_objects(rank):
        
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        loss_fn_for_VAE = MAPPING_TO_LOSS[model_config.config['loss_function']](**model_config.config['loss_kwargs'])

        VAE_obj = VAE(encoder = Encoder,
                encoder_kwargs = VAE_encoder_kwargs,
                decoder = Conv3DDecoder,
                z_dim=model_config.latent_space_dims,
                decoder_kwargs = VAE_decoder_kwargs,
                loss_function = loss_fn_for_VAE,
                property_="momentum_force",
                particles_to_sample = io_config.number_of_particles,
                ae_config="non_deterministic",
                use_encoding_in_decoder=False,
                weight_kl=model_config.config["lambd_kl"],
                device=rank)
        
        # conv_AE
    #     conv_AE_encoder_kwargs = {"ae_config":"simple",
    #                     "z_dim":model_config.latent_space_dims,
    #                     "input_dim":io_config.ps_dims,
    #                     "conv_layer_config":[16, 32, 64, 128, 256, 512],
    #                     "conv_add_bn": False}

    #     conv_AE_decoder_kwargs = {"z_dim":model_config.latent_space_dims,
    #                     "input_dim":io_config.ps_dims,
    #                     "add_batch_normalisation":False}

    #     conv_AE = ConvAutoencoder(encoder = Encoder,
    #                             encoder_kwargs = conv_AE_encoder_kwargs,
    #                             decoder = Conv3DDecoder,
    #                             decoder_kwargs = conv_AE_decoder_kwargs,
    #                             loss_function = EarthMoversLoss(),
    #                             )

        # MAF inner model (not used in final runs)
        # inner_model = PC_MAF(dim_condition=config["dim_condition"],
        #                         dim_input=config["dim_input"],
        #                         num_coupling_layers=config["num_coupling_layers"],
        #                         hidden_size=config["hidden_size"],
        #                         device=rank,
        #                         num_blocks_mat = config["num_blocks_mat"],
        #                         activation = config["activation"]
        #                         )

        # INN
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
                        device = rank)

        #model = ModelFinal(VAE_obj, inner_model, EarthMoversLoss())
        #model = ModelFinal(conv_AE, inner_model, EarthMoversLoss())
        model = ModelFinal(VAE_obj,
                           inner_model,
                           EarthMoversLoss(),
                           weight_AE=config["lambd_AE"],
                           weight_IM=config["lambd_IM"])


        #Load a pre-trained model
        # filepath = '/autofs/nccs-svm1_home1/ksteinig/src/InSituML/main/ModelHelpers/cINN/trained_models/{}/best_model_'
        # filepath = 'trained_models/{}/best_model_'
        filepath = io_config.modelPathPattern

        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        if config["load_model"] is not None:
            original_state_dict = torch.load(filepath.format(config["load_model"]), map_location=map_location)
            # updated_state_dict = {key.replace('VAE.', 'base_network.'): value for key, value in original_state_dict.items()}
            model.load_state_dict(original_state_dict)
            print('Loaded pre-trained model successfully')
        
        elif config["load_model_checkpoint"] is not None:
            model, _, _, _, _, _ = load_checkpoint(filepath.format(config["load_model_checkpoint"]), model,map_location=map_location)
            print('Loaded model checkpoint successfully')
        else:
            pass # run with random init

        lr = config["lr"]
        bs_factor = io_config.trainBatchBuffer_config["training_bs"] / 2 * world_size
        lr = lr * config["lr_scaling"](bs_factor)
        print("Skaling learning rate from {} to {} due to bs factor {}".format(config["lr"], lr, bs_factor))
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=config["betas"],
                             eps=config["eps"], weight_decay=config["weight_decay"])
        if ( "lr_annealingRate" not in config ) or config["lr_annealingRate"] is None:
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=config["lr_annealingRate"])

        return optimizer, scheduler, model

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def run_copies(rank=None, world_size=world_size, runner=None):
        
        if runner=="torchrun":
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            print(f"Start running basic DDP example on rank {rank}.")
            # create model and move it to GPU with id rank
            rank = rank % torch.cuda.device_count()

        elif runner=="mpirun":
            
            rank=int(os.environ['OMPI_COMM_WORLD_NODE_RANK'])
            
            global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            print("ranks", global_rank, rank)

            dist.init_process_group(backend='nccl',world_size=world_size, rank=global_rank)
            print(f'Initiated DDP GPU {rank}', flush=True)

        elif runner=="srun":
            
            rank = 0
            
            global_rank = int(os.environ['SLURM_PROCID'])
            
            dist.init_process_group(backend='nccl',world_size=world_size, rank=global_rank)
            print(f'Initiated DDP GPU {rank}, global_rank {global_rank}', flush=True)
        else:
            setup(rank, world_size)

        optimizer, scheduler, model = load_objects(rank)
        
        if args.type_streamer == 'streaming':
            
            from ks_transform_policies import AbsoluteSquare, BoxesAttributesParticles
            #from ks_producer_openPMD_streaming import StreamLoaderExceptionCatcher as StreamLoader
            from ks_producer_openPMD_streaming import StreamLoader

            from LoaderExceptionHandler import wrapLoaderWithExceptionHandler
            Loader = wrapLoaderWithExceptionHandler(StreamLoader)


            particleDataTransformationPolicy = BoxesAttributesParticles() #returns particle data of shape (local ranks, number_of_particles, ps_dims)
            #particleDataTransformationPolicy = ParticlesAttributes() #returns particle data of shape (number_of_particles, ps_dims)

            # radiationDataTransformationPolicy = PerpendicularAbsoluteAndPhase() #returns radiation data of shape (local ranks, frequencies)
            radiationDataTransformationPolicy = AbsoluteSquare() #returns radiation data of shape (local ranks, frequencies)
            #radiationDataTransformationPolicy = AbsoluteSquareSumRanks() # returns radiation data of shape (frequencies)

            timeBatchLoader = Loader(openPMDBuffer, 
                                        streamLoader_config,
                                        particleDataTransformationPolicy, radiationDataTransformationPolicy) ## Streaming ready
        elif args.type_streamer == 'offline':
            
            from ks_transform_policies import AbsoluteSquare, BoxesAttributesParticles
            from ks_producer_openPMD import RandomLoader

            from LoaderExceptionHandler import wrapLoaderWithExceptionHandler
            Loader = wrapLoaderWithExceptionHandler(RandomLoader)

            particleDataTransformationPolicy = BoxesAttributesParticles()
            radiationDataTransformationPolicy = AbsoluteSquare() #returns radiation data of shape (local ranks, frequencies)

            timeBatchLoader = Loader(openPMDBuffer, 
                                        streamLoader_config,
                                        particleDataTransformationPolicy, radiationDataTransformationPolicy) ## Streaming ready
        else:
            timeBatchLoader = DummyOpenPMDProducer(openPMDBuffer)


        if rank == 0:
            # print some parameters
            print("#Param streamLoader_config.amplitude_direction=", streamLoader_config["amplitude_direction"], flush=True)
            print("#Param streamLoader_config.pathpattern1=", streamLoader_config["pathpattern1"], flush=True)
            print("#Param config.load_model=", config["load_model"], flush=True)
            print("#Param config.load_model_checkpoint=", config["load_model_checkpoint"], flush=True)
            print("#Param config.loss_function=", config["loss_function"], flush=True)
            print("#Param type_streamer=", io_config.type_streamer, flush=True)
            print("#Param trainBatchBuffer_config.cl_mem_size=", io_config.trainBatchBuffer_config["cl_mem_size"], flush=True)
            print("#Param trainBatchBuffer_config.consume_size=", io_config.trainBatchBuffer_config["consume_size"], flush=True)
            print("#Param trainBatchBuffer_config.training_bs=", io_config.trainBatchBuffer_config["training_bs"], flush=True)
            print("#Param trainBatchBuffer_config.continual_bs=", io_config.trainBatchBuffer_config["continual_bs"], flush=True)
        
        #wandb_logger = WandbLogger(project="khi_public",args=config, entity='jeyhun')    
        trainBF = TrainBatchBuffer(openPMDBuffer, **io_config.trainBatchBuffer_config)
        modelTrainer = ModelTrainer(trainBF, model, optimizer, scheduler, gpu_id=rank, **io_config.modelTrainer_config, logger = None)

        ####################
        ## Start training ##
        ####################
        start_time = time.time()

        modelTrainer.start()
        timeBatchLoader.start()
        # tell the producer who is consuming, so it can check if the consumer died and terminate in this case
        timeBatchLoader.consumer_thread = modelTrainer


        modelTrainer.join()
        print("Join model trainer")
        #stdout.flush()

        #stdout.flush()
        timeBatchLoader.join()
        print("Join openPMD data loader")
        #stdout.flush()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total elapsed time: {elapsed_time:.6f} seconds")

    def run_demo(demo_fn, world_size):
        mp.spawn(demo_fn,
                args=(world_size,),
                nprocs=world_size,
                join=True)
        
    if args.runner not in ['torchrun', 'mpirun', 'srun']:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        #run_demo(run_copies, world_size)
    else:
        run_copies(runner=args.runner)

if __name__ == '__main__':
    main()
    
    
    

