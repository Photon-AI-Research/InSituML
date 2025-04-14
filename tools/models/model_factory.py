import os, sys
import torch
import torch.optim as optim
from inSituML.utilities import MMD_multiscale, fit, load_checkpoint
from inSituML.ks_models import INNModel

from inSituML.args_transform import MAPPING_TO_LOSS
from inSituML.encoder_decoder import Encoder
from inSituML.encoder_decoder import Conv3DDecoder
from inSituML.loss_functions import EarthMoversLoss
from inSituML.networks import VAE
from models.architectures import ModelFinal

def get_world_size():
    """Get the world size for distributed training."""
    world_size = None
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        print(
            (
                "[WW] WORLD_SIZE not defined in env, "
                + "falling back to SLURM_NTASKS."
            ),
            file=sys.stderr,
        )
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        raise RuntimeError("cannot determine WORLD_SIZE")
    return world_size

def get_VAE_encoder_kwargs(io_config, model_config):
    """Create encoder kwargs dictionary from configs"""
    return {
        "ae_config": "non_deterministic",
        "z_dim": model_config.latent_space_dims,
        "input_dim": io_config.ps_dims,
        "conv_layer_config": [16, 32, 64, 128, 256, 608],
        "conv_add_bn": False,
        "fc_layer_config": [544],
    }

def get_VAE_decoder_kwargs(io_config, model_config):
    """Create decoder kwargs dictionary from configs"""
    return {
        "z_dim": model_config.latent_space_dims,
        "input_dim": io_config.ps_dims,
        "initial_conv3d_size": [16, 4, 4, 4],
        "add_batch_normalisation": False,
        "fc_layer_config": [1024],
    }

def load_objects(rank, io_config, model_config, world_size):
    """Load and initialize model objects, optimizer, and scheduler."""
    
    # Get configuration
    config = model_config.config
    
    # Get model parameters
    VAE_encoder_kwargs = get_VAE_encoder_kwargs(io_config, model_config)
    VAE_decoder_kwargs = get_VAE_decoder_kwargs(io_config, model_config)
    
    # Initialize loss function
    loss_fn_for_VAE = MAPPING_TO_LOSS[
        model_config.config["loss_function"]
    ](**model_config.config["loss_kwargs"])

    # Initialize VAE
    VAE_obj = VAE(
            encoder=Encoder,
            encoder_kwargs=VAE_encoder_kwargs,
            decoder=Conv3DDecoder,
            z_dim=model_config.latent_space_dims,
            decoder_kwargs=VAE_decoder_kwargs,
            loss_function=loss_fn_for_VAE,
            property_="momentum_force",
            particles_to_sample=io_config.number_of_particles,
            ae_config="non_deterministic",
            use_encoding_in_decoder=False,
            weight_kl=model_config.config["lambd_kl"],
            device=rank,
    )

    # Initialize inner model
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
            device=rank,
        )

    # Initialize final model
    model = ModelFinal(
        VAE_obj,
        inner_model,
        EarthMoversLoss(),
        weight_AE=config["lambd_AE"],
        weight_IM=config["lambd_IM"],
    )

        # Load a pre-trained model
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    if config["load_model"] is not None:
        original_state_dict = torch.load(
            config["load_model"], map_location=map_location
        )
        # updated_state_dict = {key.replace('VAE.', 'base_network.'):
        # value for key, value in original_state_dict.items()}
        model.load_state_dict(original_state_dict)
        print("Loaded pre-trained model successfully", flush=True)

    elif config["load_model_checkpoint"] is not None:
        model, _, _, _, _, _ = load_checkpoint(
            config["load_model_checkpoint"],
            model,
            map_location=map_location,
        )
        print("Loaded model checkpoint successfully", flush=True)
    else:
        pass  # run with random init

    lr = config["lr"]
    bs_factor = (
        io_config.trainBatchBuffer_config["training_bs"] / 2 * world_size
    )
    lr = lr * config["lr_scaling"](bs_factor)
    print(
        "Scaling learning rate from {} to {} due to bs factor {}".format(
            config["lr"], lr, bs_factor
        ),
        flush=True,
        )
    
    optimizer = optim.Adam(
        [
            {
                "params": model.base_network.parameters(),
                "lr": lr * config["lrAEmult"],
            },
            {"params": model.inner_model.parameters()},
        ],  # model.parameters()
        lr=lr,
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    if ("lr_annealingRate" not in config) or config[
        "lr_annealingRate"
    ] is None:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=config["lr_annealingRate"]
        )

    return optimizer, scheduler, model