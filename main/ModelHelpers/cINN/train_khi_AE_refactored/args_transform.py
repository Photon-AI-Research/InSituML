from networks import ConvAutoencoder, VAE
from encoder_decoder import Encoder, MLPDecoder, Conv3DDecoder
from loss_functions import EarthMoversLoss, ChamfersLoss, ChamfersLossDiagonal, ChamfersLossOptimized
import ast
import torch.nn as nn


MAPPING_TO_LOSS = {
    "earthmovers":EarthMoversLoss,
    "chamfersloss":ChamfersLoss,
    "chamfersloss_d":ChamfersLossDiagonal,
    "chamfersloss_o":ChamfersLossOptimized,
    "mse":nn.MSELoss
    }

MAPPING_TO_ED = {
    "encoder_simple":Encoder,
    "mlp_decoder":MLPDecoder,
    "conv3d_decoder":Conv3DDecoder
    }

MAPPING_TO_NETWORK = {
    "convAE":ConvAutoencoder,
    "VAE":VAE
    }

def list_transform(kwargs):
    
    for k in kwargs:
        if "layer_config" in k:
            kwargs[k] = ast.literal_eval(kwargs[k])
    

def main_args_transform(hd):
    
    criterion = MAPPING_TO_LOSS[hd["loss_function"]](**hd["loss_function_params"])
    hd.update({"loss_function": criterion})
    
    #security checks    
    decoder_kwargs = ast.literal_eval(hd["decoder_kwargs"])
    if hd["decoder_type"]=="mlp_decoder":

        if "z_dim" in decoder_kwargs:
            assert decoder_kwargs["z_dim"] == hd["z_dim"]
        else:
            decoder_kwargs["z_dim"] = hd["z_dim"]

        if "n_point" in decoder_kwargs:
            assert decoder_kwargs["n_point"] == hd["particles_to_sample"]
        else:
            decoder_kwargs["z_dim"] = hd["particles_to_sample"]
        
        if "point_dim" in decoder_kwargs:
            assert decoder_kwargs["point_dim"] == 9 if hd["property_"]=="all" else 3
        else:
            decoder_kwargs["point_dim"] = 9 if hd["property_"]=="all" else 3
    #security checks
    encoder_kwargs = ast.literal_eval(hd["encoder_kwargs"])
    if hd["encoder_type"]=="encoder_simple":

        if "z_dim" in encoder_kwargs:
            assert encoder_kwargs["z_dim"] == hd["z_dim"]
        else:
            encoder_kwargs["z_dim"] = hd["z_dim"]
        
        if "input_dim" in encoder_kwargs:
            assert encoder_kwargs["input_dim"] == 9 if hd["property_"]=="all" else 3
        else:
            encoder_kwargs["input_dim"] = 9 if hd["property_"]=="all" else 3

    encoder = MAPPING_TO_ED[hd["encoder_type"]](**encoder_kwargs)
    decoder = MAPPING_TO_ED[hd["decoder_type"]](**decoder_kwargs)
    hd.update({"encoder":encoder, "decoder":decoder})
    
    model = MAPPING_TO_NETWORK[hd["network"]](**hd)
    hd.update({"model": model})
    
    return hd
