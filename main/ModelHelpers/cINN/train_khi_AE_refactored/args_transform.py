from networks import ConvAutoencoder, VAE
from encoder_decoder import Encoder, MLPDecoder, Conv3DDecoder
from loss_functions import EarthMoversLoss, ChamfersLoss, ChamfersLossDiagonal ,ChamfersLossOptimized
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
        if "layer_config" in k or "initial_conv3d_size" in k:
            kwargs[k] = ast.literal_eval(kwargs[k])
    
    return kwargs
    

def main_args_transform(hd):
    
    criterion = MAPPING_TO_LOSS[hd["loss_function"]](**hd["loss_function_params"])
    hd.update({"loss_function": criterion})
    
    #security checks    
    decoder_kwargs = list_transform(ast.literal_eval(hd["decoder_kwargs"]))
    #security checks
    encoder_kwargs = list_transform(ast.literal_eval(hd["encoder_kwargs"]))
    
    encoder = MAPPING_TO_ED[hd["encoder_type"]]
    decoder = MAPPING_TO_ED[hd["decoder_type"]]
    hd.update({"encoder":encoder,
               "decoder":decoder,
               "encoder_kwargs":encoder_kwargs, 
               "decoder_kwargs":decoder_kwargs})
    
    model = MAPPING_TO_NETWORK[hd["network"]](**hd)
    hd.update({"model": model})

    hd["val_boxes"] = ast.literal_eval(hd["val_boxes"])
    
    return hd
