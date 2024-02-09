from networks import ConvAutoencoder, VAE
from encoder_decoder import Encoder, MLPDecoder, Conv3DDecoder
from loss_functions import EarthMoversLoss, ChamfersLoss, ChamfersLossDiagonal
#, ChamfersLossOptimized
import ast
import torch.nn as nn


MAPPING_TO_LOSS = {
    "earthmovers":EarthMoversLoss,
    "chamfersloss":ChamfersLoss,
    "chamfersloss_d":ChamfersLossDiagonal,
 #   "chamfersloss_o":ChamfersLossOptimized,
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

def main_args_transform(hd):
    
    criterion = MAPPING_TO_LOSS[hd["loss_function"]](**hd["loss_function_params"])
    hd.update({"loss_function": criterion})
    
    encoder = MAPPING_TO_ED[hd["encoder_type"]](**ast.literal_eval(hd["encoder_kwargs"]))
    decoder = MAPPING_TO_ED[hd["decoder_type"]](**ast.literal_eval(hd["decoder_kwargs"]))
    hd.update({"encoder":encoder, "decoder":decoder})
    
    model = MAPPING_TO_NETWORK[hd["network"]](**hd)
    hd.update({"model": model})
    
    return hd
    
    
    return hyperparameter_defaults
