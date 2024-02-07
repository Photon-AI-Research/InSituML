import pytest
from encoder_decoder import Encoder, MLPDecoder
from unittest.mock import patch, Mock
import os 
import torch

zdim = 3
input_dim = 3
batch_size = 2
number_of_particles = 4
input_ = torch.rand(batch_size, number_of_particles, input_dim)
expected_encoder_output = torch.rand(batch_size, zdim)

def test_encoder_simple():
    
    
    encoder_obj = Encoder(zdim,
                          input_dim = input_dim, 
                          ae_config ="simple",
                          conv_layer_config = [512],
                          conv_add_bn = True,
                          conv_add_activation = True)
    

    # conv1,bn, act, conv1, maxpooling, flatten
    assert len(encoder_obj.layers) == 6  

    encoder_obj = Encoder(zdim = 3,
                          input_dim = 3, 
                          ae_config ="simple",
                          conv_layer_config = [512, 234],
                          conv_add_bn = True,
                          conv_add_activation = True)

    # 2*(conv1,bn,act), conv1, maxpooling, flatten 
    assert len(encoder_obj.layers) == 9 

    encoder_obj = Encoder(zdim = 3,
                          input_dim = 3, 
                          ae_config ="simple",
                          conv_layer_config = [512, 234],
                          conv_add_bn = False,
                          conv_add_activation = True)

    # 2*(conv1,act), conv1, maxpooling, flatten 
    assert len(encoder_obj.layers) == 7
        
    output = encoder_obj(input_)

    assert output.shape[0] == batch_size
    assert output.shape[1] == zdim
    
def test_encoder_deterministic():
    
    
    encoder_obj = Encoder(zdim,
                          input_dim = input_dim, 
                          ae_config ="deterministic",
                          conv_layer_config = [512],
                          conv_add_bn = True,
                          conv_add_activation = True,
                          fc_layer_config = [128],
                          fc_add_bn = True,
                          fc_add_activation = True)
    
    # conv1, bn, act, maxpooling, flatten, 
    # fc, bn, act, fc
    assert len(encoder_obj.layers) == 9  

    encoder_obj = Encoder(zdim = 3,
                          input_dim = 3, 
                          ae_config ="deterministic",
                          conv_layer_config = [512, 234],
                          conv_add_bn = True,
                          conv_add_activation = True,
                          fc_layer_config = [128],
                          fc_add_bn = True,
                          fc_add_activation = True)

    # 2*(conv1, bn, act), maxpooling, flatten
    # fc, bn, act, fc
    assert len(encoder_obj.layers) == 12 

    encoder_obj = Encoder(zdim = 3,
                          input_dim = 3, 
                          ae_config ="deterministic",
                          conv_layer_config = [512],
                          conv_add_bn = True,
                          conv_add_activation = True,
                          fc_layer_config = [128, 234],
                          fc_add_bn = True,
                          fc_add_activation = True)

    # conv1, bn, act, maxpooling, flatten
    # 2*(fc, bn, act), fc
    assert len(encoder_obj.layers) == 12
        
    output, _ = encoder_obj(input_)

    assert output.shape[0] == batch_size
    assert output.shape[1] == zdim

def test_encoder_non_deterministic():
    
    
    encoder_obj = Encoder(zdim,
                          input_dim = input_dim, 
                          ae_config ="non_deterministic",
                          conv_layer_config = [512],
                          conv_add_bn = True,
                          conv_add_activation = True,
                          fc_layer_config = [128],
                          fc_add_bn = True,
                          fc_add_activation = True)
    
    # conv1, bn, act, maxpooling, flatten, 
    assert len(encoder_obj.layers) == 5  

    # fc, bn, act, fc  
    assert len(encoder_obj.mean) == 4  
    
    # fc, bn, act, fc  
    assert len(encoder_obj.variance) == 4  


    encoder_obj = Encoder(zdim,
                          input_dim = input_dim, 
                          ae_config ="non_deterministic",
                          conv_layer_config = [512],
                          conv_add_bn = True,
                          conv_add_activation = True,
                          fc_layer_config = [128, 124],
                          fc_add_bn = True,
                          fc_add_activation = True)
    
    # (conv1, bn, act,) maxpooling, flatten, 
    assert len(encoder_obj.layers) == 5  

    # 2*(fc, bn, act), fc  
    assert len(encoder_obj.mean) == 7 
    
    # 2*(fc, bn, act), fc 
    assert len(encoder_obj.variance) == 7

    mean, variance = encoder_obj(input_)

    assert mean.shape[0] == batch_size
    assert mean.shape[1] == zdim

    assert variance.shape[0] == batch_size
    assert variance.shape[1] == zdim

def test_mlp_decoder():
    
    decoder_obj = MLPDecoder(zdim,
                             number_of_particles, 
                             input_dim,
                             layer_config = [256])

    # (fc, act), fc, flatten, unflatten  
    assert len(decoder_obj.layers) == 5

    decoder_obj = MLPDecoder(zdim,
                             number_of_particles, 
                             input_dim,
                             layer_config = [256, 123])

    # 2*(fc, act), fc, flatten, unflatten  
    assert len(decoder_obj.layers) == 7
    
    output = decoder_obj(expected_encoder_output)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == number_of_particles
    assert output.shape[2] == input_dim
    
