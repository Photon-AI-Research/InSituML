import pytest
from encoder_decoder import Encoder
from unittest.mock import patch, Mock
import os 
import torch


def test_encoder():
    
    encoder_obj = Encoder(zdim = 3,
                          input_dim = 3, 
                          ae_config ="deterministic",
                          conv_layer_config = [512],
                          conv_add_bn = True,
                          conv_add_activation = True,
                          kernel_size = 1,
                          fc_layer_config=[128],
                          fc_add_bn = True,
                          fc_add_activation = True)
    
    print(encoder_obj.layers)
    input_ = torch.rand(1, 2, 3)
    output = encoder_obj(input_)
    assert True
