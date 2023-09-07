import os

import torch
import torch.nn as nn

from ModelHelpers.ContinualLearner import ContinualLearner
from ModelHelpers.DeviceHelper import to_device

class MLP(ContinualLearner):

    def __init__(self, input_channels, input_sizes, n_layers, n_conv_layers, filters, latent_size, act, onlineEWC=False, ewc_lambda=0, gamma=0):
        super().__init__(onlineEWC=onlineEWC, ewc_lambda=ewc_lambda, gamma=gamma)
        self.input_channels = input_channels
        self.input_sizes = input_sizes
        self.first_input_dim = self._get_input_dim()
        self.filters = filters
        self.act = act
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.encoder_module = self._init_encoder()
        self.decoder_module = self._init_decoder()
    def _get_input_dim(self):
        mul = self.input_channels
        for ip_size in self.input_sizes:
            mul *= ip_size
        return mul

    def _init_encoder(self):
        encoder = nn.ModuleList()
        first_layer = nn.Linear(self.first_input_dim, self.filters[0], bias=False)
        encoder.append(first_layer)
        for layer in range(self.n_layers):
            if layer == self.n_layers - 1:
                l_layer =nn.Linear(self.filters[layer], self.latent_size, bias=False)
            else:
                 l_layer =nn.Linear(self.filters[layer], self.filters[layer + 1], bias=False)
            encoder.append(l_layer)
        return encoder
    
    def _init_decoder(self):
        decoder = nn.ModuleList()
        last_layer = nn.Linear(self.filters[0], self.first_input_dim, bias=False)
        for layer in range(self.n_layers):
            if layer == 0:
                l_layer =nn.Linear(self.latent_size, self.filters[-(layer+1)], bias=False)
            else:
                 l_layer =nn.Linear(self.filters[-layer], self.filters[-(layer+1)], bias=False)
            decoder.append(l_layer)
        decoder.append(last_layer)

        return decoder
    
    def _weights_init(self, m):          
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias:
                nn.init.xavier_uniform_(m.bias)

    def encode(self, x):
        for layer in self.encoder_module:
            x = layer(x)
            x = self.act(x)
        return x
    
    def decode(self, x):
        for layer in self.decoder_module:
            x = layer(x)
            x = self.act(x)
        return x
    
    def from_dec_to_enc(self,x):
        x = self.decode(x)
        encoded = self.encode(x)
        return encoded

    def forward(self,x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded, None
    
    def save_checkpoint(self, path, name, after_task):
        state = {
            'model': self.state_dict(),
            'input_channels' : self.input_channels,
            'input_sizes' : self.input_sizes,
            'n_layers' : self.n_layers,
            'filters': self.filters,
            'latent_size':self.latent_size,
            'act' : self.act
        }

        keys = []
        for key in state['model'].keys():
            if "EWC" in key:
                keys.append(key)
            
            if "Grad_Ref_Estimate" in key:
                keys.append(key)
        
        for key in keys:
            state['model'].pop(key, None)
            
        filePath = os.path.join(path, str(name) + "_" + str(after_task))
 
        try:
            torch.save(state, filePath)
        except:
            pass
            
