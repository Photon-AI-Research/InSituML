import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class AddLayersMixin:
    
    def add_layers_seq(self, layers_kind, config,
                       add_activation = True,
                       add_batch_normalisation = True):
        
        layers = []
        for idx, channel_size in enumerate(config):
            if idx==0:
                layers.append(getattr(nn, layer_kind)(input_dim, channel_size, 1))
            else:
                layers.append(getattr(nn, layer_kind)(config[idx-1], channel_size, 1))
            
            if add_batch_normalisation:
                layers.append(nn.BatchNorm1d(channel_size, 1))
            
            if add_activation:
                layers.append(nn.relu())
        
        return layers



#simple encodr design
class Encoder(AddLayersMixin, nn.Module):
    def __init__(self, zdim,
                 input_dim = 3, 
                 ae_config = "determistic",
                 conv_layer_config = [128, 128, 256, 512],
                 fc_layer_config=[256, 128]):
        
        super().__init__()
        
        conv_layers = self.add_layers_seq("Conv1d",
                                           conv_layer_config,
                                           input_dim)
        
        self.ae_config = ae_config
        
        self.ll_size = conv_layer_config[-1]

        conv_layers += [nn.AdaptiveMaxPool1d(1),
                        nn.Flatten()]
        
        if ae_config == "determistic":
            conv_layers += [nn.Unflatten(1, (-1, self.ll_size))]
            fc_layers = self.add_layers_seq("Linear",
                                            fc_layer_config,
                                            self.ll_size)
            
            final_layers = conv_layers + fc_layers + \
                           [nn.Linear(self.ll_size, zdim)]
                       
            self.layers = nn.Sequential(*final_layers)

        elif ae_config == "non_deterministic":

            conv_layers += [nn.Unflatten(1, (-1, self.ll_size))]
            self.layers = nn.Sequential(*conv_layers)
            
            fc_layers_mean = self.add_layers_seq("Linear",
                                                 fc_layer_config,
                                                 self.ll_size)
            
            fc_layers_var = self.add_layers_seq("Linear",
                                                 fc_layer_config,
                                                 self.ll_size)
            
            partition_mean = fc_layers_mean + [nn.Linear(self.ll_size, zdim)]
            
            partition_var = fc_layers_var + [nn.Linear(self.ll_size, zdim)]
            
            self.mean = nn.Sequential(*partition_mean)
            self.variance = nn.Sequential(*partition_var)
        
        else:
            self.layers = nn.Sequential(*conv_layers)
            

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)

        if self.ae_config == "deterministic":
            return x, 0
        elif self.ae_config == "non_deterministic":
            return self.mean(x), self.variance(x)
        else:
            return x

#decoder design
class MLP_Decoder(AddLayersMixin, nn.Module):
    def __init__(self, zdim, n_point, point_dim,
                layer_config = [256, 256]):
        
        super().__init__()
        self.zdim = zdim
        self.n_point = n_point
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        layers = self.add_layers_seq("Linear", 
                                     self.zdim,
                                     layer_config,
                                     add_batch_normalisation = False)
        output.reshape(-1,self.n_point,self.point_dim)
        
        layers = layers + [nn.Linear(layer_config[-1], self.n_point_3)]
        
        layers = layers + [nn.Flatten(0),
                                nn.Unflatten(0, (-1, self.n_point, self.point_dim))]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.layer(z)
