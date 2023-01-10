import torch
import torch.nn as nn

from ModelHelpers.ContinualLearner import ContinualLearner


class AutoEncoder2D(ContinualLearner):

    def __init__(self,input_channels, input_sizes, n_layers, n_conv_layers, filters, latent_size, act, onlineEWC=False, ewc_lambda=0, gamma=0):
        super().__init__(onlineEWC=onlineEWC, ewc_lambda=ewc_lambda, gamma=gamma)
        self.input_channels = input_channels
        self.input_sizes = input_sizes
        self.n_layers = n_layers
        self.n_conv_layers = n_conv_layers
        self.filters = filters
        self.latent_size = latent_size
        self.act = act
        self.enc_layers , self.pooling_layers = self.__init_encoder_layers()
        self.dec_layers , self.unpooling_layers = self.__init_decoder_layers()
        self.interim_size = self._get_final_flatten_size()
        self.linear_layer_encoder = self._init_linear_layer_encoder()
        self.linear_layer_decoder = self._init_linear_layer_decoder()
        self.up_sample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.flatten_layer = nn.Flatten()
    
    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('leaky_relu', 0.2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('leaky_relu', 0.2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias:
                nn.init.xavier_uniform_(m.bias)

    def _init_linear_layer_encoder(self):
        return nn.Linear(self._get_final_flatten_size(),self.latent_size,bias = False)
    
    def _init_linear_layer_decoder(self):
        return nn.Linear(self.latent_size,self._get_final_flatten_size(),bias = False)

    def __half_int(self,val,power):
        return int(val/(2**power))
    
    def _get_final_flatten_size(self):
        return self.__half_int(self.input_sizes[0],self.n_layers) * self.__half_int(self.input_sizes[1],self.n_layers) * self.filters[self.n_layers - 1]
    
    def _get_ConvAndActBox(self,in_channels,out_channels, kernel_size = 3 , stride = 1, padding = 1):
        conv = nn.Conv2d(in_channels,out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        act = self.act
        
        pool = nn.MaxPool2d(2, return_indices = True)
        return conv, act, pool
    
    def _get_DeConvAndUnPoolBox(self, in_channels, out_channels, kernel_size = 3 , stride = 1, padding = 1):
        deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        act = self.act
        unpool = nn.MaxUnpool2d(2)
        return deconv, act, unpool

    def __init_encoder_layers(self):
        modules = nn.ModuleList()
        pooling_layers = nn.ModuleList()

        for layer in range(self.n_layers):
            conv_act = []
            for sub_layer in range(self.n_conv_layers):
                #for first layer
                if layer == 0 and sub_layer == 0:
                    conv, act, pool = self._get_ConvAndActBox(self.input_channels, self.filters[layer])
                elif sub_layer == 0:
                    conv, act, pool = self._get_ConvAndActBox(self.filters[layer-1], self.filters[layer])
                else:
                    conv, act, pool = self._get_ConvAndActBox(self.filters[layer], self.filters[layer])

                conv_act.append(conv)
                conv_act.append(act)
            
            pooling_layers.append(pool)
            modules.append(nn.Sequential(*conv_act))
        
        return modules, pooling_layers
    
    def __init_decoder_layers(self):
        modules = nn.ModuleList()
        unpooling_layers = nn.ModuleList()

        for layer in range(self.n_layers):
            conv_act = []
            for sub_layer in range(self.n_conv_layers):
                #for last layer
                if layer == self.n_layers - 1 and sub_layer == self.n_conv_layers - 1:
                    conv, act, unpool = self._get_DeConvAndUnPoolBox(self.filters[self.n_layers - layer - 1], self.input_channels)
                elif sub_layer == self.n_conv_layers - 1:
                    conv, act, unpool = self._get_DeConvAndUnPoolBox(self.filters[self.n_layers - 1 -layer], self.filters[self.n_layers - 1 - layer-1])
                else:
                    conv, act, unpool = self._get_DeConvAndUnPoolBox(self.filters[self.n_layers - 1 - layer], self.filters[self.n_layers - 1 -layer])

                conv_act.append(act)
                conv_act.append(conv)
                
            
            unpooling_layers.append(unpool)
            modules.append(nn.Sequential(*conv_act))
        
        return modules, unpooling_layers

    def encode(self,x):
        pooling_indexes_list = []
        for i in range(self.n_layers):
            x = self.enc_layers[i](x)
            x, indices = self.pooling_layers[i](x)
            pooling_indexes_list.append(indices)
            
        x = self.flatten_layer(x)
        x = self.linear_layer_encoder(x)
        x = self.act(x)
        return x, pooling_indexes_list

    def decode(self, x, pooling_indexes_list):
        #calculate reshape dimensions
        x = self.linear_layer_decoder(x)
        x = x.view(-1,self.filters[self.n_layers - 1],self.__half_int(self.input_sizes[0],self.n_layers),self.__half_int(self.input_sizes[1],self.n_layers))
        b_size = x.size()[0]

        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                op_size = torch.Size((b_size, self.filters[0],self.input_sizes[0],self.input_sizes[1] ))
            else:
                op_size = pooling_indexes_list[-(i+2)].size()
            x = self.unpooling_layers[i](x, pooling_indexes_list[-(i+1)], output_size = op_size)
            x = self.dec_layers[i](x)

        return x
        
    def forward(self,x):
        encoded, pooling_index_list = self.encode(x)
        decoded = self.decode(encoded, pooling_index_list)
        return encoded, decoded, pooling_index_list
    
    def from_dec_to_enc(self,x):
        x = self.linear_layer_decoder(x)
        x = x.view(-1,self.filters[self.n_layers - 1],self.__half_int(self.input_sizes[0],self.n_layers),self.__half_int(self.input_sizes[1],self.n_layers))
        for i in range(self.n_layers):
            x = self.up_sample(x)
            x = self.dec_layers[i](x)
        encoded, _ = self.encode(x)
        return encoded
    
    def save_checkpoint(self, path, name, after_task):
        state = {
            'model': self.state_dict(),
            'input_channels' : self.input_channels,
            'input_sizes' : self.input_sizes,
            'n_layers' : self.n_layers,
            'filters': self.filters,
            'n_conv_layers': self.n_conv_layers,
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
            
        filePath = path + str(name) + "_" + str(after_task)
 
        try:
            torch.save(state, filePath)
        except:
            pass
