from ModelHelpers.ContinualLearner import ContinualLearner
from ModelHelpers.DeviceHelper import get_default_device, to_device, DeviceDataLoader
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import math

class DimensionAutoEncoderModelWithPool(ContinualLearner):
    def __init__(self,input_channels, input_sizes, loss_func, n_layers, n_conv_layers, filters, latent_size, act, onlineEWC = False, ewc_lambda = 0.0, gamma = 0.0):
        super().__init__(onlineEWC ,ewc_lambda, gamma)
        self.loss_func = loss_func
        self.input_channels = input_channels
        self.interim_channels = filters
        self.input_sizes = input_sizes
        self.n_layers = n_layers
        self.n_conv_layers = n_conv_layers
        self.latent_size = latent_size
        self.act = act
        self.tanh = nn.Tanh()        
        self.flatten_layer = nn.Flatten()
        
        self.encoder_layers , self.pooling_layers = self._init_encoder_layer()
        self.linear_layer_encoder = self._init_linear_layer_encoder()
        self.linear_layer_decoder = self._init_linear_layer_decoder()
        self.decoder_layers , self.unpooling_layers = self._init_decoder_layer()
    
    def __half_int(self,val,power):
        return int(val/(2**power))
    
    def __get_final_layer_size(self):
        return int((2**self.n_layers) * self.input_channels)
    
    def _init_encoder_layer(self):
        modules = nn.ModuleList()
        pooling_layers = nn.ModuleList()
        
        self.inlayer_conv, _ , _=  self._get_ConvAndActBox(self.input_channels, self.interim_channels, 3, 1, 1)
        for i in range(self.n_layers):
            conv_act = []
            for j in range(self.n_conv_layers):
                conv, act,pool = self._get_ConvAndActBox(self.interim_channels, self.interim_channels, 3, 1, 1)
                conv_act.append(conv)
                conv_act.append(act)
            
            pooling_layers.append(pool)

            modules.append(nn.Sequential(*conv_act))
            
        return modules, pooling_layers
    
    def _init_decoder_layer(self):
        modules = nn.ModuleList()
        unpooling_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            conv_act = []
            for j in range(self.n_conv_layers):
                conv, act, unpool = self._get_DeConvAndUnPoolBox(self.interim_channels, self.interim_channels, 3, 1, 1)
                conv_act.append(act)
                conv_act.append(conv)
            
            unpooling_layers.append(unpool)
        
            modules.append(nn.Sequential(*conv_act))
        self.outlayer_conv, _, _ =  self._get_DeConvAndUnPoolBox(self.interim_channels, self.input_channels, 3, 1, 1)
            
        return modules, unpooling_layers
    
    def _get_final_flatten_size(self):
        return self.__half_int(self.input_sizes[0],self.n_layers) * self.__half_int(self.input_sizes[1],self.n_layers) * self.__half_int(self.input_sizes[2],self.n_layers) * self.interim_channels
    
    def _init_linear_layer_encoder(self):
        return nn.Linear(self._get_final_flatten_size(),self.latent_size,bias = False)
    
    def _init_linear_layer_decoder(self):
        return nn.Linear(self.latent_size,self._get_final_flatten_size(),bias = False)
            
    def _get_ConvAndActBox(self,in_channels,out_channels, kernel_size = 3 , stride = 1, padding = 1):
        conv = nn.Conv3d(in_channels,out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        act = self.act
        
        pool = nn.MaxPool3d(2, return_indices = True)
        return conv, act, pool
    
    def _get_DeConvAndUnPoolBox(self, in_channels, out_channels, kernel_size = 3 , stride = 1, padding = 1):
        deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        act = self.act
        unpool = nn.MaxUnpool3d(2)
        return deconv, act, unpool
    
    def _weights_init(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('leaky_relu', 0.2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        if isinstance(m, nn.ConvTranspose3d):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('leaky_relu', 0.2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias:
                nn.init.xavier_uniform_(m.bias)
    
    
    def encode(self,x):
        pooling_indexes_list = []
        x = self.inlayer_conv(x)
        x = self.act(x)
        x0 = x
        for i in range(self.n_layers):
            #for j in range(self.n_conv_layers):
            x = self.encoder_layers[i](x)
            #x = torch.add(x,x0)
            x, indices = self.pooling_layers[i](x)
            pooling_indexes_list.append(indices)
            x0 = x
            
        x = self.flatten_layer(x)
        return x, pooling_indexes_list
    
    def decode(self, x, pooling_indexes_list):
        #calculate reshape dimensions
        x = x.view(-1,self.interim_channels,self.__half_int(self.input_sizes[0],self.n_layers),self.__half_int(self.input_sizes[1],self.n_layers),self.__half_int(self.input_sizes[2],self.n_layers))
        
        x0 = x
        for i in range(self.n_layers):
            x = self.decoder_layers[i](x)
            #x = torch.add(x,x0)
            x = self.unpooling_layers[i](x, pooling_indexes_list[-(i+1)])
            x0 = x
        
        x = self.act(x)
        x = self.outlayer_conv(x)
        x = self.act(x)
        return x
        
    def forward(self, x):
        x,pooling_indexes = self.encode(x)
        encoded = self.linear_layer_encoder(x)
        
        x = self.linear_layer_decoder(encoded)
        
        decoded = self.decode(x, pooling_indexes)
        
        return encoded, decoded        

    def training_step(self, batch):
        data = batch 
        _ , out = self(data)                 # Generate decoded data
        loss = self.loss_func(out, data) # Calculate loss
        if self.EWC_task_count == 1: 
            ewc_loss = self.ewc_lambda * self.ewc_loss()
            print("EWC LOSS: ",ewc_loss.item(), self.ewc_lambda)
            if math.isnan(ewc_loss):
                return loss
            return loss + ewc_loss , ewc_loss
        return loss , torch.tensor(0., device=self.device)

    def validation_step(self, batch):
        data = batch 
        _ , out = self(data)                 # Generate decoded data
        loss = self.loss_func(out, data)
        return loss
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
        
    def save_checkpoint(self, path, name, after_task):
        state = {
            'model': self.state_dict(),
            'input_channels' : self.input_channels,
            'input_sizes' : self.input_sizes,
            'loss_func' : self.loss_func,
            'n_layers' : self.n_layers,
            'filters': self.interim_channels,
            'n_conv_layers': self.n_conv_layers,
            'latent_size':self.latent_size,
            'act' : self.act
        }

        keys = []
        for key in state['model'].keys():
            if "EWC" in key:
                keys.append(key)
        
        for key in keys:
            state['model'].pop(key, None)
            
        filePath = path + str(name) + "_" + str(after_task)
        print(path,type(path))
        print(name,type(name))
        print(after_task,type(after_task))
        print(filePath,type(filePath))
            
        torch.save(state, filePath)
