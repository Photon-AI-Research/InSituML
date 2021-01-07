import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionAutoEncoderModel(nn.Module):
    def __init__(self,input_channels, input_sizes, loss_func, n_layers, latent_size):
        super(DimensionAutoEncoderModel,self).__init__()
        self.loss_func = loss_func
        self.input_channels = input_channels
        self.input_sizes = input_sizes
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.flatten_layer = nn.Flatten()
        
        self.encoder_layers, self.pooling_layers = self._init_encoder_layer()
        self.linear_layer_encoder = self._init_linear_layer_encoder()
        self.linear_layer_decoder = self._init_linear_layer_decoder()
        self.decoder_layers, self.unpooling_layers = self._init_decoder_layer()
    
    def __half_int(self,val,power):
        return int(val/(2**power))
    
    def __get_final_layer_size(self):
        return int((2**self.n_layers) * self.input_channels)
    
    def _init_encoder_layer(self):
        modules = nn.ModuleList()
        pooling_layers = nn.ModuleList()
        for i in range(self.n_layers):
            conv_act = []
            conv, act, pool = self._get_ConvAndPoolBox((2**i)*self.input_channels, (2**(i+1))*self.input_channels, 3, 1, 1)
            
            conv_act.append(conv)
            conv_act.append(act)
            
            pooling_layers.append(pool)
            modules.append(nn.Sequential(*conv_act))
            
        return modules, pooling_layers
    
    def _init_decoder_layer(self):
        modules = nn.ModuleList()
        unpooling_layers = nn.ModuleList()
        for i in range(self.n_layers -1 ,-1,-1):
            deconv_act = []
            deconv, act, unpool = self._get_DeConvAndUnPoolBox((2**(i+1))*self.input_channels,(2**i)*self.input_channels, 3, 1, 1)
            
            deconv_act.append(act)
            deconv_act.append(deconv)
            
            modules.append(nn.Sequential(*deconv_act))
            unpooling_layers.append(unpool)
            
        return modules, unpooling_layers
    
    def _get_final_flatten_size(self):
        return self.__half_int(self.input_sizes[0],self.n_layers) * self.__half_int(self.input_sizes[1],self.n_layers) * self.__half_int(self.input_sizes[2],self.n_layers) * self.__get_final_layer_size()
    
    def _init_linear_layer_encoder(self):
        return nn.Linear(self._get_final_flatten_size(),self.latent_size,bias = False)
    
    def _init_linear_layer_decoder(self):
        return nn.Linear(self.latent_size,self._get_final_flatten_size(),bias = False)
            
    def _get_ConvAndPoolBox(self,in_channels,out_channels, kernel_size = 3 , stride = 1, padding = 1):
        conv = nn.Conv3d(in_channels,out_channels, kernel_size = kernel_size, stride = stride, padding = stride)
        act = nn.ReLU(True)
        pool = nn.MaxPool3d(2, return_indices = True)
        return conv, act, pool
    
    def _get_DeConvAndUnPoolBox(self, in_channels, out_channels, kernel_size = 3 , stride = 1, padding = 1):
        deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = stride)
        act = nn.ReLU(True)
        unpool = nn.MaxUnpool3d(2)
        return deconv, act, unpool
    
    
    def encode(self,x):
        pooling_indexes_list = []
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)
            x, indices = self.pooling_layers[i](x)
            pooling_indexes_list.append(indices)
        print(x.size())
        x = self.flatten_layer(x)
        print("yoyo",x.size())
        return x, pooling_indexes_list
    
    def decode(self, x, indices_list):
        pooling_indexes_list = indices_list
        #calculate reshape dimensions
        last_enc_channel_out = (2**self.n_layers) * self.input_channels
        x = x.view(1,last_enc_channel_out,self.__half_int(self.input_sizes[0],self.n_layers),self.__half_int(self.input_sizes[1],self.n_layers),self.__half_int(self.input_sizes[2],self.n_layers))
                
        for i in range(self.n_layers):
            x = self.unpooling_layers[i](x,pooling_indexes_list[self.n_layers - i - 1])
            x = self.decoder_layers[i](x)
            
        return x
        
    def forward(self, x):
        x, pooling_indexes_list = self.encode(x)
        x = self.linear_layer_encoder(x)
        x = self.linear_layer_decoder(x)
        
        x = self.decode(x, pooling_indexes_list)
        
        return x
    
    def training_step(self, batch):
        data = batch 
        out = self(data)                 # Generate decoded data
        loss = self.loss_func(out, data) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        return {'val_loss': self.training_step(batch)}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
        
    def save_checkpoint(self, model, path, epoch):
        state = {
            'model': model.state_dict(),
        }
        torch.save(state, path + 'model_' + str(epoch))
    
    def load_checkpoint(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])  