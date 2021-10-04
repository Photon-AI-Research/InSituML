from ModelHelpers.ContinualLearner import ContinualLearner
from ModelHelpers.DeviceHelper import get_default_device, to_device
import torch
import torch.nn as nn
import math

class DimensionAutoEncoderModel(ContinualLearner):
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
        self.flatten_layer = nn.Flatten()
        
        self.inlayer_conv, _ =  self._get_ConvAndActBox(self.input_channels, self.interim_channels, 3, 1, 1)
        self.encoder_layers, self.downsamples, self.after_down_samples = self._init_encoder_layer()
        self.linear_layer_encoder = self._init_linear_layer_encoder()
        self.linear_layer_decoder = self._init_linear_layer_decoder()
        self.decoder_layers, self.upsamples, self.before_up_samples = self._init_decoder_layer()
        self.outlayer_conv, _ =  self._get_ConvAndActBox(self.interim_channels, self.input_channels, 3, 1, 1)
    
    def __half_int(self,val,power):
        return int(val/(2**power))
    
    def __get_final_layer_size(self):
        return int((2**self.n_layers) * self.input_channels)
    
    def _init_encoder_layer(self):
        modules = nn.ModuleList()
        downsamples = nn.ModuleList()
        after_down_samples = nn.ModuleList()
        
        for i in range(self.n_layers):
            conv_act = []
            for j in range(self.n_conv_layers):
                conv, act = self._get_ConvAndActBox(self.interim_channels, self.interim_channels, 3, 1, 1)
                conv_act.append(conv)
                conv_act.append(act)
            
            conv, act = self._get_ConvAndActBox(self.interim_channels, self.interim_channels, 3, 2, 1)
            downsamples.append(conv)
            
            modules.append(nn.Sequential(*conv_act))
            
        for j in range(self.n_layers):
            conv_act = []
            conv, act = self._get_ConvAndActBox(self.interim_channels, self.interim_channels, 3, 1, 1)
            conv_act.append(conv)
            conv_act.append(act)
            after_down_samples.append(nn.Sequential(*conv_act))
        
        return modules, downsamples, after_down_samples
    
    def _init_decoder_layer(self):
        modules = nn.ModuleList()
        upsamples = nn.ModuleList()
        before_up_samples = nn.ModuleList()

        for j in range(self.n_layers):
            conv_act = []
            conv, act = self._get_ConvAndActBox(self.interim_channels, self.interim_channels, 3, 1, 1)
            conv_act.append(conv)
            conv_act.append(act)
            before_up_samples.append(nn.Sequential(*conv_act))
        
        
        for i in range(self.n_layers):
            conv_act = []
            for j in range(self.n_conv_layers):
                conv, act = self._get_ConvAndActBox(self.interim_channels, self.interim_channels, 3, 1, 1)
                conv_act.append(conv)
                conv_act.append(act)
                
            
            upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            upsamples.append(upsample)

            modules.append(nn.Sequential(*conv_act))
            
        return modules, upsamples, before_up_samples
    
    def _get_final_flatten_size(self):
        return self.__half_int(self.input_sizes[0],self.n_layers) * self.__half_int(self.input_sizes[1],self.n_layers) * self.__half_int(self.input_sizes[2],self.n_layers) * self.interim_channels
    
    def _init_linear_layer_encoder(self):
        return nn.Linear(self._get_final_flatten_size(),self.latent_size,bias = False)
    
    def _init_linear_layer_decoder(self):
        return nn.Linear(self.latent_size,self._get_final_flatten_size(),bias = False)
            
    def _get_ConvAndActBox(self,in_channels,out_channels, kernel_size = 3 , stride = 1, padding = 1):
        conv = nn.Conv3d(in_channels,out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        act = self.act
        return conv, act
    
    def _get_DeConvAndUnPoolBox(self, in_channels, out_channels, kernel_size = 3 , stride = 1, padding = 1):
        deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        act = self.act
        return deconv, act
    
    def _weights_init(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        if isinstance(m, nn.ConvTranspose3d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias:
                nn.init.xavier_uniform_(m.bias)
    
    
    def encode(self,x):
        x = self.inlayer_conv(x)
        x = self.act(x)
        x0 = x
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)
            x = torch.add(x,x0)
            x = self.downsamples[i](x)
            x = self.act(x)
            x0 = x
            
        for j in range(self.n_layers):
            x = self.after_down_samples[j](x)
            x = self.act(x)
        x = self.flatten_layer(x)
        return x
    
    def decode(self, x):
        #calculate reshape dimensions
        x = x.view(1,self.interim_channels,self.__half_int(self.input_sizes[0],self.n_layers),self.__half_int(self.input_sizes[1],self.n_layers),self.__half_int(self.input_sizes[2],self.n_layers))
        
        for j in range(self.n_layers):
            x = self.before_up_samples[j](x)
            
        x0 = x
        for i in range(self.n_layers):
            x = self.decoder_layers[i](x)
            x = torch.add(x,x0)
            x = self.upsamples[i](x)
            x0 = x
        
        x = self.outlayer_conv(x)
        
        return x
        
    def forward(self, x):
        x = self.encode(x)
        encoded = self.linear_layer_encoder(x)
        x = self.linear_layer_decoder(encoded)
        decoded = self.decode(x)
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
        return loss, torch.tensor(0., device=self.device)
    
    def validation_step(self, batch):
        return {'val_loss': self.training_step(batch)}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
        
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
        torch.save(state, filePath)