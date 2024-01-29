import torch
import torch.nn as nn
from encoder_decoder import Encoder as encoder
from encoder_decoder import MLP_Decoder as decoder
from utilities import sample_gaussian, kl_normal

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
# Define the convolutional autoencoder class
class ConvAutoencoder(nn.Module):
    def __init__(self, config):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512, config["hidden_size"], kernel_size=1),
            nn.AdaptiveMaxPool1d(config["dim_pool"]), 
            nn.Flatten()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16,4,4,4)),
            nn.ConvTranspose3d(16, 4,kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(4, 9,kernel_size=2, stride=2),
            nn.Flatten(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #Adding None for compatibility
        #TODO: calculate loss here. 
        return None, x


class VAE(nn.Module):
    def __init__(self, loss_function = None, point_dim=3, z_dim=4):
        super(VAE, self).__init__()
        self.n_point = 150000
        self.point_dim = point_dim
        self.z_dim = z_dim
        self.loss_function = loss_function
        self.use_deterministic_encoder = False
        self.use_encoding_in_decoder = False
        self.encoder = encoder(self.z_dim,
                               self.point_dim,
                               self.use_deterministic_encoder)
        
        if not self.use_deterministic_encoder and self.use_encoding_in_decoder:
            self.decoder = decoder(2 *self.z_dim,self.n_point,self.point_dim)
        else:
            self.decoder = decoder(self.z_dim,self.n_point,self.point_dim)
        #set prior parameters of the vae model p(z) with 0 mean and 1 variance.
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.type = 'VAE'
    
    def forward(self, inputs):
        x = inputs
        m, v = self.encoder(x)
        if self.use_deterministic_encoder:
            y = self.decoder(m)
            kl_loss = torch.zeros(1)
        else:
            z =  sample_gaussian(m,v)
            decoder_input = z if not self.use_encoding_in_decoder else \
            torch.cat((z,m),dim=-1) 
            y = self.decoder(decoder_input)
            #compute KL divergence loss :
            p_m = self.z_prior[0].expand(m.size())
            p_v = self.z_prior[1].expand(v.size())
            kl_loss = kl_normal(m,v,p_m,p_v)
        #compute reconstruction loss 
        if self.loss_function is not None:
            x_reconst = self.loss_function(y.contiguous(),x.contiguous())
        # mean or sum
        x_reconst = x_reconst.mean()
        kl_loss = kl_loss.mean()

        nelbo = x_reconst + kl_loss
        #might be useful for later.
        #ret = {'nelbo':nelbo, 'kl_loss':kl_loss, 'x_reconst':x_reconst}
        return nelbo, y
    

    def sample_point(self, batch):
        p_m = self.z_prior[0].expand(batch,self.z_dim).to(device)
        p_v = self.z_prior[1].expand(batch,self.z_dim).to(device)
        z =  sample_gaussian(p_m,p_v)
        decoder_input = z if not self.use_encoding_in_decoder else \
        torch.cat((z,p_m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
        y = self.decoder(decoder_input)
        return y

    def reconstruct_input(self, x):
        m, v = self.encoder(x)
        if self.use_deterministic_encoder:
            y = self.decoder(m)
        else:
            z =  sample_gaussian(m,v)
            decoder_input = z if not self.use_encoding_in_decoder else \
            torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
            y = self.decoder(decoder_input)
        return y

