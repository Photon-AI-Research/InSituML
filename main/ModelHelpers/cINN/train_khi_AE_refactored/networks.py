import torch
import torch.nn as nn
from encoder_decoder import Encoder as encoder
from encoder_decoder import MLP_Decoder as decoder
from utilities import sample_gaussian, kl_normal, inspect_and_select

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
# Define the convolutional autoencoder class
@inspect_and_select
class ConvAutoencoder(nn.Module):
    def __init__(self, property_, hidden_size, dim_pool):
        super().__init__()
        self.input_dim = 9 if property_ == "all" else 3

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 16, kernel_size=1),
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
            nn.Conv1d(512, hidden_size, kernel_size=1),
            nn.AdaptiveMaxPool1d(dim_pool), 
            nn.Flatten()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16,4,4,4)),
            nn.ConvTranspose3d(16, 4,kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(4, self.input_dim,kernel_size=2, stride=2),
            nn.Flatten(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #Adding None for compatibility
        #TODO: calculate loss here. 
        return None, x

@inspect_and_select
class VAE(nn.Module):
    def __init__(self, loss_function = None, 
                 property_="positions", z_dim=4, 
                 use_deterministic_encoder=True,
                 use_encoding_in_decoder=True
                 ):
        super().__init__()
        self.point_dim = 9 if property_ == "all" else 3
        self.n_point = 150000
        self.z_dim = z_dim
        self.loss_function = loss_function
        self.use_deterministic_encoder = use_deterministic_encoder
        self.use_encoding_in_decoder = use_encoding_in_decoder
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

