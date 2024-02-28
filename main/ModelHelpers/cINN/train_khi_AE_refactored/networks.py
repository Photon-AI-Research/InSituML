import torch
import torch.nn as nn
from .encoder_decoder import Encoder
from .encoder_decoder import MLPDecoder, Conv3DDecoder
from .utilities import sample_gaussian, kl_normal, inspect_and_select

# property_ to input_dim
P2ID = { 
    "positions":3,
    "momentum":3,
    "force":3,
    "momentum_force":6,
    "all":9
    }
# Define the convolutional autoencoder class
@inspect_and_select
class ConvAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, encoder_kwargs, 
                 decoder_kwargs, loss_function):
        super().__init__()
        
        self.loss_function = loss_function
        # Encoder
        self.encoder = encoder(**encoder_kwargs)
        # Decoder
        self.decoder = decoder(**decoder_kwargs)

    def forward(self, x):
        y = self.reconstruct_input(x)
        loss = self.loss_function(y,x)
        return loss, y
    
    def reconstruct_input(self, x):
        #z is the latent space.
        z = self.encoder(x)
        y = self.decoder(z)
        return y

@inspect_and_select
class VAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_kwargs, 
                 decoder_kwargs, loss_function = None, 
                 property_="positions", z_dim=4,
                 particles_to_sample = 4000,
                 ae_config = "deterministic",
                 use_encoding_in_decoder=True
                 ):
        super().__init__()
        
        self.input_dim = P2ID[property_]
        self.particles_to_sample = particles_to_sample
        self.z_dim = z_dim
        self.loss_function = loss_function
        
        self.ae_config = ae_config
        
        self.use_encoding_in_decoder = use_encoding_in_decoder
        
        self.check_kwargs(encoder_kwargs, "encoder")
        self.check_kwargs(decoder_kwargs, "decoder")
        
        if ae_config=="non_deterministic" and use_encoding_in_decoder:
            decoder_kwargs["z_dim"] = 2*z_dim
            
        self.encoder = encoder(**encoder_kwargs)
        self.decoder = decoder(**decoder_kwargs)
        
        #set prior parameters of the vae model p(z) with 0 mean and 1 variance.
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.type = 'VAE'

    def check_kwargs(self, kwargs, type_):
        for val in ["z_dim", "ae_config", "input_dim", "particles_to_sample"]:
            network_val = getattr(self, val)
            if val not in kwargs:
                kwargs[val] = network_val
                
            elif network_val != kwargs[val]:
                raise ValueError(f"The {val} for {type_} does not match with network:{network_val} not equal {kwargs[val]}")
    
    def forward(self, inputs):
        x = inputs
        m, v = self.encoder(x)
        if self.ae_config == "deterministic":
            y = self.decoder(m)
            kl_loss = torch.zeros(1)
        elif self.ae_config == "non_deterministic":
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
        return nelbo, y, z

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
        if self.ae_config =="deterministic":
            y = self.decoder(m)
        else:
            z =  sample_gaussian(m,v)
            decoder_input = z if not self.use_encoding_in_decoder else \
            torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
            y = self.decoder(decoder_input)
        return y
