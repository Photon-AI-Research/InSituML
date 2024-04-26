from math import sqrt
from os import environ
import pathlib

#########################
## Model configuration ##
#########################

rad_dims = 512 # Number of frequencies in radiation data

latent_space_dims = 544

l_predict = environ.get("LAM_PREDICT", None)
l_latent = environ.get("LAM_LATENT", None)
l_rev = environ.get("LAM_REV", None)
l_kl = environ.get("LAM_KL", None)
l_recon = environ.get("LAM_RECON", None)

lambd_IM = 0.001
lambd_predict = ( 3. if l_predict is None else float(l_predict) ) * lambd_IM
lambd_latent = ( 300. if l_latent is None else float(l_latent) ) * lambd_IM
lambd_rev = ( 400. if l_rev is None else float(l_rev) ) * lambd_IM
lambd_AE = 1.0 if l_recon is None else float(l_recon)
lambd_kl = ( 0.001 if l_kl is None else float(l_kl) ) / lambd_AE
lambd_IM = 1

lr = float(environ.get("LR_REST", 0.0001))
lr_ae = float(environ.get("LR_AE", 0.0005))

config = dict(
dim_input = 1024,
dim_condition = rad_dims,
num_coupling_layers = 4,
hidden_size = 256,
num_blocks_mat = 6,
activation = 'gelu',
lr = lr,
lrAEmult = (lr_ae / lr),
y_noise_scale = 1e-1,
zeros_noise_scale = 5e-2,
lambd_predict = lambd_predict,
lambd_latent = lambd_latent,
lambd_rev = lambd_rev,
lambd_kl = lambd_kl,
lambd_AE = lambd_AE,
lambd_IM = lambd_IM,
ndim_tot = 544,
ndim_x = 544,
ndim_y = 512,
ndim_z = 32,
load_model = None, #'inn_vae_latent_544_sim007_24k0zbm4/best_model_',
load_model_checkpoint = None, #'inn_vae_latent_544_sim014_859eopan/model_150', #'inn_vae_latent_544_sim014_859eopan/model_950',
    
#   "earthmovers"
#   "chamfersloss"
#   "chamfersloss_d"
#   "chamfersloss_o"
## for optimized chamfer distance
loss_function = 'chamfersloss',
loss_kwargs = {},

## for emd without peops library.
# loss_function = 'earthmovers',
# loss_kwargs = {},

betas = (0.8, 0.9),
eps = 1e-6,
weight_decay = 2e-5,
lr_annealingRate = None,
lr_scaling = ( lambda x : sqrt(x) )
)

config_inn = dict(

)

normalization_values = dict(
    momentum_mean = 1.2091940752668797e-08,
    momentum_std = 0.11923234769525472,
    force_mean = -2.7682006649827533e-09,
    force_std = 7.705477610810592e-05
)
