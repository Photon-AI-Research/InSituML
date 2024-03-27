from math import sqrt

#########################
## Model configuration ##
#########################

rad_dims = 512 # Number of frequencies in radiation data

latent_space_dims = 544

config = dict(
dim_input = 1024,
dim_condition = rad_dims,
num_coupling_layers = 4,
hidden_size = 256,
num_blocks_mat = 6,
activation = 'gelu',
lr = 0.001,
y_noise_scale = 1e-1,
zeros_noise_scale = 5e-2,
lambd_predict = 3.,
lambd_latent = 300.,
lambd_rev = 400.,
lambd_kl = 0.001,
lambd_AE = 1.0,
lambd_IM = 0.001,
ndim_tot = 544,
ndim_x = 544,
ndim_y = 512,
ndim_z = 32,
load_model = None, #'inn_vae_latent_544_sim007_24k0zbm4/best_model_',
load_model_checkpoint = 'inn_vae_latent_544_sim014_859eopan/model_150', #'inn_vae_latent_544_sim014_859eopan/model_950',
    
## for optimized chamfer distance
# loss_function = 'chamfersloss_o',
# loss_kwargs = {'property_':'momentum_force'},

## for emd without peops library.
loss_function = 'earthmovers',
loss_kwargs = {},

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
