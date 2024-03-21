#########################
## Model configuration ##
#########################

rad_dims = 512 # Number of frequencies in radiation data

latent_space_dims = 1024

config = dict(
dim_input = 1024,
dim_condition = rad_dims,
num_coupling_layers = 4,
hidden_size = 256,
num_blocks_mat = 6,
activation = 'relu',
lr = 0.00001,
y_noise_scale = 1e-1,
zeros_noise_scale = 5e-2,
lambd_predict = 3.,
lambd_latent = 300.,
lambd_rev = 400.,
ndim_tot = 1024,
ndim_x = 1024,
ndim_y = 512,
ndim_z = 512,
load_model = '2vsik2of'
)

config_inn = dict(

)
