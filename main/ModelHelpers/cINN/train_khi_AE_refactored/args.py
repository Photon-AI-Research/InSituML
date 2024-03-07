import argparse

def get_args(parser=None):
    
    if parser is None:
        parser = argparse.ArgumentParser(
        description="""For running Autoencoders on khi data"""
    )
    
    parser.add_argument('--property_',
                        type=str,
                        default='positions',
                        help="Whether to train on positions, momentum, forces or all")

    parser.add_argument('--learning_rate',
                        type=float,
                        default='1e-3',
                        help="Set the learning rate")

    parser.add_argument('--z_dim',
                        type=int,
                        default='128',
                        help="Set the latent space dimensions")

    parser.add_argument('--num_epochs',
                        type=int,
                        default='5',
                        help="Number of epochs")
    
    parser.add_argument('--timebatchsize',
                        type=int,
                        default='4',
                        help="Set the timebatchsize")
    
    parser.add_argument('--particlebatchsize',
                        type=int,
                        default='4',
                        help="Set the particlebatchsize")
    
    parser.add_argument('--val_boxes',
                        type=str,
                        default='[19,5,3]',
                        help="Validation boxes")
    
    parser.add_argument('--lossfunction',
                        type=str,
                        default='chamfersloss',
                        help="Choose the loss function")

    parser.add_argument('--network',
                        type=str,
                        default='VAE',
                        help="Choose the loss function")
    
    parser.add_argument('--project_kw',
                        type=str,
                        default='',
                        help="Choose the project keyword for runs")
    
    parser.add_argument('--ae_config',
                        type=str,
                        default="deterministic",
                        help="Three choices for encoder config: simple, non_deterministic, or deterministic")

    parser.add_argument('--use_encoding_in_decoder',
                        type=bool,
                        default=False,
                        help="Whether to use encodings in the decoder or otherwise")

    parser.add_argument('--particles_to_sample',
                        type=int,
                        default=4000,
                        help="How many particles to sample.")

    parser.add_argument('--pathpattern1',
                        type=str,
                        default="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
                        help="Path to the particles data")

    parser.add_argument('--pathpattern2',
                        type=str,
                        default= "/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/radiation_ex_002/{}.npy",
                        help="Path to radiation data")

    parser.add_argument('--t0',
                        type=int,
                        default = 1000,
                        help="Start time step from the data")

    parser.add_argument('--t1',
                        type=int,
                        default = 2001,
                        help="Last time step from the data")

    parser.add_argument('--encoder_type',
                        type=str,
                        default = "encoder_simple",
                        help="Kind of Encoder")

    parser.add_argument('--encoder_kwargs',
                        type=str,
                        default = '{"z_dim":128,"input_dim":6,"ae_config":"deterministic"}',
                        help="Encoder keyword arguments")

    parser.add_argument('--decoder_type',
                        type=str,
                        default = "mlp_decoder",
                        help="Kind of Decoder")

    parser.add_argument('--decoder_kwargs',
                        type=str,
                        default = '{"z_dim":128, "particles_to_sample":4000, "input_dim":6}',
                        help="Decoder keyword arguments")
    
    
    return parser.parse_args()
