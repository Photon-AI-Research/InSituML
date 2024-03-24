import pathlib

modelPathPattern = str(pathlib.Path(__file__).parent.resolve()) + '/trained_models/{}/best_model_'

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 100

data_stats = np.load('mean_std_007/global_stats_900_1001.npz')

normalization_values = dict(
    momentum_mean = data_stats['mean_momentum'],
    momentum_std = data_stats['std_momentum'],
    force_mean = data_stats['mean_force'],
    force_std = data_stats['std_force'],
)

streamLoader_config = dict(
    t0 = 890,
    t1 = 900, # endpoint=false, t1 is not used in training
    pathpattern1 = "openPMD/simData.sst", # streaming on frontier
    pathpattern2 = "radiationOpenPMD/e_radAmplitudes.sst", # streaming on frontier
    streaming_config = "@inconfig.json", # set to None when reading from file
    #pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/openPMD/simData_%T.bp", # files on frontier
    #pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/radiationOpenPMD/e_radAmplitudes%T.bp", # files on frontier
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    normalization = normalization_values,
    number_particles_per_gpu = 1000
    ## offline training params
    num_epochs = 1
)

openPMD_queue_size=4

trainBatchBuffer_config = dict(
    continual_bs=4,
    stall_loader=False,
    consume_size=4,
)

runner="srun"
type_streamer="streaming"
