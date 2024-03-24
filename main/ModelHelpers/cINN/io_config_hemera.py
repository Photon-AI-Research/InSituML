modelPathPattern = 'trained_models/{}/best_model_'
import numpy as np

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
    t0 =  890,
    t1 = 900, # endpoint=false, t1 is not used in training
    # t0 =  1800,
    # t1 = 1810, # endpoint=false, t1 is not used in training
    streaming_config = None,
    pathpattern1 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/007_KHI_rad4dir_smallY/simOutput/openPMD/simData_%T.bp", # files on hemera
    pathpattern2 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/007_KHI_rad4dir_smallY/simOutput/radiationOpenPMD/e_radAmplitudes%T.bp", # files on hemera
    # pathpattern1 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/008_KHI_rad4dir_smallY_highRes/simOutput/openPMD/simData_%T.bp", # files on hemera
    # pathpattern2 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/008_KHI_rad4dir_smallY_highRes/simOutput/radiationOpenPMD/e_radAmplitudes%T.bp", # files on hemera
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    normalization = normalization_values,
    number_particles_per_gpu = 1000,
    ## offline training params
    num_epochs = 2
)

openPMD_queue_size=8

trainBatchBuffer_config = dict(
    continual_bs=4,
    stall_loader=True,
    consume_size=4,
)

runner="mpirun"
type_streamer="streaming"
