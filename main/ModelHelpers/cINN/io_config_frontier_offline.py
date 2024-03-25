import pathlib

modelPathPattern = str(pathlib.Path(__file__).parent.resolve()) + '/trained_models/{}/best_model_'

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 4000
checkpoint_interval = 5

streamLoader_config = dict(
    t0 = 890,
    t1 = 898, # endpoint=false, t1 is not used in training
    streaming_config = None, # set to None when reading from file
    pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/008_KHI_withRad_randomInit_8gpus/simOutput/openPMD/simData_%T.bp5", # files for 8GPUs on frontier
    pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/008_KHI_withRad_randomInit_8gpus/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files for 8GPUs on frontier
#    pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/016_KHI_withRad_randomInit_16gpus/simOutput/openPMD/simData_%T.bp5", # files for 16GPUs on frontier
#    pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/016_KHI_withRad_randomInit_16gpus/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files for 16GPUs on frontier
#    pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/openPMD/simData_%T.bp", # files for 32GPUs on frontier
#    pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset/radiationOpenPMD/e_radAmplitudes%T.bp", # files for 32GPUs  on frontier
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu = 4096,
    ## offline training params
    num_epochs = .25
)

openPMD_queue_size=4

trainBatchBuffer_config = dict(
    continual_bs=4,
    stall_loader=False,
    consume_size=4,
)

runner="srun"
type_streamer="offline"

