modelPathPattern = 'trained_models/{}/best_model_'

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 100

normalization_values = dict(
    momentum_mean = 0.,
    momentum_std = 1.,
    force_mean = 0.,
    force_std = 1.,
)

streamLoader_config = dict(
    t0 = 890,
    t1 = 900, # endpoint=false, t1 is not used in training
    streaming_config = None,
    pathpattern1 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/openPMD/simData_%T.bp5", # files on hemera
    pathpattern2 = "/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files on hemera
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    normalization = normalization_values,
    number_particles_per_gpu = 1000
)
