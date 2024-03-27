from os import environ

modelPathPattern = 'trained_models/{}'

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 4000


streamLoader_config = dict(
    t0 =  890,
    t1 = 900, # endpoint=false, t1 is not used in training
    # t0 =  1800,
    # t1 = 1810, # endpoint=false, t1 is not used in training
    streaming_config = None,
    pathpattern1 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/014_KHI_007_noWindowFunction/simOutput/openPMD/simData_%T.bp", # files on hemera
    pathpattern2 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/014_KHI_007_noWindowFunction/simOutput/radiationOpenPMD/e_radAmplitudes%T.bp", # files on hemera
    # pathpattern1 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/008_KHI_rad4dir_smallY_highRes/simOutput/openPMD/simData_%T.bp", # files on hemera
    # pathpattern2 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/008_KHI_rad4dir_smallY_highRes/simOutput/radiationOpenPMD/e_radAmplitudes%T.bp", # files on hemera
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu = 30000,
    ## offline training params
    num_epochs = 2
)

openPMD_queue_size=8

batch_size=2

trainBatchBuffer_config = dict(
    training_bs=batch_size,
    continual_bs=batch_size,
    stall_loader=True,
    consume_size=batch_size,
    #Train buffer.
    buffersize = 10,
    #long buffer
    cl_mem_size = 100,
)
modelTrainer_config = dict(
    checkpoint_interval = 32,
    checkpoint_final = False,
    out_prefix = "slurm-{}/".format(environ["SLURM_JOBID"]) if "SLURM_JOBID" in environ else ""
)

runner="mpirun"
type_streamer="streaming"
