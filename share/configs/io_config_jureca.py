from os import environ

modelPathPattern = 'trained_models/{}'

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 4000


streamLoader_config = dict(
    t0 =  900,
    t1 = 901, # endpoint=false, t1 is not used in training
    # t0 =  1800,
    # t1 = 1810, # endpoint=false, t1 is not used in training
    streaming_config = None,
    pathpattern1 = "/p/scratch/training2406/team_hechtlab_kelling/04-01_1013/simOutput/openPMD/simData_%T.bp5", # files on hemera
    pathpattern2 = "/p/scratch/training2406/team_hechtlab_kelling/04-01_1013/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files on hemera
    amplitude_direction=2, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu = 30000,
    verbose=False,
    ## offline training params
    num_epochs = 2
)

openPMD_queue_size=8

batch_size=int(environ["BATCH_SIZE"]) if "BATCH_SIZE" in environ else 4

trainBatchBuffer_config = dict(
    training_bs=batch_size,
    continual_bs=batch_size-1, # 7 is the max we can fit on P100 with our stupid chamfer's impl
    stall_loader=True,
    consume_size=1,
    min_tb_from_unchanged_now_bf = 4,
    #Train buffer.
    buffersize = 10,
    #long buffer
    cl_mem_size = 20*32*3, # 20% of data, but all:1, so 32 blocks go to one rank
)
modelTrainer_config = dict(
    checkpoint_interval = 1000,
    checkpoint_final = True,
    out_prefix = "slurm-{}-{}-{}/".format(environ.get("SLURM_JOBID", ""), environ.get("SLURM_ARRAY_TASK_ID", ""), environ.get("TASK_ID", 0))
)

runner="mpirun"
type_streamer="streaming"
