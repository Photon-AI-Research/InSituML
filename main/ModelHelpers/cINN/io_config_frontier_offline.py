import pathlib
from os import environ

modelPathPattern = str(pathlib.Path(__file__).parent.resolve()) + '/trained_models/{}'

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6 # Actually used in the model configuration by now
            # ToDo: Use in StreamingLoader

number_of_particles = 4096

streamLoader_config = dict(
    t0 = 900,
    t1 = 1001, # endpoint=false, t1 is not used in training
    streaming_config = None, # set to None when reading from file
    pathpattern1 = "/lustre/orion/csc380/world-shared/ksteinig/008_KHI_withRad_randomInit_8gpus/simOutput/openPMD/simData_%T.bp5", # files for 8GPUs on frontier
    pathpattern2 = "/lustre/orion/csc380/world-shared/ksteinig/008_KHI_withRad_randomInit_8gpus/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5", # files for 8GPUs on frontier
    amplitude_direction=0, # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables = ["momentum", "force"], # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu = 30000,
    verbose = False
    ## offline training params
    num_epochs = .25
)

openPMD_queue_size=4

batch_size=4
trainBatchBuffer_config = dict(
    training_bs=batch_size,
    continual_bs=batch_size-1,
    stall_loader=True,
    consume_size=1,
    min_tb_from_unchanged_now_bf = int(environ["MIN_TB_FROM_UNCHANGED_NOW_BF"]) if ("MIN_TB_FROM_UNCHANGED_NOW_BF" in environ) else 4,
    radiation_data_folder = "streamedRadiation", # fixed
    #Train buffer.
    buffersize = 10,
    #long buffer
    cl_mem_size = 20*32*3 ## NOTE: *3 because the data on fonrtier is a 24-node vs 8 node sim, to keep 20% of data
)

modelTrainer_config = dict(
    checkpoint_interval = 1000,
    checkpoint_final = True,
)

runner = "srun"
type_streamer = "streaming"

