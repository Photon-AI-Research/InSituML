import pathlib
from os import environ

modelPathPattern = (
    str(pathlib.Path(__file__).parent.parent.resolve()) + "/trained_models/{}"
)

#######################################
## openPMD data loader configuration ##
#######################################
ps_dims = 6  # Actually used in the model configuration by now
# ToDo: Use in StreamingLoader

number_of_particles = 4096

streamLoader_config = dict(
    t0=900,
    t1=1001,  # endpoint=false, t1 is not used in training
    pathpattern1="openPMD/simData.sst",  # streaming on frontier
    pathpattern2="radiationOpenPMD/e_radAmplitudes.sst",  # streaming on frontier
    streaming_config="@inconfig.json",  # None, # set to None when reading from file
    amplitude_direction=0,  # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables=[
        "momentum",
        "force",
    ],  # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu=30000,
    verbose=False,
)

openPMD_queue_size = 4

batch_size = 4

trainBatchBuffer_config = dict(
    training_bs=batch_size,
    continual_bs=batch_size,
    stall_loader=True,
    consume_size=1,
    min_tb_from_unchanged_now_bf=(
        int(environ["MIN_TB_FROM_UNCHANGED_NOW_BF"])
        if ("MIN_TB_FROM_UNCHANGED_NOW_BF" in environ)
        else 16
    ),
    radiation_data_folder="streamedRadiation",  # fixed
    # Train buffer.
    buffersize=10,
    # long buffer
    cl_mem_size=20,
)


modelTrainer_config = dict(
    checkpoint_interval=350,
    checkpoint_final=True,
)

runner = "srun"
type_streamer = "streaming"
