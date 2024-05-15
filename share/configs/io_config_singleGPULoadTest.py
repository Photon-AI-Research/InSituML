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
    t0=890,
    t1=895,  # endpoint=false, t1 is not used in training
    # pathpattern1 = "openPMD/simData.sst", # streaming on frontier
    # pathpattern2 = "radiationOpenPMD/e_radAmplitudes.sst", # streaming on frontier
    streaming_config=None,  # "@inconfig.json", # set to None when reading from file
    pathpattern1="/lustre/orion/csc380/world-shared/ksteinig/016_KHI_withRad_randomInit_16gpus/simOutput/openPMD/simData_%T.bp5",  # files for 16GPUs on frontier
    pathpattern2="/lustre/orion/csc380/world-shared/ksteinig/016_KHI_withRad_randomInit_16gpus/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5",  # files for 16GPUs on frontier
    # pathpattern1 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/007_KHI_rad4dir_smallY/simOutput/openPMD/simData_%T.bp", # files on hemera
    # pathpattern2 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/007_KHI_rad4dir_smallY/simOutput/radiationOpenPMD/e_radAmplitudes%T.bp", # files on hemera
    amplitude_direction=0,  # choose single direction along which the radiation signal is observed, max: N_observer-1, where N_observer is defined in PIConGPU's radiation plugin
    phase_space_variables=[
        "momentum",
        "force",
    ],  # allowed are "position", "momentum", and "force". If "force" is set, "momentum" needs to be set too.
    number_particles_per_gpu=int(environ["NUM_PART_GPU"]),
    ## offline training params
    num_epochs=1,
)

openPMD_queue_size = 4

batch_size = int(environ["BATCH_SIZE"])

trainBatchBuffer_config = dict(
    training_bs=batch_size,
    continual_bs=batch_size,
    stall_loader=True,
    consume_size=batch_size,
)

modelTrainer_config = dict(
    out_prefix=(
        "BS-%s_PART-%s/" % (environ["BATCH_SIZE"], environ["NUM_PART_GPU"])
        if ("SLURM_JOBID" in environ) and ("NUM_PART_GPU" in environ)
        else ""
    )
)

# runner="mpirun"
runner = "srun"
type_streamer = "streaming"
