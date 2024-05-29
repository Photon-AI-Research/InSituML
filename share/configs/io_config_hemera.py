from os import environ

#######################################
# openPMD data loader configuration #
#######################################
ps_dims = 6  # Actually used in the model configuration by now
# ToDo: Use in StreamingLoader

number_of_particles = 4000


streamLoader_config = dict(
    t0=900,
    t1=998,
    # t0 =  1800,
    # t1 = 1810,
    streaming_config=None,
    # pathpattern1 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/
    # 014_KHI_007_noWindowFunction/simOutput/openPMD/simData_%T.bp",
    # files on hemera
    # pathpattern2 = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/
    # 014_KHI_007_noWindowFunction/simOutput/radiationOpenPMD/
    # e_radAmplitudes%T.bp", # files on hemera
    particle_pathpattern=(
        "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
        + "24-nodes_full-picongpu-data/04-01_1013/simOutput/openPMD/"
        + "simData_%T.bp5"
    ),  # files for 96GCDs on hemera
    radiation_pathpattern=(
        "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/"
        + "24-nodes_full-picongpu-data/04-01_1013/simOutput/"
        + "radiationOpenPMD/e_radAmplitudes_%T.bp5"
    ),
    # files for 96GCDs on hemera
    amplitude_direction=0,  # choose single direction along which the radiation
    # signal is observed, max: N_observer-1, where N_observer is defined in
    # PIConGPU's radiation plugin
    phase_space_variables=[
        "momentum",
        "force",
    ],  # allowed are "position", "momentum", and "force". If "force" is set,
    #     "momentum" needs to be set too.
    number_particles_per_gpu=30000,
    verbose=False,
    # offline training params
    num_epochs=2,
)

openPMD_queue_size = 8

batch_size = int(environ["BATCH_SIZE"]) if "BATCH_SIZE" in environ else 4

out_prefix = (
    "slurm-{}/".format(environ["SLURM_JOBID"])
    if "SLURM_JOBID" in environ
    else ""
)

trainBatchBuffer_config = dict(
    training_bs=batch_size,
    continual_bs=batch_size
    - 1,  # 7 is the max we can fit on P100 with our stupid chamfer's impl
    stall_loader=True,
    consume_size=1,
    min_tb_from_unchanged_now_bf=4,
    radiation_data_folder=out_prefix,
    # Train buffer.
    buffersize=10,
    # long buffer
    cl_mem_size=20
    * 32
    * 3,  # 20% of data, but all:1, so 32 blocks go to one rank
)
modelTrainer_config = dict(
    checkpoint_interval=800, checkpoint_final=True, out_prefix=out_prefix
)

runner = "mpirun"
type_streamer = "streaming"
