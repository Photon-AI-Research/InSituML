import time
import os
import argparse
import pathlib
import torch
from queue import Queue
import torch.multiprocessing as mp
import torch.distributed as dist

from inSituML.ac_train_batch_buffer import TrainBatchBuffer
from inSituML.ac_consumer_trainer import ModelTrainer
from inSituML.dummy_openpmd_producer import DummyOpenPMDProducer
import inSituML.dtensor as dtensor
from models.model_factory import load_objects, get_world_size
import importlib.util


def config_import(name, arg_config):
    """Import configuration from a Python file."""
    spec = importlib.util.spec_from_file_location(name, arg_config)
    config = importlib.util.module_from_spec(spec)
    return spec, config


def main():
    parser = argparse.ArgumentParser(
        description="""For running openPMDproduction based trainings.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    configs_path = os.path.join(str(pathlib.Path(__file__).parent.resolve()),
                                "../share/configs/")

    parser.add_argument(
        "--runner",
        type=str,
        default=None,
        help=(
            "Which runner type is in use: srun (frontier),"
            + " mpirun, torchrun. Overrides seting in io_config."
        ),
    )

    parser.add_argument(
        "--type_streamer",
        type=str,
        default=None,
        help=(
            "Which type of streamer to produce: streaming,"
            + " offline, dummy. Overrides seting in io_config."
        ),
    )

    parser.add_argument('--io_config',
                        type=str,
                        default=configs_path + '/io_config.py',
                        help="IO/streaming/data/paths -related config")

    parser.add_argument('--model_config',
                        type=str,
                        default=configs_path + '/model_config.py',
                        help="model config")

    args = parser.parse_args()


    # Import configurations
    spec, io_config = config_import("io_config", args.io_config)
    spec.loader.exec_module(io_config)
    spec, model_config = config_import("model_config", args.model_config)
    spec.loader.exec_module(model_config)
    
    if args.runner is None:
        args.runner = io_config.runner

    if args.type_streamer is None:
        args.type_streamer = io_config.type_streamer

    if "training_bs" not in io_config.trainBatchBuffer_config:
        io_config.trainBatchBuffer_config["training_bs"] = 4

    # Buffer shared between openPMD data loader and model
    openPMDBuffer = Queue(io_config.openPMD_queue_size)

    # nomraliztion values loaded from model_config,
    # because they are related to the pre-trained model
    streamLoader_config = io_config.streamLoader_config
    streamLoader_config["normalization"] = model_config.normalization_values

    config = model_config.config

    world_size = get_world_size()

    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def run_copies(rank=None, world_size=world_size, runner=None):

        if runner == "torchrun":
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            print(
                f"Start running basic DDP example on rank {rank}.", flush=True
            )
            # create model and move it to GPU with id rank
            rank = rank % torch.cuda.device_count()

        elif runner == "mpirun":

            rank = int(os.environ["OMPI_COMM_WORLD_NODE_RANK"])
            if torch.cuda.device_count() == 1:
                rank = 0
            global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            print(f"ranks {global_rank} ({rank}) / {world_size}", flush=True)

            dist.init_process_group(
                backend="nccl", world_size=world_size, rank=global_rank
            )
            dtensor.init_global_device_mesh()
            print(f"Initiated DDP GPU {rank}", flush=True)

        elif runner == "srun":

            rank = 0

            global_rank = int(os.environ["SLURM_PROCID"])

            dist.init_process_group(
                backend="nccl", world_size=world_size, rank=global_rank
            )
            dtensor.init_global_device_mesh()
            print(
                f"Initiated DDP GPU {rank}, global_rank {global_rank}",
                flush=True,
            )
        else:
            setup(rank, world_size)

        optimizer, scheduler, model = load_objects(rank, io_config, model_config, world_size)

        if args.type_streamer == "streaming":

            from inSituML.ks_transform_policies import (
                AbsoluteSquare,
                BoxesAttributesParticles,
            )

            # from ks_producer_openPMD_streaming
            # import StreamLoaderExceptionCatcher as StreamLoader
            from inSituML.ks_producer_openPMD_streaming import StreamLoader

            from inSituML.LoaderExceptionHandler import (
                wrapLoaderWithExceptionHandler,
            )

            Loader = wrapLoaderWithExceptionHandler(StreamLoader)

            particleDataTransformationPolicy = BoxesAttributesParticles()
            # returns particle data of shape
            # (local ranks, number_of_particles, ps_dims)

            # particleDataTransformationPolicy = ParticlesAttributes()
            # returns particle data of shape (number_of_particles, ps_dims)

            # radiationDataTransformationPolicy =
            #                           PerpendicularAbsoluteAndPhase()
            # returns radiation data of shape (local ranks, frequencies)
            radiationDataTransformationPolicy = AbsoluteSquare()
            # returns radiation data of shape (local ranks, frequencies)
            # radiationDataTransformationPolicy = AbsoluteSquareSumRanks()
            # returns radiation data of shape (frequencies)

            timeBatchLoader = Loader(
                openPMDBuffer,
                streamLoader_config,
                particleDataTransformationPolicy,
                radiationDataTransformationPolicy,
            )  # Streaming ready
        elif args.type_streamer == "offline":

            from inSituML.ks_transform_policies import (
                AbsoluteSquare,
                BoxesAttributesParticles,
            )
            from inSituML.ks_producer_openPMD import RandomLoader

            from inSituML.LoaderExceptionHandler import (
                wrapLoaderWithExceptionHandler,
            )

            Loader = wrapLoaderWithExceptionHandler(RandomLoader)

            particleDataTransformationPolicy = BoxesAttributesParticles()
            radiationDataTransformationPolicy = (
                AbsoluteSquare()
            )  # returns radiation data of shape (local ranks, frequencies)

            timeBatchLoader = Loader(
                openPMDBuffer,
                streamLoader_config,
                particleDataTransformationPolicy,
                radiationDataTransformationPolicy,
            )  # Streaming ready
        else:
            timeBatchLoader = DummyOpenPMDProducer(openPMDBuffer)

        if dist.get_rank() == 0:
            # print some parameters
            print(
                "#Param streamLoader_config.amplitude_direction=",
                streamLoader_config["amplitude_direction"],
                flush=True,
            )
            print(
                "#Param streamLoader_config.particle_pathpattern=",
                streamLoader_config["particle_pathpattern"],
                flush=True,
            )
            for k in config:
                print("#Param config.{}=".format(k), config[k], flush=True)
            print("#Param type_streamer=", io_config.type_streamer, flush=True)
            print(
                "#Param trainBatchBuffer_config.cl_mem_size=",
                io_config.trainBatchBuffer_config["cl_mem_size"],
                flush=True,
            )
            print(
                "#Param trainBatchBuffer_config.consume_size=",
                io_config.trainBatchBuffer_config["consume_size"],
                flush=True,
            )
            print(
                "#Param trainBatchBuffer_config.training_bs=",
                io_config.trainBatchBuffer_config["training_bs"],
                flush=True,
            )
            print(
                "#Param trainBatchBuffer_config.continual_bs=",
                io_config.trainBatchBuffer_config["continual_bs"],
                flush=True,
            )
            print(
                "#Param trainBatchBuffer_config.min_tb_from_unchanged_now_bf=",
                io_config.trainBatchBuffer_config.get(
                    "min_tb_from_unchanged_now_bf", 0
                ),
                flush=True,
            )
            print(
                "#Param trainBatchBuffer_config.max_tb_from_unchanged_now_bf=",
                io_config.trainBatchBuffer_config.get(
                    "max_tb_from_unchanged_now_bf", 3
                ),
                flush=True,
            )

        # wandb_logger = WandbLogger(project="khi_public",
        #                            args=config,
        #                            entity='jeyhun')
        trainBF = TrainBatchBuffer(
            openPMDBuffer, **io_config.trainBatchBuffer_config
        )
        modelTrainer = ModelTrainer(
            trainBF,
            model,
            optimizer,
            scheduler,
            gpu_id=rank,
            **io_config.modelTrainer_config,
            logger=None,
        )

        ####################
        #  Start training  #
        ####################
        start_time = time.time()

        modelTrainer.start()
        timeBatchLoader.start()
        # tell the producer who is consuming, so it can check if the consumer
        # died and terminate in this case
        timeBatchLoader.consumer_thread = modelTrainer

        modelTrainer.join()
        print("Join model trainer", flush=True)

        timeBatchLoader.join()
        print("Join openPMD data loader", flush=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total elapsed time: {elapsed_time:.6f} seconds", flush=True)

    def run_demo(demo_fn, world_size):
        mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

    if args.runner not in ["torchrun", "mpirun", "srun"]:
        n_gpus = torch.cuda.device_count()
        assert (
            n_gpus >= 2
        ), f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        # run_demo(run_copies, world_size)
    else:
        run_copies(runner=args.runner)



if __name__ == "__main__":
    main()
