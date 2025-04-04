from threading import Thread
import torch
from .cl_memory import ExperienceReplay
from random import sample
import os
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD


class RadiationDataWriter:

    def __init__(self, dirpath, rank):

        self.dirpath = dirpath

        self.rank = comm.Get_rank()

        os.makedirs(self.dirpath, exist_ok=True)

        self.timestep = 0
        # First timestep must only gather data
        self.start_write = False

    def write(self):

        filename = self.dirpath + "/ts_" + str(self.timestep) + ".npy"

        self.request.Wait()
        np.save(filename, self.data_gathered)

    def __call__(self, data):

        if self.rank == 0 and self.start_write:
            self.write()

        self.data_gathered = None
        if self.rank == 0:
            self.data_gathered = np.zeros(
                (comm.size,) + data.shape, dtype=data.dtype
            )

        self.data = data

        self.request = comm.Igather(self.data, self.data_gathered)

        self.start_write = True

        self.timestep += 1

    def __del__(self):
        if self.rank == 0 and self.start_write:
            self.write()


class TrainBatchBuffer(Thread):
    """
    This class creates a ring buffer where oldest enteries produced
    openPMDProducer are either discarded or sent to Continual Learning
    based ExperienceReplay memory buffer.

    Args:

    openPMDBuffer (Queue): Queue shared between openPMD producer
    and train buffer.

    training_bs (int): training batch size to send to the model to train on.

    buffersize (int): Size of train buffer.

    max_tb_from_unchanged_now_bf (int): Maximum number of training batches
    that can be extracted from the unchanged
    state of train(or now) buffer. After extracting these many batches
    trainer would wait for more data to read from
    openPMDBuffer. State of train/now buffer only changes once there is
    more data to be read from the openPMDBuffer.

    use_continual_learning (Bool): Whether to use use continual learning or not.
      If yes, will create memory buffer for the continual learning.

    cl_mem_size (int): Continual learning memory buffer size.

    do_tranpose (bool): Whether to do the transpose of particle data or not.
    It depends if the producer
    produces (number_of_particles, particle_dims) or the transposed of this.
    And the model or trainer requires.

    """

    def __init__(
        self,
        openPMDBuffer,
        training_bs=4,
        buffersize=5,
        max_tb_from_unchanged_now_bf=3,
        min_tb_from_unchanged_now_bf=0,
        continual_bs=4,
        cl_mem_size=2048,
        do_tranpose=True,
        stall_loader=False,
        verbose=False,
        consume_size=None,
        radiation_data_folder=None,
    ):

        Thread.__init__(self)

        self.openPMDbuffer = openPMDBuffer
        self.do_tranpose = do_tranpose
        self.training_bs = training_bs
        self.continual_bs = continual_bs

        self.rank = comm.Get_rank()
        if radiation_data_folder:
            self.radiation_data_writer = RadiationDataWriter(
                radiation_data_folder, self.rank
            )
        else:
            self.radiation_data_writer = None

        if consume_size is None:
            self.consume_size = training_bs
        else:  # number of items consumed from the loaded is, in general,
            #    independent of batch size
            self.consume_size = consume_size

        self.use_continual_learning = self.continual_bs > 0
        # continual learning related required variables
        if self.use_continual_learning:
            self.er_mem = ExperienceReplay(mem_size=cl_mem_size)
            self.n_obs = 0

        self.i_step = 0

        self.buffer_ = []
        self.buffersize = buffersize

        # only added so that thread is
        # not run by default thread runner.
        self.run_thread = False

        self.stall_loader = stall_loader
        self.verbose = verbose

        # to indicate whether there are
        # still production from openPMD production.
        self.openpmdProduction = True
        self.noReadCount = 0
        self.min_tb_from_unchanged_now_bf = min_tb_from_unchanged_now_bf
        self.max_tb_from_unchanged_now_bf = max(
            max_tb_from_unchanged_now_bf, min_tb_from_unchanged_now_bf
        )

        self.particles_radiation = []  # unpack buffer

    def get_data(self):
        """
        This is an extra failsafe to avoid this thread being blocked
        because of empty
        producer queue as batch creation can continue. Seconds to wait
        before openPMDBuffer.get() throws an empty
        exception.
        Reference as Queue.qsize() documentation:
        https://docs.python.org/3/library/queue.html#queue.Queue.qsize
        Return the approximate size of the queue. Note, qsize() > 0 doesn’t
        guarantee that a subsequent get()
        will not block..
        """

        try:
            particles_radiation = self.openPMDbuffer.get(block=False)
            return particles_radiation
        except Exception as ex:
            print(f"Exception {ex} was raised")
            return False

    def run(self):

        if not self.run_thread or not self.openpmdProduction:
            return

        openPMDBufferReadCount = 0
        openPMDBufferSize = self.openPMDbuffer.qsize()

        updating = False
        if openPMDBufferSize:
            updating = True
            if self.verbose:
                print("Updating the train buffer")

        while openPMDBufferReadCount < min(
            self.consume_size, openPMDBufferSize
        ):
            # This condition can discard items left in particles_radiation even
            # openPMDbuffer does not have enough items to deliver consume_size,
            # but this case has no relevance, because when we srem and not
            # stall the producer items will not be bunched.
            if (not self.stall_loader and openPMDBufferSize > 0) or len(
                self.particles_radiation
            ) == 0:
                # get a particles, radiation from the queue
                particles_radiation = self.get_data()

                if particles_radiation is None:
                    self.openpmdProduction = False
                    break
                elif not particles_radiation:
                    break

                if self.radiation_data_writer is not None:
                    self.radiation_data_writer(particles_radiation[1].numpy())

                # in case items are bunched-up by the producer,
                # we keep superfluous ones for the next round
                self.particles_radiation = self.reshape(particles_radiation)
                if self.verbose:
                    print(
                        "##TrainBatchBuffer## particles_radiation.reshape",
                        len(self.particles_radiation),
                    )

            itemsToTake = min(
                self.consume_size - openPMDBufferReadCount,
                len(self.particles_radiation),
            )
            itemsToSchedule = len(self.buffer_) + itemsToTake - self.buffersize
            if itemsToSchedule > 0:
                # extracts the first elements.
                last_elements = self.buffer_[:itemsToSchedule]
                self.buffer_ = self.buffer_[itemsToSchedule:]

                if self.use_continual_learning:
                    # add the last element to memory, if continual learning is
                    # required.
                    X = [ele[0] for ele in last_elements]
                    Y = [ele[1] for ele in last_elements]

                    self.er_mem.update_memory(
                        X, Y, n_obs=self.n_obs, i_step=self.i_step
                    )

                    self.n_obs += len(last_elements)
                    self.i_step += 1

            self.buffer_ += self.particles_radiation[:itemsToTake]
            self.particles_radiation = self.particles_radiation[itemsToTake:]

            openPMDBufferReadCount += itemsToTake
            self.noReadCount = 0

        else:
            self.noReadCount += 1

        if self.verbose or self.rank == 0:
            print(
                "##TrainBatchBuffer## openPMDBufferReadCount",
                openPMDBufferReadCount,
            )

        self.run_thread = False

        if updating and self.verbose:
            print("Train Buffer Updated")

    def reshape(self, particles_radiation):
        # reshapes from gpu box indices to buffer
        # (gpu_box, number_of_particles, dims) ->
        # (number_of_particles_box_1, dims_box_1,
        #  number_of_particles_box_2, dims_box_2..)
        particles, radiation = particles_radiation

        if self.do_tranpose:
            particles_radiation = [
                [particles[idx].permute(1, 0), radiation[idx]]
                for idx in range(len(particles))
            ]
        else:
            particles_radiation = [
                [particles[idx], radiation[idx]]
                for idx in range(len(particles))
            ]

        return particles_radiation

    def get_batch(self):
        if self.verbose:
            print("Attempting a batch extraction from train buffer")

        self.run_thread = True
        # No training until there batch size element in the buffer.
        if len(self.buffer_) < self.buffersize or (
            self.noReadCount >= self.min_tb_from_unchanged_now_bf
            and self.openpmdProduction
        ):
            self.run()
        else:
            self.noReadCount += 1
        # No training until there batch size element in the buffer.
        if len(self.buffer_) < self.training_bs or (
            self.noReadCount > self.max_tb_from_unchanged_now_bf
            and self.openpmdProduction
        ):
            if self.verbose or self.rank == 0:
                print(
                    "Batch extraction failed.. \n"
                    + "Either train buffer has less element "
                    + "than training size \n"
                    f"Train Buffer Size: {len(self.buffer_)}, "
                    + f"training batch size: {self.training_bs} \n"
                    + "Or maximum number batches have extracted from "
                    + "unmodified train buffer state. Maximum train batches "
                    + "allowed from unchanged trainbuffer state: "
                    + f"{self.max_tb_from_unchanged_now_bf}\n"
                )
            return None

        # random sampling
        random_sample = sample(self.buffer_, self.training_bs)

        particles_batch = torch.stack([x[0] for x in random_sample])
        radiation_batch = torch.stack([x[1] for x in random_sample])

        if self.use_continual_learning and self.n_obs >= self.continual_bs:
            # sample from memory
            mem_part_batch, mem_rad_batch = self.er_mem.sample(
                self.continual_bs
            )
            particles_batch = torch.cat([particles_batch, mem_part_batch])
            radiation_batch = torch.cat([radiation_batch, mem_rad_batch])

        return particles_batch, radiation_batch
