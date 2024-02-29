"""
Loader for *streamed* PIConGPU openPMD particle and radiation data to be used for insitu machine learning model training.
The data is put in a buffer provided during construction of the producer class.
The buffer is expected to be fillable by a `put()` method.
TODO: Furthermore, a policy is taken which describes the data structure that needs to be put in the buffer.
This policy actually performs the required transformation on the data, if required.
For example, it performs data normalization and layouts the data as requested for the training.

!!!!!!!
This file only works with the openPMD-api version installed from this branch:
    https://github.com/franzpoeschel/openPMD-api/tree/pic_env
!!!!!!!

Authors: Klaus Steiniger, Franz Poeschel
"""
from threading import Thread

import numpy as np
from torch import from_numpy as torch_from_numpy
from torch import cat as torch_cat
from torch import float32 as torch_float32
from torch import complex64 as torch_complex64
from torch import empty as torch_empty
from torch import zeros as torch_zeros
from torch import transpose as torch_transpose
from torch import angle as torch_angle
from torch import abs as torch_abs

from mpi4py import MPI

import openpmd_api as opmd

from ks_helperfuncs import *

from sys import stdout

def distribution_strategy(dataset_extent,
                          mpi_rank,
                          mpi_size,
                          strategy_identifier=None):
    import os
    import re
    if strategy_identifier is None or not strategy_identifier:
        if 'OPENPMD_CHUNK_DISTRIBUTION' in os.environ:
            strategy_identifier = os.environ[
                'OPENPMD_CHUNK_DISTRIBUTION'].lower()
        else:
            strategy_identifier = 'hostname_roundrobinofsourceranks_roundrobinofsourceranks'  # default
    match = re.search('hostname_(.*)_(.*)', strategy_identifier)
    if match is not None:
        inside_node = distribution_strategy(dataset_extent,
                                            mpi_rank,
                                            mpi_size,
                                            strategy_identifier=match.group(1))
        second_phase = distribution_strategy(
            dataset_extent,
            mpi_rank,
            mpi_size,
            strategy_identifier=match.group(2))
        return opmd.FromPartialStrategy(opmd.ByHostname(inside_node), second_phase)
    elif strategy_identifier == 'roundrobin':
        return opmd.RoundRobin()
    elif strategy_identifier == 'roundrobinofsourceranks':
        return opmd.RoundRobinOfSourceRanks()
    elif strategy_identifier == 'binpacking':
        return opmd.BinPacking()
    elif strategy_identifier == 'slicedataset':
        return opmd.ByCuboidSlice(opmd.OneDimensionalBlockSlicer(), dataset_extent,
                                mpi_rank, mpi_size)
    elif strategy_identifier == 'fail':
        return opmd.FailingStrategy()
    elif strategy_identifier == 'discard':
        return opmd.DiscardingStrategy()
    else:
        raise RuntimeError("Unknown distribution strategy: " +
                           strategy_identifier)

# Specify environment variable OPENPMD_CHUNK_DISTRIBUTION to modify the
# logic used in this function.
# Recommendation: set OPENPMD_CHUNK_DISTRIBUTION to
# `hostname_roundrobinofsourceranks_fail` to ensure failure when openPMD cannot
# figure out which data chunks are local to the current host.
# The rank table can be requested in openPMD by setting
# {"rank_table": "posix_hostname"} in the writer
#  (functionality not on the dev branch yet!).
# If the data does not define a rank table, a possible setting for the
# environment variable is `OPENPMD_CHUNK_DISTRIBUTION=slicedataset`.
def determine_local_region(record_component, comm, inranks, outranks,
                          strategy_identifier=None):
    if isinstance(strategy_identifier, opmd.Strategy):
        distribution = strategy_identifier
    else:
        distribution = distribution_strategy(
            record_component.shape, comm.rank, comm.size, strategy_identifier)
    all_chunks = record_component.available_chunks()
    # Little hack, the source_id might not be equivalent to the writing MPI rank due to
    # data aggregation in ADIOS2.
    # Since we know that each rank writes one chunk in PIConGPU,
    # we just assign the ranks explicitly here.
    all_chunks = opmd.ChunkTable([
        opmd.WrittenChunkInfo(offset=chunk.offset, extent=chunk.extent, rank=i) \
        for i, chunk in zip(range(len(all_chunks)), all_chunks)
    ])
    chunk_distribution = distribution.assign(all_chunks, inranks, outranks)
    res = dict()
    for target_rank, chunks in chunk_distribution.items():
        chunks = chunks.merge_chunks_from_same_sourceID()
        for source_rank, chunks_from_source_rank in chunks.items():
            if len(chunks_from_source_rank) != 1:
                raise RuntimeError(
                    "Need one contiguous slice of particles to load per source rank, got {} regions from rank {} instead.".format(
                        len(chunks_from_source_rank), source_rank
                    )
                )
        source_ranks = [source_rank for source_rank in chunks]
        all_chunks = opmd.ChunkTable(
            [opmd.WrittenChunkInfo(chunk.offset, chunk.extent, source_rank) \
                for source_rank, chunks_from_source_rank in chunks.items() \
                    for chunk in chunks_from_source_rank])

        res[target_rank] = all_chunks

    if comm.rank == 0:
        for target_rank, table in res.items():
            for chunk in table:
                print("Target rank {} loads from source ranks {}, {} - {}".format(
                    target_rank, chunk.source_id, chunk.offset, chunk.extent))
    return res

# Assign chunks from the same source ranks as in a previously calculated
# chunk distribution. Used to keep chunk distribution consistent between:
#
# * electron data
# * electron patches data
# * radiation data
class SelectAccordingToChunkDistribution(opmd.Strategy):
    def __init__(self, electrons_chunk_distribution):
        super().__init__()
        self.source_to_target = dict()
        for target, chunks in electrons_chunk_distribution.items():
            for chunk in chunks:
                self.source_to_target[chunk.source_id] = target

    def assign(self, chunks, *_):
        res = opmd.Assignment()
        for unassigned_chunk in chunks:
            # We could theoretically ignore the chunk if the target rank
            # is different from the current rank, but it's not a huge overhead
            # and it makes debugging simpler.
            target_rank = self.source_to_target[unassigned_chunk.source_id]
            if target_rank not in res:
                res[target_rank] = opmd.ChunkTable()
            res[target_rank].append(unassigned_chunk)
        return res

class StreamLoader(Thread):
    """ Thread providing PIConGPU particle and radiation data from an openPMD stream for the ML model training.

    There is not much happening here, except loading data from the stream, transforming in a layout as used in streaming, normalizing, and filling the shared buffer with the data.

    This is orchestrated in the run() method.
    """

    def __init__(self, batchDataBuffer, hyperParameterDefaults, particleDataTransformationPolicy=None, radiationDataTransformationPolicy=None):
        """ Set parameters of the loader

        Arguments:
            batchDataBuffer (e.g. queue.Queue) : buffer to put the data into (where the consumer reads it)
            hyperParameterDefaults (dict) : Defines timesteps, paths to data, etc.
            dataReadPolicy (functor) : Provides particle and radiation data per time step
            dataTransformationPolicy (functor) :
        """
        Thread.__init__(self)
        # instantiate all required parameters
        self.data = batchDataBuffer
        self.particlePathpattern = hyperParameterDefaults['pathpattern1']
        self.radiationPathPattern = hyperParameterDefaults['pathpattern2']

        self.rng = np.random.default_rng()
        self.particleTransformPolicy = particleDataTransformationPolicy
        self.radiationTransformPolicy = radiationDataTransformationPolicy
        self.comm = MPI.COMM_WORLD
        self.hyperParameterDefaults = hyperParameterDefaults

    def run(self):
        """Function being executed when thread is started."""
        # Open openPMD particle and radiation series
        openpmd_stream_config = """
            defer_iteration_parsing = true
            [adios2.engine.parameters]
            OpenTimeoutSecs = 300
        """

        # Open openPMD particle and radiation series
        series = opmd.Series(
            self.particlePathpattern,
            opmd.Access.read_linear,
            self.comm,
            openpmd_stream_config)
        radiationSeries = opmd.Series(
            self.radiationPathPattern,
            opmd.Access.read_linear,
            self.comm,
            openpmd_stream_config)

        # The streams wait until a reader connects.
        # To avoid deadlocks, we need to open both concurrently
        # (PIConGPU opens the streams one after the other)
        t1 = Thread(target=series.parse_base)
        t2 = Thread(target=radiationSeries.parse_base)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        inranks = series.get_rank_table(collective=True)
        if not inranks:
            import sys
            print("[WARNING] No chunk table found in data. Will map source to sink ranks somehow, but this might scale terribly in streaming setups.", file=sys.stderr)
        outranks = opmd.HostInfo.MPI_PROCESSOR_NAME.get_collective(self.comm)

        # We need to use the __iter__() and __next__() manually since both these
        # iterators need to be processed concurrently.
        # zip() might also work, but manual iteration is better since it gives
        # us greater control over when to wait for data (e.g. we can avoid
        # useless waiting)
        particle_iterations = series.read_iterations().__iter__()
        radiation_iterations = radiationSeries.read_iterations().__iter__()

        # start reading data
        while True:
            """Work on PIConGPU data for this iteration."""
            try:
                iteration = particle_iterations.__next__()
            except StopIteration:
                break

            def get_next_radiation():
                try:
                    radIter = radiation_iterations.__next__()
                except StopIteration:
                    raise RuntimeError(
                        "Streams getting out of sync? Particles at {}, but no further Radiation data available.".format(
                            iteration.iteration_index))

                if iteration.iteration_index != radIter.iteration_index:
                    raise RuntimeError(
                        "Iterations getting out of sync? Particles at {}, but Radiation at {}.".format(
                            iteration.iteration_index,
                            radIter.iteration_index))
                return radIter

            if not (self.hyperParameterDefaults['t0']
                    <= iteration.iteration_index
                    <= self.hyperParameterDefaults['t1']):
                print("Skipping iteration {} as it is not in the specified range [t0,t1]=[{},{}]".format(
                    iteration.iteration_index,
                    self.hyperParameterDefaults['t0'],
                    self.hyperParameterDefaults['t1']
                ))
                get_next_radiation()
                continue

            print("Start processing iteration %i"%(iteration.time))
            stdout.flush()
            ## obtain particle distribution over GPUs ##
            #
            ps = iteration.particles["e_all"] #particle species

            # memorize particle distribution over GPUs in array
            e_pos_x = ps["position"]["x"]
            chunk_distribution = determine_local_region(
                e_pos_x, self.comm, inranks, outranks
            )
            local_particles_chunks = chunk_distribution[self.comm.rank]
            num_processed_chunks_per_rank = len(local_particles_chunks)

            # Every GPU will hold a different number of particles.
            # But we need to keep the number of particles per GPU constant in order to construct the dataset.
            numParticles = np.array([chunk.extent[0] for chunk in local_particles_chunks])
            particlePerGPU = numParticles.min()
            print("particles per GPU", particlePerGPU)
            randomParticlesPerGPU = np.array([
                self.rng.choice(numParticles[i], particlePerGPU, replace=False) \
                    for i in range(num_processed_chunks_per_rank)])

            # Helper object to store enqueued load buffers.
            # Multiple buffers per component possible when processing data from
            # multiple GPUs in one (parallel) Python instance.
            class EnqueuedBuffers(object):
                def __init__(self):
                    self.position = []
                    self.positionOffset = []
                    self.momentum = []
                    self.momentumPrev1 = []

            # Workflow:
            #
            # 1. Enqueue all load operations on the particles stream and store
            #    the to-be-filled buffers in the loaded_buffers dict.
            # 2. Close the iteration with a single flush.
            #    a) There is only one communication back to the writer.
            #    b) After closing the iteration, the writer knows that the data
            #       is no longer needed, can discard the data and go on.
            # 3. Perform all further computations afterwards.
            #
            # This is a compromise: We use a slightly higher amount of host
            # memory in exchange for loading data once only.
            loaded_buffers = dict()
            for i_c, component in enumerate(["x", "y", "z"]):
                """Read particle data component-wise to reduce host memory usage.
                   And immediately reshape by subdividing in particles per GPU.
                   Also, reduce to requested number of particles per GPU.
                """
                component_buffers = EnqueuedBuffers()
                for chunk in local_particles_chunks:
                    component_buffers.position.append(
                        ps["position"][component].load_chunk(chunk.offset, chunk.extent))
                    component_buffers.positionOffset.append(
                        ps["positionOffset"][component].load_chunk(chunk.offset, chunk.extent))
                    component_buffers.momentum.append(
                        ps["momentum"][component].load_chunk(chunk.offset, chunk.extent))
                    component_buffers.momentumPrev1.append(
                        ps["momentumPrev1"][component].load_chunk(chunk.offset, chunk.extent))

                loaded_buffers[component] = component_buffers

            # Particle patches are needed further below for determining the GPU bounding box.
            # This only loads the patch information for the locally processed GPUs.
            # Do NOT load all of them as this will be a NxN communication pattern,
            # e.g. it will not scale (I'll talk to the ADIOS2 devs on how we can avoid this
            # situation in the future).
            # IF the entire patches info is needed after all, then use MPI to distribute this
            # local information to all ranks.
            gpuBoxExtent = dict()
            gpuBoxOffset = dict()
            patches_chunk_distribution = determine_local_region(
                ps.particle_patches["extent"]["x"], self.comm, inranks, outranks,
                SelectAccordingToChunkDistribution(chunk_distribution)
            )
            # import ipdb
            # ipdb.set_trace(context=30)
            local_patch_chunk = patches_chunk_distribution[self.comm.rank]
            local_patch_chunk.merge_chunks()
            if len(local_patch_chunk) != 1:
                raise RuntimeError("Patches: Need to load contiguous regions.")
            else:
                local_patch_chunk = local_patch_chunk[0]

            for component in ["x", "y", "z"]:
                gpuBoxExtent[component] = ps.particle_patches["extent"][component].load_chunk(
                    local_patch_chunk.offset, local_patch_chunk.extent
                )
                gpuBoxOffset[component] = ps.particle_patches["offset"][component].load_chunk(
                    local_patch_chunk.offset, local_patch_chunk.extent
                )
            numParticles = ps.particle_patches["numParticles"].load_chunk(
                local_patch_chunk.offset, local_patch_chunk.extent
            )
            numParticlesOffsets = ps.particle_patches["numParticlesOffset"].load_chunk(
                local_patch_chunk.offset, local_patch_chunk.extent
            )

            iteration.close() # trigger enqueued data loads

            # prepare torch tensor to hold particle data in shape (phaseSpaceComponents, GPUs, particlesPerGPU)
            loaded_particles = torch_empty((9, num_processed_chunks_per_rank, particlePerGPU), dtype=torch_float32)
            for i_c, component in enumerate(["x", "y", "z"]):
                component_buffers = loaded_buffers[component]
                position = [component_buffers.position[j] + component_buffers.positionOffset[j] \
                        for j in range(num_processed_chunks_per_rank)]
                # TODO: Normalize positions on per GPU basis!
                ## TODO: need to scale and normalize these values
                ## Is it required to normalize positions and other phase space componentes? YES, need to do this!
                del component_buffers.position
                del component_buffers.positionOffset
                pos_reduced = np.array([
                    p[r] for r, p in zip(randomParticlesPerGPU, position)
                ])
                del position
                mom_reduced = np.array([
                    m[r] for r, m in zip(randomParticlesPerGPU, component_buffers.momentum)
                ])
                del component_buffers.momentum
                momPrev1_reduced = np.array([
                    m[r] for r, m in zip(randomParticlesPerGPU, component_buffers.momentumPrev1)
                ])
                del component_buffers.momentumPrev1

                del component_buffers
                del loaded_buffers[component]

                loaded_particles[0+i_c] = torch_from_numpy(pos_reduced) # absolute position in cells
                loaded_particles[3+i_c] = torch_from_numpy(mom_reduced) # momentum in gamma*beta
                loaded_particles[6+i_c] = torch_from_numpy(mom_reduced - momPrev1_reduced) # force
                del pos_reduced
                del mom_reduced
                del momPrev1_reduced

            del loaded_buffers

            if self.particleTransformPolicy is not None:
                loaded_particles = self.particleTransformPolicy(loaded_particles)
            else:
                """transform data to shape (GPUs, phaseSpaceComponents, particlesPerGPU)"""
                loaded_particles = torch_transpose(loaded_particles, 0, 1)

            ## obtain radiation data per GPU ##
            #
            radIter = get_next_radiation()

            cellExtensionNames = {"x" : "cell_width", "y" : "cell_height", "z" : "cell_depth"}
            r_offset = np.empty((num_processed_chunks_per_rank, 3)) # shape: (local GPUs, components)
            n_vec = np.empty((radIter.meshes["DetectorDirection"]["x"].shape[0], 3)) # shape: (N_observer, components)

            DetectorFrequency = radIter.meshes["DetectorFrequency"]["omega"][0, :, 0] # shape: (frequencies)
            # radiationSeries.flush()

            distributed_amplitudes = torch_empty((3, num_processed_chunks_per_rank, len(DetectorFrequency)), dtype=torch_complex64) # shape: (components, local GPUs, frequencies)

            rad_chunk_distribution = determine_local_region(
                radIter.meshes["Amplitude_distributed"]["x"], self.comm, inranks, outranks,
                SelectAccordingToChunkDistribution(chunk_distribution)
            )
            local_radiation_chunk = rad_chunk_distribution[self.comm.rank]
            local_radiation_chunk.merge_chunks()
            if len(local_radiation_chunk) != 1:
                raise RuntimeError("Radiation: Need to load contiguous regions.")
            else:
                local_radiation_chunk = local_radiation_chunk[0]

            loaded_buffers = dict()
            for i_c, component in enumerate(["x", "y", "z"]):
                component_buffers = EnqueuedBuffers()
                # See (https://picongpu.readthedocs.io/en/latest/usage/plugins/radiation.html#openpmd-output)
                # for a description of the radiation plugin's output structure.
                component_buffers.Dist_Amplitude = \
                    radIter.meshes["Amplitude_distributed"][component].load_chunk(
                        local_radiation_chunk.offset,
                        local_radiation_chunk.extent) # shape: (GPUs, N_observer, frequencies)
                # read components of the observation direction vector n_vec
                component_buffers.DetectorDirection = radIter.meshes["DetectorDirection"][component][:, 0, 0] # shape: (N_observer)
                loaded_buffers[component] = component_buffers

            radIter.close()

            amplitude_direction = self.hyperParameterDefaults['amplitude_direction']

            for i_c, component in enumerate(["x", "y", "z"]):
                component_buffers = loaded_buffers[component]
                component_buffers.Dist_Amplitude *= radIter.meshes["Amplitude_distributed"][component].unit_SI
                component_buffers.DetectorDirection *= radIter.meshes["DetectorDirection"][component].unit_SI

                r_offset[:, i_c] = gpuBoxOffset[component] * iteration.get_attribute(cellExtensionNames[component])
                n_vec[:, i_c] = component_buffers.DetectorDirection
                #  reduce Dist_Amplitude data to single observation direction
                distributed_amplitudes[i_c] = torch_from_numpy(component_buffers.Dist_Amplitude[:, amplitude_direction, :]) # shape of component i_c: (local GPUs, frequencies)

            # time retardation correction
            # QUESTION: The `iteration.iteration_index`=int variable appears in here, not the actual time? (but r_offset is also in cells...)
            # ANSWER: Calculation is fully in PIConGPU coordinates. All fine.
            phase_offset = torch_from_numpy(
                np.exp(
                    -1.j * DetectorFrequency[np.newaxis, np.newaxis, :]
                    * (iteration.iteration_index - np.dot(r_offset, n_vec.T)[:, :, np.newaxis] / 1.0)))\
                    [:, amplitude_direction, :]
            distributed_amplitudes = distributed_amplitudes*phase_offset

            # Transform to shape: (GPUs, components, frequencies)
            distributed_amplitudes = torch_transpose(distributed_amplitudes, 0, 1) # shape: (local GPUs, components, frequencies)

            if self.radiationTransformPolicy is not None:
                distributed_amplitudes = self.radiationTransformPolicy(distributed_amplitudes)

            # Put particle and radiation data in shared buffer
            self.data.put([loaded_particles, distributed_amplitudes])

            print("Done loading iteration %i"%(iteration.time))
            stdout.flush()


        # signal that there are no further items
        print("Processed all iterations")
        stdout.flush()
        self.data.put(None)

        # close series
        series.close()
        radiationSeries.close()

