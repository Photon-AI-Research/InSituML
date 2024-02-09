"""
Loader for *streamed* PIConGPU openPMD particle and radiation data to be used for insitu machine learning model training.
The data is put in a buffer provided during construction of the producer class.
The buffer is expected to be fillable by a `put()` method.
TODO: Furthermore, a policy is taken which describes the data structure that needs to be put in the buffer.
This policy actually performs the required transformation on the data, if required.
For example, it performs data normalization and layouts the data as requested for the training.
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

import openpmd_api as opmd

from ks_helperfuncs import *

from sys import stdout

class StreamLoader(Thread):
    """ Thread providing PIConGPU particle and radiation data from an openPMD stream for the ML model training.

    There is not much happening here, except loading data from the stream, transforming in a layout as used in streaming, normalizing, and filling the shared buffer with the data.

    This is orchestrated in the run() method.
    """

    def __init__(self, batchDataBuffer, hyperParameterDefaults, dataTransformationPolicy=None):
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
#        self.particlePathpattern = hyperParameterDefaults["pathpattern1"],
        self.particlePathpattern = "/home/franzpoeschel/git-repos/streamed_analysis/pic_run/openPMD/simData_%T.bp5"
#        self.radiationPathPattern = hyperParameterDefaults["pathpattern2"],
        self.radiationPathPattern = "/home/franzpoeschel/git-repos/streamed_analysis/pic_run/radiationOpenPMD/e_radAmplitudes_%T.bp5"

        self.rng = np.random.default_rng()
        self.transformPolicy = None #dataTransformationPolicy

    def run(self):
        """Function being executed when thread is started."""
        # Open openPMD particle and radiation series
        series = opmd.Series(self.particlePathpattern, opmd.Access.read_only)
        radiationSeries = opmd.Series(self.radiationPathPattern, opmd.Access.read_only)

        # start reading data
        for iteration in series.read_iterations():
            """Work on PIConGPU data for this iteration."""
            print("Start processing iteration %i"%(iteration.time))
            stdout.flush()
            ## obtain particle distribution over GPUs ##
            #
            ps = iteration.particles["e_all"] #particle species

            numParticles = ps.particle_patches["numParticles"][opmd.Mesh_Record_Component.SCALAR].load()
            numParticlesOffsets = ps.particle_patches["numParticlesOffset"][opmd.Mesh_Record_Component.SCALAR].load()
            series.flush()

            totalParticles = np.sum(numParticles) # numParticlesOffsets[-1] + numParticles[-1]?

            # memorize particle distribution over GPUs in array
            local_region = {"offset": [numParticlesOffsets[0]], "extent": [totalParticles]}

            numParticlesOffsets = np.concatenate((numParticlesOffsets, [-1]), dtype=np.int64) # append to use array for indexing

            # Every GPU will hold a different number of particles.
            # But we need to keep the number of particles per GPU constant in order to construct the dataset.
            particlePerGPU = numParticles.min()
            print("particles per GPU", numParticles)
            print("choose particles", particlePerGPU)
            randomParticlesPerGPU = np.array([ self.rng.choice(numParticles[i], particlePerGPU, replace=False) for i in np.arange(len(numParticles))])

            # prepare torch tensor to hold particle data in shape (phaseSpaceComponents, GPUs, particlesPerGPU)
            loaded_particles = torch_empty((9, len(numParticles), particlePerGPU), dtype=torch_float32)
            for i_c, component in enumerate(["x", "y", "z"]):
                """Read particle data component-wise to reduce host memory usage.
                   And immediately reshape by subdividing in particles per GPU.
                   Also, reduce to requested number of particles per GPU.
                """
                position = (ps["position"][component].load_chunk(local_region["offset"], local_region["extent"])
                    + ps["positionOffset"][component].load_chunk(local_region["offset"], local_region["extent"])
                )
                pos_reduced = np.array([
                    position[numParticlesOffsets[i]:numParticles[i]][randomParticlesPerGPU[i]]
                    for i in np.arange(len(numParticles))
                ])

                ## TODO:
                ## Is it required to normalize positions and other phase space componentes? YES, need to do this!

                momentum = ps["momentum"][component].load_chunk(local_region["offset"], local_region["extent"])
                mom_reduced = np.array([
                    momentum[numParticlesOffsets[i]:numParticles[i]][randomParticlesPerGPU[i]]
                    for i in np.arange(len(numParticles))
                ])

                momentumPrev1 = ps["momentumPrev1"][component].load_chunk(local_region["offset"], local_region["extent"])
                momPrev1_reduced = np.array([
                    momentumPrev1[numParticlesOffsets[i]:numParticles[i]][randomParticlesPerGPU[i]]
                    for i in np.arange(len(numParticles))
                ])

                loaded_particles[0+i_c] = torch_from_numpy(pos_reduced) # absolute position in cells
                loaded_particles[3+i_c] = torch_from_numpy(mom_reduced) # momentum in gamma*beta
                loaded_particles[6+i_c] = torch_from_numpy(mom_reduced - momPrev1_reduced) # force

            if self.transformPolicy is not None:
                loaded_particles = self.transformPolicy(loaded_particles)
            else:
                """transform data to shape (GPUs, phaseSpaceComponents, particlesPerGPU)"""
                loaded_particles = torch_transpose(loaded_particles, 0, 1)

            ## obtain radiation data per GPU ##
            #
# TODO
# FRANZ: here we need to get the iteration from the radiation data stream
            radIter = radiationSeries.iterations[int(iteration.time)]

            cellExtensionNames = {"x" : "cell_width", "y" : "cell_height", "z" : "cell_depth"}
            r_offset = np.empty((len(numParticles), 3)) # shape: (GPUs, components)
            n_vec = np.empty((radIter.meshes["DetectorDirection"]["x"].shape[0], 3)) # shape: (radiation measurement directions along x, components)

            DetectorFrequency = radIter.meshes["DetectorFrequency"]["omega"][0, :, 0]
            radiationSeries.flush()

            distributed_amplitudes = torch_empty((3, len(r_offset), len(DetectorFrequency)), dtype=torch_complex64) # shape: (components, GPUs, frequencies)

            for i_c, component in enumerate(["x", "y", "z"]):
                gpuBoxExtent = ps.particle_patches["extent"][component].load()
                gpuBoxOffset = ps.particle_patches["offset"][component].load()
                series.flush()

                Dist_Amplitude = radIter.meshes["Amplitude_distributed"][component].load_chunk() # shape: (GPUs, directions, frequencies)
                # MAGIC: only look along one direction
                DetectorDirection = radIter.meshes["DetectorDirection"][component][:, 0, 0] # shape: (x directions)
                radiationSeries.flush()

                r_offset[:, i_c] = gpuBoxOffset * iteration.get_attribute(cellExtensionNames[component])
                n_vec[:, i_c] = DetectorDirection

                # MAGIC: index direction = 0 to get ex vector = [1,0,0]
                i_direction = 0
                distributed_amplitudes[i_c] = torch_from_numpy(Dist_Amplitude[:, i_direction, :]) # shape of component i_c: (GPUs, frequencies)


            # time retardation correction
            # QUESTION: The `iteration.iteration_index`=int variable appears in here, not the actual time? (but r_offset is also in cells...)
            phase_offset = torch_from_numpy(np.exp(-1.j * DetectorFrequency[np.newaxis, np.newaxis, :] * (iteration.iteration_index + np.dot(r_offset, n_vec.T)[:, :, np.newaxis] / 1.0)))[:, i_direction, :]
            distributed_amplitudes = distributed_amplitudes/phase_offset

            # Transform to shape: (GPUs, components, frequencies)
            distributed_amplitudes = torch_transpose(distributed_amplitudes, 0, 1) # shape: (GPUs, components, frequencies)

            # MAGIC: Just look at y&z component
            r = distributed_amplitudes[:, 1:, :]

            # Compute the phase (angle) of the complex number
            phase = torch_angle(r)
            # Compute the absolute value of the complex number
            absolute = torch_abs(r)
            r = torch_cat((absolute, phase), dim=1).to(torch_float32)

            # Put particle and radiation data in shared buffer
            self.data.put([loaded_particles, r])

            print("Done loading iteration %i"%(iteration.time))
            stdout.flush()

            iteration.close() # It is currently not possible to reopen an iteration in openPMD
            radIter.close() # It is currently not possible to reopen an iteration in openPMD


        # signal that there are no further items
        print("Processed all iterations")
        stdout.flush()
        self.data.put(None)

        # close series
        series.close()
        radiationSeries.close()

