"""
Loader for PIConGPU openPMD particle and radiation data to be used for insitu machine learning model training.
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

class Loader(Thread):
    """ Thread providing PIConGPU particle and radiation data for the ML model training.

    PIConGPU data is provided by a dataReadPolicy on a per-timestep basis as an ndarray
    of shape (numLocalSubvolumes, phaseSpaceComponents, numParticlesPerSubvolume) (TODO!).
    phaseSpaceComponents are: (pos_x, pos_y, pos_z, mom_x, mom_y, mom_z, force_x, force_y, force_z)
    This data can be reshaped according to the needs of the training/model by a dataTransformationPolicy.
    Loaded and reshaped data is put in a buffer which is shared with the training thread.

    All of this is orchestrated in the run() method.
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
        self.particlePathpattern = "/gpfs/alpine2/csc380/proj-shared/ksteinig/2024-02_KHI-for-ML_reduced/001/simOutput/openPMD/simData_%T.bp"
#        self.radiationPathPattern = hyperParameterDefaults["pathpattern2"],
        self.radiationPathPattern = "/gpfs/alpine2/csc380/proj-shared/ksteinig/2024-02_KHI-for-ML_reduced/001/simOutput/radiationOpenPMD/e_radAmplitudes_%T_0_0_0.h5"
        self.t0 = hyperParameterDefaults["t0"]
        self.t1 = hyperParameterDefaults["t1"] # endpoint=false, t1 is not loaded from disk
        self.timebatchSize = hyperParameterDefaults["timebatchsize"]
        self.timebatchSliceSize = hyperParameterDefaults["particlebatchsize"]
        self.numEpochs = hyperParameterDefaults["num_epochs"]
        self.transformPolicy = None #dataTransformationPolicy

        self.totalTimebatchNumber = int((self.t1-self.t0)/self.timebatchSize)
        self.particlePerGPU = np.int64(10000) # MAGIC: Number of randomly choosen particles per GPU
        
        self.rng = np.random.default_rng()

    def run(self):
        """Function being executed when thread is started."""
        from sys import stdout
        # Open openPMD particle and radiation series
        series = opmd.Series(self.particlePathpattern, opmd.Access.read_only)
        radiationSeries = opmd.Series(self.radiationPathPattern, opmd.Access.read_only)

        # start reading data
        i_epoch = int(0)
        i_tb = int(0)
        perm = self.rng.permutation(self.t1-self.t0)
        while i_epoch < self.numEpochs:
            """Iterate over all timebatches in all epochs."""
            print("Start epoch ", i_epoch)
            ###############################################
            # Fill timebatch with particle and radiation data
            bi = perm[i_tb:i_tb+self.timebatchSize]
            radiation = []
            particles = []
            for step in (bi+self.t0):
                """iterate over all timesteps belonging to a timebatch"""
                iteration = series.iterations[step]

                ## obtain particle distribution over GPUs ##
                #
                ps = iteration.particles["e_all"] #particle species

                numParticles = ps.particle_patches["numParticles"][opmd.Mesh_Record_Component.SCALAR].load()
                numParticlesOffsets = ps.particle_patches["numParticlesOffset"][opmd.Mesh_Record_Component.SCALAR].load()
                series.flush()

                totalParticles = np.sum(numParticles) # numParticlesOffsets[-1] + numParticles[-1]?

                # prepare selection of particle indices that will be used for training from the whole data set
                # In case numParticlesPerGpu < particlePerGpuForTraining, rng.choice() will throw a ValueError and stop.
                # In streaming setups, we should catch this error and have a mitigation strategy in order to be able to continue.
                randomParticlesPerGPU = np.array([ self.rng.choice(np.int64(numParticles[i]-1), self.particlePerGPU, replace=False) for i in np.arange(len(numParticles))])

                local_region = {"offset": [numParticlesOffsets[0]], "extent": [totalParticles]}

                numParticlesOffsets = np.concatenate((numParticlesOffsets, [-1]), dtype=np.int64) # append to use array for indexing

                # prepare torch tensor to hold particle data in shape (phaseSpaceComponents, GPUs, particlesPerGPU)
                loaded_particles = torch_empty((9, len(numParticles), self.particlePerGPU), dtype=torch_float32)
                for i_c, component in enumerate(["x", "y", "z"]):
                    """Read particle data component-wise to reduce host memory usage.
                       And immediately reshape by subdividing in particles per GPU.
                       Also, reduce to requested number particles per GPU.
                    """
                    position = (ps["position"][component].load_chunk(local_region["offset"], local_region["extent"])
                        + ps["positionOffset"][component].load_chunk(local_region["offset"], local_region["extent"])
                    )
                    pos_reduced = np.array([
                        position[numParticlesOffsets[i]:numParticlesOffsets[i+1]][randomParticlesPerGPU[i]]
                        for i in np.arange(len(numParticles))
                    ])

                    ## Is it required to normalize positions and other phase space componentes?

                    #del position #to save memory?
                    # If memory is of concern, I could also first reduce position and delete it,
                    # then load, reduce, and delete positionOffset,
                    # and finally add the two reduced arrays.

                    momentum = ps["momentum"][component].load_chunk(local_region["offset"], local_region["extent"])
                    mom_reduced = np.array([
                        momentum[numParticlesOffsets[i]:numParticlesOffsets[i+1]][randomParticlesPerGPU[i]]
                        for i in np.arange(len(numParticles))
                    ])

                    momentumPrev1 = ps["momentumPrev1"][component].load_chunk(local_region["offset"], local_region["extent"])
                    momPrev1_reduced = np.array([
                        momentumPrev1[numParticlesOffsets[i]:numParticlesOffsets[i+1]][randomParticlesPerGPU[i]]
                        for i in np.arange(len(numParticles))
                    ])

                    loaded_particles[0+i_c] = torch_from_numpy(pos_reduced) # absolute position in cells
                    loaded_particles[3+i_c] = torch_from_numpy(mom_reduced) # momentum in gamma*beta
                    loaded_particles[6+i_c] = torch_from_numpy(mom_reduced - momPrev1_reduced) # force

#                iteration.close() # It is currently not possible to reopen an iteration in openPMD
                
                if self.transformPolicy is not None:
                    loaded_particles = self.transformPolicy(loaded_particles)
                else:
                    """transform data to shape (GPUs, phaseSpaceComponents, particlesPerGPU)"""
                    loaded_particles = torch_transpose(loaded_particles, 0, 1)

                particles.append(loaded_particles)


                ## obtain radiation data per GPU ##
                #
                radIter = radiationSeries.iterations[step]

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

#                radIter.close() # It is currently not possible to reopen an iteration in openPMD

                # time retardation correction
                # QUESTION: The `step`=int variable appears in here, not the actual time? (but r_offset is also in cells...)
                phase_offset = torch_from_numpy(np.exp(-1.j * DetectorFrequency[np.newaxis, np.newaxis, :] * (step + np.dot(r_offset, n_vec.T)[:, :, np.newaxis] / 1.0)))[:, i_direction, :]
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

                radiation.append(r)

            
            particles = torch_cat(particles)
            radiation = torch_cat(radiation)

            self.data.put(self.Timebatch(particles, radiation, self.timebatchSliceSize, self.rng))

            print("Finish timebatch. Timestep =", i_tb)
            stdout.flush()

            i_tb += self.timebatchSize

            if i_tb%self.totalTimebatchNumber == 0:
                """All timesteps have been read once within this epoch. Epoch finished."""
                print("Finished epoch", i_epoch)
                stdout.flush()
                i_epoch += int(1)
                i_tb = int(0)
                perm = self.rng.permutation(len(series.iterations))

        # signal that there are no further items
        print("Finish iterating all epochs")
        stdout.flush()
        self.data.put(None)

        # close series
        series.close()
        radiationSeries.close()


    class Timebatch:
        def __init__(self, particles, radiation, batchsize, rng=np.random.default_rng()):
            self.batchsize = batchsize
            self.particles = particles
            self.radiation = radiation

            self.rng = rng
            self.perm = self.rng.permutation(self.radiation.shape[0])

        def __len__(self):
            return self.radiation.shape[0] // self.batchsize

        def __getitem__(self, batch):
            i = self.batchsize*batch
            bi = self.perm[i:i+self.batchsize]

            return self.particles[bi], self.radiation[bi]

