"""
Loader for PIConGPU openPMD particle data to be used for machine learning model training.
The data is put in a buffer provided during construction of the producer class.
The buffer is expected to be fillable by a `put()` method.
Furthermore, a policy is taken which describes the data structure that needs to be put in the buffer. This policy actually performs the required transformation on the data, if required.
"""

from threading import Threading

import numpy as np
from torch import randperm as torch_randperm
from torch import from_numpy as torch_from_numpy
from torch import cat as torch_cat
from torch import float32 as torch_float32

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
        #self.particlePathpattern = hyperParameterDefaults["pathpattern1"],
        self.particlePathpattern = "/ccs/home/ksteinig/ALPINE2/2024-02_KHI-for-ML_reduced/001/simOutput/openPMD/simData_%T.bp"
        #self.radiationPathPattern = hyperParameterDefaults["pathpattern2"],
        self.radiationPathPattern = "/ccs/home/ksteinig/ALPINE2/2024-02_KHI-for-ML_reduced/001/simOutput/radiationOpenPMD/e_radAmplitudes_%T.h5"
        self.t0 = hyperParameterDefaults["t0"]
        self.t1 = hyperParameterDefaults["t1"] # endpoint=false, t1 is not loaded from disk
        self.timebatchSize = hyperParameterDefaults["timebatchsize"]
        self.timebatchSliceSize = hyperParameterDefaults["particlebatchsize"]
        self.numEpochs = hyperParameterDefaults["num_epochs"]
        self.transformPolicy = None #dataTransformationPolicy

        self.totalTimebatchNumber = int((self.t1-self.t0)/self.timebatchsize)
        self.particlePerGPU = int(10000)

    def run(self):
        """Function being executed when thread is started."""
        # Open openPMD particle and radiation series
        series = opmd.Series(self.particlePathpattern, opmd.Access.read_only)
        radiationSeries = opmd.Series(self.radiationPathpattern, opmd.Access.read_only)

        # start reading data
        i_epoch = int(0)
        i_tb = int(0)
        perm = torch_randperm(self.t1-self.t0)
        while i_epoch < self.numEpochs:
            """Iterate over all timebatches in all epochs."""
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
                rng = np.random.default_rng()
                randomParticlesPerGPU = np.array([ rng.choice(numParticles[i], self.particlePerGPU, replace=False) for i in np.arange(len(numParticles))])

                local_region = {"offset": numParticlesOffsets[0], "extent": totalParticles}

                numParticlesOffsets.append(int(-1)) # append to use array for indexing

#                for i_gpu in arange(len(numParticles)):
#                    """Chunk particle load in gpu sizes to reduce host memory usage."""
                # prepare torch tensor to hold particle data in shape (phaseSpaceComponents, GPUs, particlesPerGPU)
                loaded_particles = torch.empty((9, len(numParticles), self.particlePerGPU))
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

                    loaded_particles[0+i_c] = pos_reduced # absolute position in cells
                    loaded_particles[3+i_c] = mom_reduced # momentum in gamma*beta
                    loaded_particles[6+i_c] = mom_reduced - momPrev1_reduced # force

                iteration.close()
                
                if self.transformPolicy is not None:
                    loaded_particles = self.transformPolicy(loaded_particles)
                else:
                    """transform data to shape (GPUs, phaseSpaceComponents, particlesPerGPU)"""
                    loaded_particles = torch.transpose(loaded_particles, 0, 1)

                particles.append(loaded_particles)


                ## obtain radiation data per GPU ##
                #
                radIter = radiationSeries.iterations[step]

                # ... do things here according to transform script ...

                radIter.close()

#                r = torch.from_numpy(np.load(self.radiationPathPattern.format(index)).astype(np.cfloat) )
#                r = r[:, 1:, :]

#                # Compute the phase (angle) of the complex number
#                phase = torch.angle(r)
#                # Compute the absolute value of the complex number
#                absolute = torch.abs(r)
#                r = torch_cat((absolute, phase), dim=1).to(torch_float32)
#
#                radiation.append(r)

            
            particles = torch_cat(particles)
#            radiation = torch_cat(radiation)

#            self.data.put(Timebatch(particles, radiation, self.timebatchSliceSize))

            i_tb += self.timebatchsize

            if i_tb%totalTimebatchNumber == 0:
                """All timesteps have been read once within this epoch. Epoch finished."""
                i_epoch += int(1)
                i_tb = int(0)
                perm = torch_randperm(len(series.iterations))

        # signal that there are no further items
        self.data.put(None)

        # close series
        del series, del radiationSeries


    class Timebatch:
        def __init__(self, particles, radiation, batchsize):
            self.batchsize = batchsize
            self.particles = particles
            self.radiation = radiation

            self.perm = torch_randperm(self.radiation.shape[0])

        def __len__(self):
            return self.radiation.shape[0] // self.batchsize

        def __getitem__(self, batch):
            i = self.batchsize*batch
            bi = self.perm[i:i+self.batchsize]

            return self.particles[bi], self.radiation[bi]

