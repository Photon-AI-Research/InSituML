"""
Loader for PIConGPU openPMD particle and radiation data to be used for insitu machine learning model training.
The data is put in a buffer provided during construction of the producer class.
The buffer is expected to be fillable by a `put()` method.
Furthermore, policies are taken which transform particle or radiation data from the their standard layouts to the requested layout.
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
from torch import stack as torch_stack

# need to import mpi4py, otherwise there is an undefined symbol required by adios when importing openpmd_api (on hemera)
from mpi4py import MPI
import openpmd_api as opmd

from ks_helperfuncs import *

class RandomLoader(Thread):
    """ Thread providing PIConGPU particle and radiation data for the ML model training.

    PIConGPU data is loaded from openPMD files on a per-timestep basis in Timebatches.
    The standard format of loaded particle data is a torch.Tensor of shape (numLocalSubvolumes, phaseSpaceComponents, numParticlesPerSubvolume).
    phaseSpaceComponents are: (pos_x, pos_y, pos_z, mom_x, mom_y, mom_z, force_x, force_y, force_z)
    This data can be reshaped according to the needs of the training/model by a dataTransformationPolicy.
    Loaded and reshaped data is put in a buffer which is shared with the training thread.

    All of this is orchestrated in the run() method.
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
        self.particlePathpattern = hyperParameterDefaults["pathpattern1"]
#        self.particlePathpattern = "/gpfs/alpine2/csc380/proj-shared/ksteinig/2024-02_KHI-for-ML_reduced/001/simOutput/openPMD/simData_%T.bp"
        self.radiationPathPattern = hyperParameterDefaults["pathpattern2"]
#        self.radiationPathPattern = "/gpfs/alpine2/csc380/proj-shared/ksteinig/2024-02_KHI-for-ML_reduced/001/simOutput/radiationOpenPMD/e_radAmplitudes_%T_0_0_0.h5"
        self.t0 = hyperParameterDefaults["t0"]
        self.t1 = hyperParameterDefaults["t1"] # endpoint=false, t1 is not loaded from disk
        self.numEpochs = hyperParameterDefaults["num_epochs"]
        self.particleTransformPolicy = particleDataTransformationPolicy
        self.radiationTransformPolicy = radiationDataTransformationPolicy
        self.particlePerGPU = hyperParameterDefaults['number_particles_per_gpu']
        self.amplitude_direction = hyperParameterDefaults['amplitude_direction']
        self.normalization = hyperParameterDefaults["normalization"]
        
        self.rng = np.random.default_rng()

        self.reqPhaseSpaceVars = hyperParameterDefaults["phase_space_variables"]
        ## check input validity
        allowedVars = ["position", "momentum", "force"]
        variablesAllowed = True
        for var in self.reqPhaseSpaceVars:
            if var not in allowedVars:
                variablesAllowed = False
        if "force" in self.reqPhaseSpaceVars and "momentum" not in self.reqPhaseSpaceVars:
            variablesAllowed = False
            assert variablesAllowed, "Phase space variable 'force' can only be used in combination with 'momentum'"

        assert variablesAllowed, f"Requested phase space variables are not in allowed range {allowedVars}"

        self.verbose = hyperParameterDefaults['verbose'] if 'verbose' in hyperParameterDefaults else False

    def run(self):
        """Function being executed when thread is started."""
        from sys import stdout
        # Open openPMD particle and radiation series
        series = opmd.Series(self.particlePathpattern, opmd.Access.read_only)

        radiationSeries = opmd.Series(self.radiationPathPattern, opmd.Access.read_only)

        # start reading data
        i_epoch = int(0)
        while i_epoch < np.ceil(self.numEpochs):
            """Iterate over all timebatches in all epochs."""
            perm = self.rng.permutation(self.t1-self.t0)
            if self.numEpochs - i_epoch < 1.:
                perm = perm[:int(len(perm) * (self.numEpochs - i_epoch))]
            print("Start epoch ", i_epoch)
            ###############################################
            # Fill timebatch with particle and radiation data
            for step in (perm+self.t0):
                """iterate over all timesteps"""
                iteration = series.iterations[step]

                if self.verbose:
                    print("loading iteration ", step, iteration)

                ## obtain particle distribution over GPUs ##
                #
                ps = iteration.particles["e_all"] #particle species

                numParticles = ps.particle_patches["numParticles"][opmd.Mesh_Record_Component.SCALAR].load()
                numParticlesOffsets = ps.particle_patches["numParticlesOffset"][opmd.Mesh_Record_Component.SCALAR].load()
                series.flush()

                totalParticles = np.sum(numParticles)

                # prepare selection of particle indices that will be used for training from the whole data set
                # In case numParticlesPerGpu < particlePerGpuForTraining, rng.choice() will throw a ValueError and stop.
                # In streaming setups, we should catch this error and have a mitigation strategy in order to be able to continue.
                randomParticlesPerGPU = np.array([ self.rng.choice(numParticles[i], self.particlePerGPU, replace=False) for i in np.arange(len(numParticles))])
                randomParticles = np.array([ randomParticlesPerGPU[i] + numParticlesOffsets[i] for i in np.arange(len(numParticles))])

                local_region = {"offset": [numParticlesOffsets[0]], "extent": [totalParticles]}

                # prepare torch tensor to hold particle data in shape (phaseSpaceComponents, GPUs, particlesPerGPU)
                loaded_particles = torch_empty((len(self.reqPhaseSpaceVars)*3, len(numParticles), self.particlePerGPU), dtype=torch_float32)
                for i_c, component in enumerate(["x", "y", "z"]):
                    """Read particle data component-wise to reduce host memory usage.
                       And immediately reshape by subdividing in particles per GPU.
                       Also, reduce to requested number particles per GPU.
                    """
                    writing_index = 0
                    
                    gpuBoxExtent = ps.particle_patches["extent"][component].load()
                    gpuBoxOffset = ps.particle_patches["offset"][component].load()
                    series.flush()

                    if "position" in self.reqPhaseSpaceVars:
                        pos = ps["position"][component].load_chunk(local_region["offset"], local_region["extent"])
                        posOffset = ps["positionOffset"][component].load_chunk(local_region["offset"], local_region["extent"])
                        series.flush()
                        position = pos + posOffset
                        del pos
                        del posOffset
                        loaded_particles[writing_index+i_c] = torch_stack([
                            torch_from_numpy(position[r]) for r in randomParticles
                        ])
                        del position

                        ## Normalize Positions
                        ## TODO: The local box min and max values used for normalization,
                        ## need to be stored somewhere, in order to be able to be able to
                        ## denormalize during inference if position is used in training.
                        for particleBoxIndex in np.arange(len(numParticles)):
                            posMin = gpuBoxOffset[particleBoxIndex]
                            posMax = posMin + gpuBoxExtent[particleBoxIndex]
                            loaded_particles[writing_index+i_c, particleBoxIndex] = (loaded_particles[writing_index+i_c, particleBoxIndex] - posMin) / (posMax - posMin)
                        writing_index +=3

                    if "momentum" in self.reqPhaseSpaceVars or "force" in self.reqPhaseSpaceVars:
                        momentum = ps["momentum"][component].load_chunk(local_region["offset"], local_region["extent"])
                        series.flush()
                        loaded_particles[writing_index+i_c] = torch_stack([
                            torch_from_numpy(momentum[r]) for r in randomParticles
                        ])
                        writing_index +=3
                        del momentum
                        
                    if "force" in self.reqPhaseSpaceVars:
                        momentumPrev1 = ps["momentumPrev1"][component].load_chunk(local_region["offset"], local_region["extent"])
                        series.flush()
                        momPrev1_reduced = torch_stack([
                            torch_from_numpy(momentumPrev1[r]) for r in randomParticles
                        ])
                        loaded_particles[writing_index+i_c] = loaded_particles[writing_index-3+i_c] - momPrev1_reduced # force = momentum - momentumPrev1
                        del momPrev1_reduced


                    writing_index = 0
                    if "position" in self.reqPhaseSpaceVars:
                        writing_index +=3
                    if "momentum" in self.reqPhaseSpaceVars:
                        for particleBoxIndex in np.arange(len(numParticles)):
                            loaded_particles[writing_index+i_c, particleBoxIndex] = \
                                (loaded_particles[writing_index+i_c, particleBoxIndex] - self.normalization["momentum_mean"]) \
                                / self.normalization["momentum_std"]

                        writing_index +=3
                    if "force" in self.reqPhaseSpaceVars:
                        ## Normalize force
                        for particleBoxIndex in np.arange(len(numParticles)):
                            loaded_particles[writing_index+i_c, particleBoxIndex] = \
                                (loaded_particles[writing_index+i_c, particleBoxIndex] - self.normalization["force_mean"]) \
                                / self.normalization["force_std"]

#                iteration.close() # It is currently not possible to reopen an iteration in openPMD
                
                if self.particleTransformPolicy is not None:
                    loaded_particles = self.particleTransformPolicy(loaded_particles)


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
                    Dist_Amplitude = radIter.meshes["Amplitude_distributed"][component].load_chunk() # shape: (GPUs, directions, frequencies)
                    # MAGIC: only look along one direction
                    DetectorDirection = radIter.meshes["DetectorDirection"][component][:, 0, 0] # shape: (x directions)
                    radiationSeries.flush()

                    r_offset[:, i_c] = gpuBoxOffset * iteration.get_attribute(cellExtensionNames[component])
                    n_vec[:, i_c] = DetectorDirection

                    # Return radiation amplitude only along a specific direction
                    distributed_amplitudes[i_c] = torch_from_numpy(Dist_Amplitude[:, self.amplitude_direction, :]) # shape of component i_c: (GPUs, frequencies)

#                radIter.close() # It is currently not possible to reopen an iteration in openPMD

                # time retardation correction
                phase_offset = torch_from_numpy(np.exp(-1.j * DetectorFrequency[np.newaxis, np.newaxis, :] * (step - np.dot(r_offset, n_vec.T)[:, :, np.newaxis] / 1.0)))[:, self.amplitude_direction, :]
                distributed_amplitudes = distributed_amplitudes*phase_offset

                # Transform to shape: (GPUs, components, frequencies)
                distributed_amplitudes = torch_transpose(distributed_amplitudes, 0, 1) # shape: (GPUs, components, frequencies)

                if self.radiationTransformPolicy is not None:
                    distributed_amplitudes = self.radiationTransformPolicy(distributed_amplitudes)
                

                self.data.put([loaded_particles, distributed_amplitudes])

            """All timesteps have been read once within this epoch. Epoch finished."""
            print("Finished epoch", i_epoch)
            stdout.flush()
            i_epoch += int(1)
            perm = self.rng.permutation(len(series.iterations))

        # signal that there are no further items
        print("Finish iterating all epochs")
        stdout.flush()
        self.data.put(None)

        # close series
        series.close()
        radiationSeries.close()

