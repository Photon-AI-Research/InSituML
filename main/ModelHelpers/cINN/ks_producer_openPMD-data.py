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

from ks_helperfuncs import *

class Loader(Thread):

    def __init__(self, batchDataBuffer, hyperParameterDefaults, dataTransformationPolicy=None):
        """ Set parameters of the loader

        Arguments:
            batchDataBuffer (e.g. queue.Queue) : buffer to put the data into (where the consumer reads it)
            hyperParameterDefaults (dict) : Defines timesteps, paths to data, etc.
            dataTransformationPolicy (functor) : 
        """
        Thread.__init__(self)
        # instantiate all required parameters
        self.data = batchDataBuffer
        self.particlePathpattern = hyperParameterDefaults["pathpattern1"],
        self.radiationPathPattern = hyperParameterDefaults["pathpattern2"],
        self.t0 = hyperParameterDefaults["t0"],
        self.t1 = hyperParameterDefaults["t1"],
        self.timebatchSize = hyperParameterDefaults["timebatchsize"],
        self.timebatchSliceSize = hyperParameterDefaults["particlebatchsize"]
        self.numEpochs = hyperParameterDefaults["num_epochs"]
        self.transformPolicy = dataTransformationPolicy

        self.totalTimebatchNumber = int(self.t1-self.t0)/self.timebatchsize)

    def run(self):
        dataReads = int(0)
        i_tb = int(0)
        perm = torch_randperm(self.t1-self.t0)
        while dataReads < self.numEpochs*self.totalTimebatchNumber:
            i_tb = self.timebatchsize*timebatch
            bi = perm[i_tb:i_tb+self.timebatchSize]
            radiation = []
            particles = []
            for time in bi:
                index = time + self.t0
                
                p = np.load(self.particlePathPattern.format(index), allow_pickle = True)
                
                p = np.array([normalize_columns(element) for element in p], dtype=object)
                
                p = torch_from_numpy(np.array([random_sample(element, sample_size=10000) for element in p]), dtype = np.float32))

                if self.transformPolicy is not None:
                    p = self.transformPolicy(p)

                r = torch.from_numpy(np.load(self.radiationPathPattern.format(index)).astype(np.cfloat) )
                r = r[:, 1:, :]

                # Compute the phase (angle) of the complex number
                phase = torch.angle(r)
                # Compute the absolute value of the complex number
                absolute = torch.abs(r)
                r = torch_cat((absolute, phase), dim=1).to(torch_float32)

                particles.append(p)
                radiation.append(r)
            
            particles = torch_cat(particles)
            radiation = torch_cat(radiation)

            self.data.put(Timebatch(particles, radiation, self.timebatchSliceSize))

            dataReads += int(1)

            if i_tb%totalTimebatchNumber == 0:
                i_tb = 0
                perm = torch_randperm(self.t1-self.t0)

        # signal that there are no further items
        self.data.put(None)


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

