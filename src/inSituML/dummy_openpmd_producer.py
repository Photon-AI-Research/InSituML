from threading import Thread
import torch
from time import sleep

gpu_boxes = 2
number_of_particles = 100
ps_dims = 6

rad_dims = 512

class DummyOpenPMDProducer(Thread):
    def __init__(self, openPMDBuffer):
        Thread.__init__(self)        
        self.openPMDBuffer = openPMDBuffer
    
    def run(self):
        print('Producer: Running')

        # generate openpmd stuff.
        for i in range(10):
            # generate a value
            loaded_particles = torch.rand(gpu_boxes, ps_dims, number_of_particles)
            radiation = torch.rand(gpu_boxes, rad_dims)
            # block, to simulate effort
            sleep(2)
            # create a tuple
            item = [loaded_particles, radiation]
            # add to the queue
            self.openPMDBuffer.put(item)
            # report progress
            print(f'>producer added {i}')
        # signal that there are no further items
        self.openPMDBuffer.put(None)
        print('Producer: Done')
