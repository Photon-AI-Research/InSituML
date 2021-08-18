from threading import Thread
from StreamDataReader.stream_reader import StreamReader
import numpy as np
import time

class StreamBuffer(Thread):
    
    def __init__(self, buffer_size = 3, use_local_data = True):
        self._buffer_size = buffer_size
        self._buffer_data = None
        self.use_local_data = use_local_data
        if use_local_data:
            self.__c = 0
        else:
            self._stream = StreamReader()
        self.elp_time = []
        self.fill_buffer()
        
    def read_buffer(self):
        if self._buffer_data is not None:
            buffer_copy = self._buffer_data
            self._buffer_data = None
            return buffer_copy
        return None
    
    def fill_buffer(self):
        print("Filling buffer")
        start = time.time()
        self._buffer_data = None
        iteration_ids = []
        mesh_data = []
        for _ in range(self._buffer_size):
            if not self.use_local_data:
                data_dict = self._stream.get_next_data()
            else:
                data_dict = get_data_locally(self.__c)
            if data_dict is None:
                print("Finished iterations. Exiting Training and Validation")
                break
            if self.use_local_data:
                self.__c += 1
            iteration_ids.append(data_dict['iteration_id'])
            mesh_data.append(data_dict['meshes']['E'])
            shape = data_dict['meshes']['E_shape'] 
        self.elp_time.append(time.time() - start)
        self._buffer_data = (iteration_ids, mesh_data, shape)
    
def get_data_locally(iteration_id):
    try:
        data = np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/main/Data/data_{}.npy".format(iteration_id * 100))
        data_dict = {'iteration_id':iteration_id,
                    'meshes':{
                        'E':data,
                        'E_shape':(3,128,1280,128)
                    }
                    }
    except:
        data_dict = None
    return data_dict