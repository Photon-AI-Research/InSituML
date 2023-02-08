import os
from threading import Thread
import time

import numpy as np

from StreamDataReader.stream_reader import StreamReader


class StreamBuffer(Thread):
    
    def __init__(
            self,
            stream_config_json=os.path.join(
                os.path.dirname(__file__),
                'stream_config.json',
            ),
            buffer_size=3,
            use_local_data=False,
    ):

        """
            Initializing Stream Buffer.

            Parameters
            ----------
            buffer_size : int
                The amount of data iterations the buffer holds before reading the stream again
            use_local_data : bool
                To access data, if data is stored locally

        """

        self._buffer_size = buffer_size
        self._buffer_data = None
        self.use_local_data = use_local_data
        if use_local_data:
            self.__c = 0
        else:
            self._stream = StreamReader(stream_config_json)
        self.elp_time = []
        self.fill_buffer()
        
    def read_buffer(self):
        """
            Reads the buffer.

            Returns
            -------
            Tuple
               iteration_id, buffer_data, shape - (if applicable)

        """
        if self._buffer_data is not None:
            buffer_copy = self._buffer_data
            self._buffer_data = None
            return buffer_copy
        return None
    
    def fill_buffer(self):

        """
            Fills the buffer.

            Needs to be externally called when buffer is empty

        """

        print("Filling buffer")
        start = time.time()
        self._buffer_data = None
        data_dicts = []
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
            data_dicts.append(data_dict)
        self.elp_time.append(time.time() - start)
        self._buffer_data = data_dicts


def get_data_locally(iteration_id):
    try:
        data = np.load("/home/h5/vama551b/home/streamed-ml/StreamedML/Data/data_{}.npy".format(iteration_id * 100))
        data_dict = {
            'iteration_index': iteration_id,
            'meshes': {
                'E': (data, (3, 128, 1280, 128)),
            },
        }
    except:
        data_dict = None
    return data_dict
