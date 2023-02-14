import json
import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import openpmd_api as io


class StreamData:
    __slots__ = ['data', 'np_shape', 'pic_shape']

    def __init__(
            self,
            data: Union[np.ndarray, List[np.ndarray]],
            np_shape: Tuple,
            pic_shape: Optional[int],
    ):
        self.data = data
        self.np_shape = np_shape
        self.pic_shape = pic_shape

    def __len__(self):
        return len(self.__slots__)

    def __getitem__(self, key):
        return getattr(self, self.__slots__[key])

    def __iter__(self):
        return map(lambda key: getattr(self, key), self.__slots__)


class StreamReader():
    
    def __init__(
            self,
            stream_path,
            stream_config_json=os.path.join(
                os.path.dirname(__file__),
                'stream_config.json',
            ),
    ):
        self._stream_path = stream_path
        self.__init_from_config_file(stream_config_json)
        self._last_data = None

    def __init_from_config_file(self, stream_config_json):
        with open(stream_config_json) as stream_config:
            self._stream_cfg = json.load(stream_config)
        self._series = self._init_stream()
        self._series_iterator = self._init_series_iterator()

    def _init_stream(self):
        print("initialized stream")
        return io.Series(self._stream_path, io.Access_Type.read_only)
    
    def _init_series_iterator(self):
        print("initialized iterator")
        return iter(self._series.read_iterations())

    def __iter__(self):
        while True:
            self._last_data = self.get_next_data()
            if self._last_data is None:
                break
            yield self._last_data
            self._last_data = None

    def _get_iteration(self):
        try:
            print("Getting NEXT")
            iteration = None
            if self._series_iterator is not None:
                iteration = next(self._series_iterator) 
            return iteration
        except StopIteration:
            return None

    def _get_data_from_key(self, record, record_key):
        if record_key in record:
            current_record = record[record_key]
            if len(current_record) == 1:
                data = current_record[io.Mesh_Record_Component.SCALAR]
                if 'shape' in data.attributes:
                    pic_shape = data.get_attribute('shape')
                else:
                    pic_shape = None
                data = data[0]
                np_shape = ()
            else:
                loadedChunks = []
                np_shapes = []
                for dim in current_record:
                    rc = current_record[dim]
                    loadedChunks.append(rc.load_chunk([0], rc.shape))
                    np_shapes.append(rc.shape)

                # PIConGPU shape stays the same over all dimensions.
                if np_shapes and 'shape' in rc.attributes:
                    pic_shape = rc.get_attribute('shape')
                else:
                    pic_shape = None

                data = loadedChunks
                if isinstance(np_shapes[0], int):
                    np_shape = (len(np_shapes), np_shapes[0])
                else:
                    np_shapes[0].insert(0, len(np_shapes))
                    np_shape = tuple(np_shapes[0])
        else:
            print("Didn't find", record_key)
            return None
        return StreamData(data, np_shape, pic_shape)

    def _get_data(self,current_iteration):
        data_dict = dict(iteration_index=current_iteration.iteration_index)

        # Each element in here contains a 3-tuple with the contents:
        # - `self._stream_cfg` node under which to look for keys.
        # - `data_dict` node under which to store results for each key.
        # - The associated record that is supposed to contain the keys.
        remaining_nodes = [(self._stream_cfg, data_dict, current_iteration)]
        while remaining_nodes:
            (
                stream_cfg_node,
                data_dict_node,
                current_record,
            ) = remaining_nodes.pop()

            if isinstance(stream_cfg_node, list):
                # Process all leaf keys.
                for key in stream_cfg_node:
                    data = self._get_data_from_key(current_record, key)
                    if data is not None:
                        data_dict_node[key] = data
            else:
                # Add more remaining nodes, descend tree.
                for (key, stream_cfg_child) in stream_cfg_node.items():
                    child_found = False
                    # Access outer-most values by attribute, after that
                    # only use `getitem`.
                    if current_record is current_iteration:
                        if hasattr(current_record, key):
                            child_record = getattr(current_record, key)
                            child_found = True
                    elif key in current_record:
                            child_record = current_record[key]
                            child_found = True

                    if child_found:
                        data_dict_child = {}
                        data_dict_node[key] = data_dict_child
                        remaining_nodes.append((
                            stream_cfg_child,
                            data_dict_child,
                            child_record,
                        ))
                    else:
                        print('Did not find', key)

        current_iteration.close()
        
        return data_dict
        
    def get_next_data(self):
        # If we stopped a previous iteration, first yield the value that
        # had already been fetched. I.e. the last value from the
        # iterator that has been fetched, but not consumed.
        if self._last_data is not None:
            data = self._last_data
            self._last_data = None
            return data

        iteration = self._get_iteration()
        if iteration is not None:
            return self._get_data(iteration)
        return None

    def close(self):
        del self._series

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass
        super().__del__()
