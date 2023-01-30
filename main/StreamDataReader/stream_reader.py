import json
import os
import sys

import openpmd_api as io


class StreamReader():
    
    def __init__(
            self,
            stream_config_json=os.path.join(
                os.path.dirname(__file__),
                'stream_config.json',
            ),
    ):
        self.__init_from_config_file(stream_config_json)
    
    def __init_from_config_file(self, stream_config_json):
        try:
            with open(stream_config_json) as stream_config:
                self._stream_cfg = json.load(stream_config)
                self._stream_path = self._stream_cfg.pop("stream_path")
            self._series = self._init_stream()
            self._series_iterator = self._init_series_iterator()
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
    def _init_stream(self):
        print("initialized stream")
        return io.Series(self._stream_path, io.Access_Type.read_only)
    
    def _init_series_iterator(self):
        print("initialized iterator")
        return iter(self._series.read_iterations())
    
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
                data = current_record[io.Mesh_Record_Component.SCALAR][0]
                shape = ()
            else:
                loadedChunks = []
                shapes = []
                for dim in current_record:
                    rc = current_record[dim]
                    loadedChunks.append(rc.load_chunk([0], rc.shape))
                    shapes.append(rc.shape)
                data = loadedChunks
                if isinstance(shapes[0], int):
                    shape = (len(shapes), shapes[0])
                else:
                    shapes[0].insert(0, len(shapes))
                    shape = tuple(shapes[0])
        else:
            print("Didn't find", record_key)
            data = None
            shape = None
        return data, shape

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
                    data_dict_node[key] = self._get_data_from_key(
                        current_record, key)
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
        iteration = self._get_iteration()
        if iteration is not None:
            return self._get_data(iteration)
        return None
