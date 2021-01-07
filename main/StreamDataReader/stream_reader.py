import openpmd_api as io
import numpy as np

class StreamReader():
    def __init__(self, stream_path, record_component, record_variable):
        self.stream_path = stream_path
        self.record_component = record_component
        self.record_variable = record_variable
        self.series = self._init_stream()
        self.series_iterator = self._init_series_iterator()
    
    def _init_stream(self):
        return io.Series(self.stream_path, io.Access_Type.read_only)
    
    def _init_series_iterator(self):
        return iter(self.series.read_iterations())
    
    def _get_iteration(self):
        try:
            iteration = next(self.series_iterator) 
            return iteration
        except StopIteration:
            return None
    
    def _get_data(self,current_iteration):
        if self.record_component is 'mesh':
            if self.record_variable in current_iteration.meshes:
                current_record = current_iteration.meshes[self.record_variable]
            else
                return None, None, None
        if self.record_component is 'particle':
                #todo:
                return None, None, None
        
        iteration_id = current_iteration.iteration_index
        dimensions = ["x", "y", "z"]
        loadedChunks = []
        shapes = []
        for dim in dimensions:
            rc = current_record[dim]
            loadedChunks.append(rc.load_chunk([0], rc.shape))
            shapes.append(rc.shape)
        current_iteration.close()
        
        return iteration_id, loadedChunks, shapes[0]
        
    def get_next_data(self):
        iteration = self._get_iteration()
        if iteration is not None:
            return self._get_data(iteration)
        return None, None, None