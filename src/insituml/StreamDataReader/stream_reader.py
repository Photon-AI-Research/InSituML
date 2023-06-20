import json
import os
import time
from typing import Optional, Tuple

import numpy as np
import openpmd_api as io


class StreamData:
    __slots__ = ['data', 'np_shape', 'pic_shape', 'unit_si']

    def __init__(
            self,
            data: np.ndarray,
            np_shape: Tuple,
            pic_shape: Optional[int],
            unit_si: Optional[float],
    ):
        self.data = data
        self.np_shape = np_shape
        self.pic_shape = pic_shape
        self.unit_si = unit_si

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

    def __chunk_distribution_strategy(strategy_identifier=None):
        if strategy_identifier is None or not strategy_identifier:
            # if 'OPENPMD_CHUNK_DISTRIBUTION' in os.environ:
            #     strategy_identifier = os.environ[
            #         'OPENPMD_CHUNK_DISTRIBUTION'].lower()
            # else:
            #     strategy_identifier = 'hostname_roundrobin_discard'  # default

            # hardcoding this strategy for now
            strategy_identifier = 'hostname_roundrobin_discard'  # default
        import re
        match = re.search('hostname_(.*)_(.*)', strategy_identifier)
        if match is not None:
            inside_node = StreamReader.__chunk_distribution_strategy(
                strategy_identifier=match.group(1))
            second_phase = StreamReader.__chunk_distribution_strategy(
                strategy_identifier=match.group(2))
            return io.FromPartialStrategy(io.ByHostname(inside_node),
                                          second_phase)
        elif strategy_identifier == 'roundrobin':
            return io.RoundRobin()
        elif strategy_identifier == 'binpacking':
            return io.BinPacking()
        elif strategy_identifier == 'discard':
            return io.DiscardingStrategy()
        # elif strategy_identifier == 'slicedataset':
        #     return io.ByCuboidSlice(io.OneDimensionalBlockSlicer(),
        #                             dataset_extent, mpi_rank, mpi_size)
        elif strategy_identifier == 'fail':
            return io.FailingStrategy()
        else:
            raise RuntimeError("Unknown distribution strategy: " +
                               strategy_identifier)


    def __init_from_config_file(self, stream_config_json):
        with open(stream_config_json) as stream_config:
            self._stream_cfg = json.load(stream_config)
        self._series = self._init_stream()
        self._series_iterator = self._init_series_iterator()

    def _init_stream(self):
        print("initializing stream")
        sleep_duration = 2
        sleep_limit = 60

        sleeped_secs = 0

        directory = os.path.dirname(self._stream_path)
        print("Waiting for dir", directory)
        while not os.path.isdir(directory):
            time.sleep(sleep_duration)
            sleeped_secs += sleep_duration
            if sleeped_secs >= sleep_limit:
                raise OSError(
                    f'stream file did not appear after {sleep_limit} seconds')

        max_tries = 10
        sleep_duration = 10
        for i in range(max_tries):
            try:
                return io.Series(self._stream_path, io.Access_Type.read_only)
            except RuntimeError as ex:
                prev_msg = str(ex)
                print('opening', self._stream_path, 'failed...')
                time.sleep(10)
        raise RuntimeError(prev_msg)

    def _init_series_iterator(self):
        print("initializing iterator")
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

    def _get_data_from_key(self, record, record_key, chunk_assignment):

        def get_chunk_assignment(rc, chunk_assignment):
            if chunk_assignment:
                return chunk_assignment
            available_chunks = rc.available_chunks()
            # @todo: for other chunk distribution strategies, we would need to
            # somehow emulate the other hostnames here too
            # This is a dictionary "MPI Rank" -> "hostname that it runs on"
            # The distribution strategy that is hardcoded right now
            # (see __chunk_distribution_strategy) works with
            # only this information specified below, for other distributions
            # we would need the other hostnames here, too
            # (It does not matter if this is not really an MPI application,
            # the ranks just need to be consistently emulated for the
            # distribution algorithms to work)
            target_hostnames = {0: io.HostInfo.HOSTNAME.get()}
            source_hostnames = self._series.mpi_ranks_meta_info
            if not source_hostnames:
                print("[Warning] Data source contains no host_table."
                    " Chunk distribution will not work as expected.")
                offset = [0 for _ in rc.shape]
                return io.ChunkInfo(offset, rc.shape)
            distribution_algorithm =\
                StreamReader.__chunk_distribution_strategy()
            chunk_assignment = distribution_algorithm.assign(available_chunks,
                                                    source_hostnames,
                                                    target_hostnames)
            # Get chunks that were assigned to our rank
            chunk_assignment = chunk_assignment[0]
            chunk_assignment.merge_chunks()
            if len(chunk_assignment) > 1:
                raise RuntimeError("Need contiguous chunks!")
            return chunk_assignment[0]

        def constant_or_array(rc, chunk_assignment):
            if rc.constant:
                data = np.zeros(rc.shape) + rc.get_attribute('value')
                np_shape = ()

                pic_shape = rc.shape
                unit_si = rc.unit_SI

                return StreamData(data, np_shape, pic_shape, unit_si), chunk_assignment
            else:
                chunk_assignment = \
                    get_chunk_assignment(rc, chunk_assignment) \
                    if chunk_assignment is None else chunk_assignment
                data = rc.load_chunk(
                    chunk_assignment.offset, chunk_assignment.extent)
                np_shape = rc.shape

                if isinstance(np_shape, int):
                    np_shape = (1, np_shape)
                else:
                    np_shape = (1,) + tuple(np_shape)

                # PIConGPU shape stays the same over all dimensions,
                # but we duplicate it anyway.
                if 'shape' in rc.attributes:
                    pic_shape = rc.get_attribute('shape')
                else:
                    pic_shape = None
                unit_si = rc.unit_SI

                return StreamData(data, np_shape, pic_shape, unit_si), chunk_assignment

        if record_key in record:
            current_record = record[record_key]
            if current_record.scalar:
                rc = current_record[io.Mesh_Record_Component.SCALAR]
                result, chunk_assignment =\
                    constant_or_array(rc, chunk_assignment)
            else:
                result = {}
                for dim in current_record:
                    rc = current_record[dim]
                    dim_result, chunk_assignment =\
                        constant_or_array(rc, chunk_assignment)
                    result[dim] = dim_result
                if not result:
                    print('Got no per-entry data for', record_key)
                    return None, chunk_assignment
        else:
            print("Didn't find", record_key)
            return None, chunk_assignment
        return result, chunk_assignment

    def _get_data(self,current_iteration):
        data_dict = dict(iteration_index=current_iteration.iteration_index)
        chunk_assignment = None

        # Each element in here contains a 3-tuple with the contents:
        # - `self._stream_cfg` node under which to look for keys.
        # - `data_dict` node under which to store results for each key.
        # - The associated record that is supposed to contain the keys.

        # This loop is a somewhat weird depth-first traversal of the openPMD
        # hierarchy. Since the hierarchy of the particle markup is one layer
        # deeper, the traversal of one particle species works different from
        # the traversal of one field type.
        remaining_nodes = [(self._stream_cfg, data_dict, current_iteration)]
        while remaining_nodes:
            (
                stream_cfg_node,
                data_dict_node,
                current_record,
            ) = remaining_nodes.pop()


            if isinstance(stream_cfg_node, list):
                # Process all leaf keys.

                # If we are here, then `current_record` is:
                #
                # 1. Either a container of field types
                # 2. Or a single particle species
                #
                # The chunk assignment for loading data must be consistent
                # across all datasets in a field type or particle species.
                # Due to the somewhat weird way that this loop works, the reset
                # of the chunk_assignment back to None must happen at different
                # places.

                # Need to reset the chunk assignment here when starting
                # with a new particle species
                if isinstance(current_record, io.ParticleSpecies):
                    chunk_assignment = None

                for key in stream_cfg_node:
                    # We need to save the chunk assignment in order to use it
                    # again for different components of the same species.
                    # If `current_record[key]` is field however, then one call
                    # to _get_data_from_key() loads the entire mesh and we must
                    # not use a predefined chunk assignment
                    if isinstance(current_record, io.Mesh_Container):
                        chunk_assignment = None
                    data, chunk_assignment = \
                        self._get_data_from_key(current_record, key, chunk_assignment)
                    if data is not None:
                        data_dict_node[key] = data
            else:
                # Add more remaining nodes, descend tree.
                for (key, stream_cfg_child) in stream_cfg_node.items():
                    child_found = False
                    # If we have an empty list, assume the key is the leaf.
                    if not stream_cfg_child:
                        stream_cfg_child = [key]
                        child_record = current_record
                        child_found = True
                    # Access outer-most values by attribute, after that
                    # only use `getitem`.
                    elif (
                            current_record is current_iteration
                            and hasattr(current_record, key)
                    ):
                        # Equivalent to calling either `current_record.meshes`
                        # or `current_record.particles`
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
