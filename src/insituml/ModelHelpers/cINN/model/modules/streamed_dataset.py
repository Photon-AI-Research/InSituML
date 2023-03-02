import torch
from torch.utils.data import IterableDataset
import numpy as np

from . import data_preprocessing
from . import streamed_radiation


class StreamedPCDataset(IterableDataset):
    def __init__(
            self,
            stream_reader,
            radiation_stream_reader,
            num_points=-1,
            species='e',
            normalize=False,
            a=0.,
            b=1.,
    ):
        '''
        Prepare dataset
        Args:
            stream_reader(Union[StreamReader, StreamBuffer]): A stream
                reader (optionally with buffer) to use for openPMD
                streaming.
            radiation_stream_reader(Union[StreamReader, StreamBuffer]):
                A stream reader (optionally with buffer) to use for
                openPMD radiation streaming.
            num_points(integer): Number of points to sample from each electron cloud.
                                 If -1, take a complete electron cloud.
            species(string): name of particle species to be loaded from openPMD file
            normalize(boolean): True if normalize each point to be in range [a, b]
        '''
        super().__init__()

        self._stream_reader = stream_reader
        print('initialized pc stream')
        self._last_data = self._stream_reader.get_next_data()
        assert self._last_data is not None, \
            'stream returned no data upon first read'
        print('got pc data')

        self._radiation_stream = streamed_radiation.RadiationDataStream(
            radiation_stream_reader)
        print('initialized rad stream')
        self._last_radiation_data = \
            self._radiation_stream.cache_next()
        print('got rad data')
        assert self._last_radiation_data is not None, \
            'radiation stream returned no data upon first read'

        self.get_data_phase_space = \
            data_preprocessing.get_data_phase_space_streamed
        self.get_data_radiation = \
            data_preprocessing.get_radiation_spectra_2_projections_streamed
        self.normalize = normalize
        self.num_points = num_points
        self.a, self.b = a, b
        self.species = species

        print('\nGet min/max from phase space data...')
        arr = self.get_data_phase_space(
            self._last_data, species=self.species)
        self.shape_ps = arr.shape
        print(f'{self.shape_ps = }')
        arr = arr[~np.isnan(arr).any(axis=1)]
        print(f'after NaN filtering: {arr.shape = }')

        self.vmin_ps = [np.min(arr[:, i]) for i in range(arr.shape[1])]
        self.vmax_ps = [np.max(arr[:, i]) for i in range(arr.shape[1])]
        #print(self.vmin_ps)

        self.vmin_ps = torch.tensor(self.vmin_ps).float()
        self.vmax_ps = torch.tensor(self.vmax_ps).float()
        
        print('PS Minima: ')
        print('\t', self.vmin_ps)
        
        print('PS Maxima: ')
        print('\t', self.vmax_ps)
        
        print('\nGet min/max from radiation data...')
        arr = self.get_data_radiation(self._radiation_stream)
        self.shape_rad = arr.shape
        self.vmin_rad = np.min(arr)
        self.vmax_rad = np.max(arr)
        print('Radiation Minima: ')
        print('\t', self.vmin_rad[0,0])
        
        print('Radiation Maxima: ')
        print('\t', self.vmax_rad[0,0])

    def _process_data(self, data):
        phase_space = self.get_data_phase_space(
            self._last_data, species=self.species)

        if self._radiation_stream.cache_is_empty:
            self._radiation_stream.cache_next()
        radiation = self.get_data_radiation(self._radiation_stream)
        self._radiation_stream.empty_cache()
        return phase_space, radiation

    def __iter__(self):
        # If we stopped a previous iteration, first yield the value that
        # had already been fetched. Same for the first element that we
        # prefetch for statistics calculation.
        if self._last_data is not None:
            yield self._process_data(self._last_data)
            self._last_data = None

        for self._last_data in self._stream_reader:
            yield self._process_data(self._last_data)
            # This seems redundant but if the iterator is destroyed
            # while we are waiting for the next element from the stream
            # reader and we did not have this, we would duplicate a
            # value. Also, this may free memory earlier.
            self._last_data = None

    def close(self):
        self._stream_reader.close()

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass
