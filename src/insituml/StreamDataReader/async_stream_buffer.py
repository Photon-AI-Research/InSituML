import os
from queue import Empty, Queue
from threading import Thread
import time

from .stream_reader import StreamReader
from .StreamBuffer import get_data_locally


class AsyncStreamBuffer:
    def __init__(
            self,
            stream_path,
            stream_config_json=os.path.join(
                os.path.dirname(__file__),
                'stream_config.json',
            ),
            buffer_size=3,
            use_local_data=False,
            elp_time_max_size=10000,
            max_step_duration_sec=180,
    ):

        """Construct stream buffer.

        Parameters
        ----------
        buffer_size: int
            The amount of data iterations the buffer holds before
            reading the stream again
        """
        self._buffer_data = Queue(buffer_size)
        self.use_local_data = use_local_data
        if use_local_data:
            self._iteration_id = 0
        else:
            self._stream = StreamReader(stream_path, stream_config_json)
        self.elp_time = []
        self._elp_time_max_size = elp_time_max_size
        self._max_step_duration_sec = max_step_duration_sec

        self._last_data = None
        self._is_finished = False
        self._is_closed = False
        self._fill_buffer_thread = Thread(target=self._fill_buffer)
        self._fill_buffer_thread.start()

    @property
    def _buffer_size(self):
        return self._buffer_data.maxsize

    def __iter__(self):
        while True:
            self._last_data = self.get_next_data()
            if self._last_data is None:
                break
            yield self._last_data
            self._last_data = None

    def get_next_data(self):
        """Return one data iteration from the buffer. If the stream has
        been completely iterated, return `None`.

        Blocks to wait until data is available.
        """
        # If we stopped a previous iteration, first yield the value that
        # had already been fetched.
        if self._last_data is not None:
            data = self._last_data
            self._last_data = None
            return data

        # This is full of race condition in case of multiple consumers.
        # Since we are only accessing the queue from one consumer, it
        # does not matter, though.
        data = None
        if not self._is_finished or not self._buffer_data.empty():
            try:
                # We expect a simulation step takes less than 3 minutes.
                data = self._buffer_data.get(
                    timeout=self._max_step_duration_sec)
            except Empty:
                if not self._is_finished or not self._buffer_data.empty():
                    # We timed out but data should still/was still coming.
                    # So we re-raise the timeout exception.
                    raise
        return data

    def read_buffer(self):
        """Read the contents of the buffer. Buffer will be consumed.

        Returns
        -------
        List[Dict]
           all samples currently in the buffer
        """
        buf_data = []
        for i in range(self._buffer_size):
            try:
                data = self._buffer_data.get_nowait()
            except Empty:
                break

            buf_data.append(data)
        return buf_data

    def _fill_buffer(self):
        """Fill the buffer."""
        print("Filling buffer")
        while not self._is_closed:
            start = time.time()

            if not self.use_local_data:
                data_dict = self._stream.get_next_data()
            else:
                data_dict = get_data_locally(self._iteration_id)
            if data_dict is None:
                print("Finished iterations. Exiting Training and Validation")
                self._is_finished = True
                break
            if self.use_local_data:
                self._iteration_id += 1

            self.elp_time.append(time.time() - start)
            if len(self.elp_time) >= self._elp_time_max_size:
                # Reduce size of infinitely growing list.
                self.elp_time = [sum(self.elp_time) / len(self.elp_time)]
            self._buffer_data.put(data_dicts)

    def close(self):
        if self._is_closed:
            return

        self._is_closed = True
        self._fill_buffer_thread.join()
        self._stream.close()
        del self._buffer_data

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass
