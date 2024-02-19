## Copied from src/insituml/toy_data/memory.py from the last commit in this branch
## https://github.com/Photon-AI-Research/InSituML/tree/feature-fix-plotting-toy-cl
## required was seen based on example script in the same branch here:
## https://github.com/Photon-AI-Research/InSituML/blob/feature-fix-plotting-toy-cl/examples/streaming_toy_data/toy_cl.py
## also mentioned in this issue. 
## https://github.com/Photon-AI-Research/InSituML/issues/28

from collections import Counter
import random

import torch as T
# XXX seeds for shuffle not fixed
# XXX lots of copies and list ops, must be optimized obviously
class ExperienceReplay:
    """
    Naive implementation of

        On Tiny Episodic Memories in Continual Learning
        http://arxiv.org/abs/1902.10486
    """

    def __init__(self, mem_size: int, shuffle: bool = True):
        self.mem_size = mem_size
        self.shuffle = shuffle
        # (i_step, x, y) tuples
        self.mem = []

    def sample(self, batch_size):
        # Returns min(len(mem), batch_size) elements
        batch = self.mem[:batch_size][:]
        if self.shuffle:
            random.shuffle(batch)
        return tuple(
            T.stack(tuple((entry[ii] for entry in batch))) for ii in [1, 2]
        )

    # XXX The paper says pass n_obs, but shouldn't that better be an internal
    # counter here??
    def update_memory(self, X: T.Tensor, Y: T.Tensor, n_obs: int, i_step: int):
        """
        Algorithm 2 Reservoir sampling update

        Parameters
        ----------
        X, Y
            (batch) data to for memory update
        n_obs
            Number of data points observed so far.
        i_step
            Time step or "task" index. Only for book keeping.
        """
        jj = 0
        for x, y in zip(X, Y):
            if len(self.mem) < self.mem_size:
                self.mem.append((i_step, x, y))
            else:
                ii = random.randint(0, n_obs + jj)
                if ii < self.mem_size:
                    self.mem[ii] = (i_step, x, y)
            jj += 1

    def status(self):
        """Report number of samples ordered by i_step.

        Example
        -------
        6 steps, mem_size=512

        {0: 28, 1: 44, 2: 83, 3: 172, 4: 152, 5: 33}
        """
        c = Counter((entry[0] for entry in self.mem))
        return dict(c.most_common())
