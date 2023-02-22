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
        # (x,y) paris
        self.mem = []

    def sample_batch(self, batch_size):
        # Returns min(len(mem), batch_size) elements
        batch = self.mem[:batch_size][:]
        if self.shuffle:
            random.shuffle(batch)
        return tuple(T.stack(tuple((xy[ii] for xy in batch))) for ii in [0, 1])

    # XXX The paper says pass n_obs, but shouldn't that better be an internal
    # counter here??
    def update_memory(self, X: T.Tensor, Y: T.Tensor, n_obs: int):
        """
        Algorithm 2 Reservoir sampling update

        Parameters
        ----------
        X, Y
            (batch) data to for memory update
        n_obs
            Number of data points observed so far.
        """
        jj = 0
        for x, y in zip(X, Y):
            if len(self.mem) < self.mem_size:
                self.mem.append((x, y))
            else:
                ii = random.randint(0, n_obs + jj)
                if ii < self.mem_size:
                    self.mem[ii] = (x, y)
            jj += 1
