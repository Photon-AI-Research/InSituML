from functools import wraps
import itertools
from typing import Callable, Generator, Iterator

import numpy as np

import torch as T
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import IterableDataset, DataLoader


def _pol2cart(radius, phi):
    return (radius * np.cos(phi), radius * np.sin(phi))


def points_on_circle(radius, npoints):
    """
    Generate equidistant points on a circle.

    Note that torch.linspace() doesn't have the `endpoint` arg since it is not
    compliant to the Python array API standard [1], so we need to do it in
    numpy and convert.

    [1] https://github.com/pytorch/pytorch/issues/70919
    """
    return T.from_numpy(
        np.array(
            _pol2cart(
                radius, np.linspace(0, 2 * np.pi, npoints, endpoint=False)
            ),
            dtype=np.float32,
        ).T
    )


def generate_fake_toy(npoints=5):
    """
    Deterministic toy data for testing.

    Returns
    -------
    X : (npoints, 3)
    Y : (npoints, 3)
    """
    X = T.ones((npoints, 3)) * T.arange(npoints)[:, None]
    return X, X * 10


def generate_toy8(
    label_kind: str,
    npoints: int,
    scale: float = 0.2**2.0,
    seed: int = None,
) -> tuple[T.Tensor, T.Tensor]:
    """
    One toy8 dataset in 2d: 8 clusters of points with varying labels per
    cluster. npoints points in total, so npoints / 8 per cluster.

    Parameters
    ----------
    label_kind
        "all" : each cluster has a different one-hot label
        "some" : some have the same, some a different label
        "none" : all clusters have label [1,0,0,...,0]
    npoints
        Number of points for all 8 clusters in total
    scale
        Variance for MVN (cov = simga**2 * eye(...), scale=sigma**2)
    seed
        Random seed, use None to not fix seed.

    Returns
    -------
    X : (npoints, 2)
        2d multi-modal Gaussian samples
    Y : (npoints, 8)
        one-hot labels
    """

    ndim_x = 2
    ndim_y = 8

    assert npoints % ndim_y == 0, f"npoints is not a multiple of {ndim_y}"

    if seed is not None:
        T.manual_seed(seed)

    verts = [
        (-2.4142, 1.0),
        (-1.0, 2.4142),
        (1.0, 2.4142),
        (2.4142, 1.0),
        (2.4142, -1.0),
        (1.0, -2.4142),
        (-1.0, -2.4142),
        (-2.4142, -1.0),
    ]

    # label_maps[label_kind][ic] = one-hot position index per cluster ic
    label_maps = {
        "all": [0, 1, 2, 3, 4, 5, 6, 7],
        "some": [0, 0, 0, 0, 1, 1, 2, 3],
        "none": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    mapping = label_maps[label_kind]

    X = T.empty((npoints, ndim_x))
    Y = T.zeros((npoints, ndim_y))
    # npoints per cluster (mode)
    nc = npoints // ndim_y

    for ic, v in enumerate(verts):
        mvn = MultivariateNormal(
            loc=T.tensor(v), covariance_matrix=scale * T.eye(ndim_x)
        )
        X[ic * nc : (ic + 1) * nc, :] = mvn.sample([nc])
        Y[ic * nc : (ic + 1) * nc, mapping[ic]] = 1.0

    return X, Y


def td_gen(
    X: T.Tensor = None,
    Y: T.Tensor = None,
    xy_func: Callable=None,
    time_x_func: Callable = lambda X, t: X + T.tensor([t] * X.shape[-1]),
    time_y_func: Callable = lambda Y, t: Y + (Y > 0) * t,
    time_func_mode: str = "rel",
    time: T.Tensor = T.linspace(0, 10, 10),
) -> Generator:
    """
    Generator for "time dependent" arrays given time axis.

    Parameters
    ----------
    xy_func
        Function that generates the initial (X, Y), such as
        lambda: generate_toy8(label_kind="all", npoints=512)
    X, Y
        Initial (X,Y). Use this or xy_func.
    time_x_func
        Function must have signature `func(X: T.Tensor, t: float)` where
        X.shape = (npoints, ndim_x). Modify and return updates to X using time
        step. Function must *not* modify X in-place. See `time_func_mode` for
        more.
    time_y_func
        Same as `time_x_func` but for Y. Y.shape = (npoints, ndim_y)
    time_func_mode
        How to call time functions.

        "abs" : call as time_foo_func(start, time[i])
        "rel" : call as time_foo_func(last, dt)

        where `start` is the initial X (Y) and `last` is the X (Y) from the
        previous time step time[i-1].

        Functions are called for each time step, therefore the first step is
        not treated special. For certain functions that do e.g. linear
        operations in t such as time_x_func = lambda x,t: x + t, both modes
        are equal if time[0] = 0.0.
    time
        Time axis.

    Returns
    -------
    Yield one tuple (X, Y) per time step. See generate_toy8() for
    shapes. The iterator is len(time) "long".
    """
    if xy_func is not None:
        assert [X, Y].count(None) == 2, "xy_func given, X and Y must be None"
        X, Y = xy_func()

    for i_time, v_time in enumerate(time):
        if time_func_mode == "rel":
            if i_time > 0:
                dt = time[i_time] - time[i_time - 1]
                X = time_x_func(X, dt)
                Y = time_y_func(Y, dt)
            yield X.clone().detach(), Y.clone().detach()
        elif time_func_mode == "abs":
            yield time_x_func(X, v_time), time_y_func(Y, v_time)
        else:
            raise ValueError(f"Illegal {time_func_mode=}")


def arrays_from_itr(itr: Iterator):
    """
    Call iterator and return arrays. We assume itr stops after `nsteps` calls.

    Parameters
    ----------
    itr
        Must yield a 2-tuple of tensors, (x,y)

    Returns
    -------
    X : (nsteps, *x.shape)
    Y : (nsteps, *y.shape)
    """
    lst = list(itr)
    X = T.stack([x[0] for x in lst])
    Y = T.stack([x[1] for x in lst])
    return X, Y


@wraps(td_gen)
def td_arrays(*args, **kwds):
    """
    Wrapper of td_gen() that returns arrays. Useful to generate arrays for
    visualization.

    Returns
    -------
    X : (nsteps, npoints, ndim_x)
    Y : (nsteps, npoints, ndim_y)
    """
    return arrays_from_itr(td_gen(*args, **kwds))


class TimeDependentDataset(IterableDataset):
    """Yield stream of single (x,y) pairs. At any point in time call step() to
    increment time by dt and thus change how (x,y) is created, until the next
    step() call.

    With cycle=False, this is like TensorDataset(X, Y), only that (x,y) are
    modified over time in repeated iterations in epochs as in

        ds = TimeDependentDataset(..., cycle=False)
        dl = DataLoader(ds, ...)
        for i_epoch in range(5):
            for i_batch, (x, y) in enumerate(dl):
                print(i_epoch, i_batch, ds.time, x, y)
            ds.step()

    Note that step() can be called anywhere, i.e. also every N'th epoch or
    whatnot.

    With cycle=True, there is no concept of an epoch, i.e. the iterator is
    infinite. To fetch data from a whole epoch in this case, use

        ds = TimeDependentDataset(..., cycle=True)
        DataLoader(ds, batch_size=X.shape[0])
    """

    def __init__(
        self,
        X: T.Tensor=None,
        Y: T.Tensor=None,
        xy_func: Callable = None,
        time_x_func: Callable = lambda x, t: x + t,
        time_y_func: Callable = lambda y, t: y + (y > 0) * t,
        dt: float = 1.0,
        cycle: bool = True,
    ):
        """
        Parameters
        ----------
        xy_func
            Function that generates the initial (X, Y), such as
            lambda: generate_toy8(label_kind="all", npoints=512)
        X, Y
            Initial (X,Y). Use this or xy_func.
        time_x_func
            Function must have signature `func(x: T.Tensor, t: float)` where
            x.shape = (ndim_x,). Modify and return updates to x using current
            absolute time value `t`. Function must *not* modify x in-place.
        time_y_func
            Same as `time_x_func` but for y. y.shape = (ndim_y,)
        dt
            Time step. We only implement the equivalent of time_func_mode="abs"
            from td_gen().
        cycle
            Whether or not to emulate epochs.
        """
        if xy_func is not None:
            assert [X, Y].count(None) == 2, "xy_func given, X and Y must be None"
            X, Y = xy_func()
        self.X, self.Y = X, Y
        self.time = 0
        self.dt = dt
        self.time_x_func = time_x_func
        self.time_y_func = time_y_func
        self.cycle = cycle

        self.npoints = self.X.shape[0]
        assert (
            self.Y.shape[0] == self.npoints
        ), "X and Y have not the same npoints"

    def _get_xy_itr(self):
        if self.cycle:
            return itertools.cycle(zip(self.X, self.Y))
        else:
            return (
                (self.X[ii, :], self.Y[ii, :]) for ii in range(self.X.shape[0])
            )

    def __iter__(self):
        for x, y in self._get_xy_itr():
            xt = self.time_x_func(x, self.time)
            yt = self.time_y_func(y, self.time)
            yield xt, yt

    def step(self):
        self.time += self.dt


# FIXME (StS)
#
# Time axis equivalent to
#   linspace(0, nsteps, nsteps) * dt
# or
#   linspace(0, nsteps-1, nsteps) * dt
# ?? Calling ds.step() after yield might be wrong. Check against logic in
# td_gen().
#
def tdds_gen(ds: TimeDependentDataset, nsteps: int, batch_size: int = None):
    """
    Same as td_gen() but using TimeDependentDataset.

    The time axis is defined by nsteps and ds.dt

    To extract all data, we use batch_size=ds.npoints by default.
    """
    if batch_size is None:
        batch_size = ds.npoints
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if ds.cycle:
        dl_itr = iter(dl)
        for _ in range(nsteps):
            yield next(dl_itr)
            ds.step()
    else:
        for _ in range(nsteps):
            for x, y in dl:
                yield (x, y)
            ds.step()


@wraps(tdds_gen)
def tdds_arrays(*args, **kwds):
    return arrays_from_itr(tdds_gen(*args, **kwds))
