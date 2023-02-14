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

    assert npoints % ndim_y == 0, "npoints is not a multiple of {ndim_y}"

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


def generate_td(
    xy_func: Callable = lambda: generate_toy8(
        label_kind="all", npoints=512
    ),
    time_x_func: Callable = lambda X, t: X + T.tensor([t] * X.shape[-1]),
    time_y_func: Callable = lambda Y, t: Y + (Y > 0) * t,
    time_func_mode: str = "rel",
    time: T.Tensor = T.linspace(0, 10, 10),
) -> Generator:
    """
    Generate "time dependent" toy data, given time axis.

    Parameters
    ----------
    xy_func
        Function that generates the initial (X, Y).
    time_x_func
        Function must have signature `func(X: T.Tensor, t: float)` where
        X.shape = (npoints, ndim_x) and . Modify and
        return updates to X using time step. Function must *not* modify
        X in-place. See `time_func_mode` for more.
    time_y_func
        Same as `time_x_func` but for Y. Y.shape = (npoints, ndim_y)
    time_func_mode
        How to call time functions.

        "abs" : call as time_foo_func(start, time[i])
        "rel" : call as time_foo_func(last, dt)

        where `start` is the output of `xy_func()` and `last` is the
        X / Y from the previous time step time[i-1].

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
    if time_func_mode == "abs":
        X_0, Y_0 = xy_func()
    elif time_func_mode == "rel":
        X, Y = xy_func()
    else:
        raise ValueError(f"Illegal {time_func_mode=}")

    for i_time, v_time in enumerate(time):
        if time_func_mode == "rel":
            if i_time > 0:
                dt = time[i_time] - time[i_time - 1]
                X = time_x_func(X, dt)
                Y = time_y_func(Y, dt)
            yield X.clone().detach(), Y.clone().detach()
        elif time_func_mode == "abs":
            yield time_x_func(X_0, v_time), time_y_func(Y_0, v_time)


def arrays_from_itr(itr: Iterator):
    """
    Call iterator and returns arrays. We assume itr stops after `nsteps` calls.

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


def generate_td_array(*args, **kwds):
    """
    Returns
    -------
    X : (nsteps, npoints, ndim_x)
    Y : (nsteps, npoints, ndim_y)
    """
    return arrays_from_itr(generate_td(*args, **kwds))


def generate_fake_toy(npoints=5):
    """
    Deterministic toy data for testing.

    Returns
    -------
    X : (npoints, 3)
    Y : (npoints, 3)
    """
    X = np.ones((npoints, 3)) * np.arange(npoints)[:, None]
    return T.from_numpy(X), T.from_numpy(X * 10)


class ToyIterDataset(IterableDataset):
    """Yield stream of single (x,y) pairs. At any point in time call step() to
    increment time and thus change how (x,y) is created, until the next step()
    call.

    There is no concept of an epoch. To fake an epoch, use

    >>> ds=ToyIterDataset(...)
    >>> DataLoader(ds, batch_size=npoints)

    where npoints is the number of points which xy_func() generates.
    """
    def __init__(
        self,
        xy_func: Callable = lambda: generate_fake_toy(6),
        time_x_func: Callable = lambda x, t: x + t,
        time_y_func: Callable = lambda y, t: y + (y > 0) * t,
        dt: float = 1.0,
    ):
        """
        Parameters
        ----------
        xy_func
            Function that generates the initial (X, Y).
        time_x_func
            Function must have signature `func(x: T.Tensor, t: float)` where
            x.shape = (ndim_x,). Modify and
            return updates to x using time step. Function must *not* modify
            x in-place.
        time_y_func
            Same as `time_x_func` but for y. y.shape = (ndim_y,)
        dt
            Time step. We only implement the equivalent of time_func_mode="abs"
            from generate_td().
        """

        self.time = 0
        self.dt = dt
        self.X, self.Y = xy_func()
        self.time_x_func = time_x_func
        self.time_y_func = time_y_func

    def __iter__(self):
        for x, y in itertools.cycle(zip(self.X, self.Y)):
            xt = self.time_x_func(x, self.time)
            yt = self.time_y_func(y, self.time)
            yield xt, yt

    def step(self):
        self.time += self.dt


def iter_ds(ds: ToyIterDataset, batch_size: int, nsteps: int):
    dl_itr = iter(DataLoader(ds, batch_size=batch_size, shuffle=False))
    for _ in range(nsteps):
        yield next(dl_itr)
        ds.step()
