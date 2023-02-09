from functools import wraps
from typing import Callable, Generator

import numpy as np


def _pol2cart(radius, phi):
    """
    Helper for more general toyN dataset.

    verts = np.array(_pol2cart(1, linspace(0, 2*pi, N, endpoint=False))).T
    """
    return (radius * np.cos(phi), radius * np.sin(phi))


def generate_toy8(
    label_kind: str,
    npoints: int,
    rng=np.random.default_rng(seed=None),
) -> tuple[np.ndarray, np.ndarray]:
    """
    One toy8 dataset in 2d: 8 clusters of points with varying labels per cluster.
    npoints points in total, so npoints / 8 per cluster.

    Parameters
    ----------
    label_kind
        "all" : each cluster has a different one-hot label
        "some" : some have the same, some a different label
        "none" : all clusters have label [1,0,0,...,0]
    npoints
        npoints for all 8 clusters in total
    rng
        numpy RNG instance

    Returns
    -------
    pos : (npoints, 2)
    onehot_labels : (npoints, 8)
    """

    assert npoints % 8 == 0, "npoints is not a multiple of 8"

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

    pos = np.empty((npoints, 2))
    onehot_labels = np.zeros((npoints, 8))
    # npoints per cluster (mode)
    nc = npoints // 8

    for ic, v in enumerate(verts):
        pos[ic * nc : (ic + 1) * nc, :] = rng.normal(
            loc=v, scale=0.2, size=(nc, 2)
        )
        onehot_labels[ic * nc : (ic + 1) * nc, mapping[ic]] = 1.0

    return pos, onehot_labels


def generate_td(
    pos_lab_func: Callable = lambda: generate_toy8(
        label_kind="all", npoints=512
    ),
    time_pos_func: Callable = lambda x, t: x + np.array([t] * x.shape[-1]),
    time_lab_func: Callable = lambda x, t: x + (x > 0) * t,
    time_func_mode: str = "rel",
    time: np.ndarray = np.linspace(0, 10, 10),
) -> Generator:
    """
    Generate "time dependent" toy data.

    Parameters
    ----------
    pos_lab_func
        Function that generates the initial (positions, labels).
    time_pos_func, time_lab_func
        Functions must have signature `func(x: np.ndarray, t: float)` where
        x.shape = (npoints, ndim_pos), ndim_pos=2 for toy8. Modify and return
        updates to positions / labels using time step. Functions must *not*
        modify `x` in-place. See `time_func_mode` for more.
    time_func_mode
        How to call time functions.

        "abs" : call as time_foo_func(start, time[i])
        "rel" : call as time_foo_func(last, dt)

        where `start` is the output of `pos_lab_func()` and `last` is the
        positions / labels from the previous time step time[i-1].

        Functions are called for each time step, therefore the first step is not treated
        special. For certain functions that do e.g. linear operations in t such as
        time_pos_func = lambda x,t: x + t, both modes are equal if time[0] = 0.0.
    time
        Time axis.

    Returns
    -------
    Yield one tuple (positions, labels) per time step. See generate_toy8() for
    shapes. The iterator is len(time) "long".
    """
    positions, labels = pos_lab_func()
    if time_func_mode == "abs":
        pos_0, lab_0 = positions.copy(), labels.copy()

    for i_time, v_time in enumerate(time):
        if time_func_mode == "rel":
            if i_time > 0:
                dt = time[i_time] - time[i_time - 1]
                positions = time_pos_func(positions, dt)
                labels = time_lab_func(labels, dt)
            yield positions.copy(), labels.copy()
        elif time_func_mode == "abs":
            positions = time_pos_func(pos_0, v_time)
            labels = time_lab_func(lab_0, v_time)
            yield positions, labels


@wraps(generate_td)
def generate_td_array(*args, **kwds):
    """
    Version of generate_td that returns arrays.

    Returns
    -------
    positions : (ntime, npoints, ndim_pos)
    labels : (ntime, npoints, ndim_lab)
    """
    lst = list(generate_td(*args, **kwds))
    pos = np.array([x[0] for x in lst])
    labels = np.array([x[1] for x in lst])
    return pos, labels
