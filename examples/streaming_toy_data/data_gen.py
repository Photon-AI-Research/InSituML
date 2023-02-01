from functools import wraps, partial
from typing import Callable, Generator

import numpy as np


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
    label_kind : {all, some, none}
    npoints
        npoints for all 8 clusters in total

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
    dt_pos_func: Callable = lambda x, dt: x + np.array([dt] * x.shape[-1]),
    dt_lab_func: Callable = lambda x, dt: x + (x > 0) * dt,
    ntime: int = 10,
    dt=1.0,
) -> Generator:

    """
    Parameters
    ----------
    pos_lab_func,
    dt_pos_func, dt_time_func
        Modify and return updates to positions / labels using time step.
    ntime
        Number of time steps.
    dt
        Time step.

    Returns
    -------
    Yield one tuple (positions, labels) per time step. See generate_toy8() for
    shapes.
    """
    assert ntime > 0, "ntime must be > 0"
    positions, labels = pos_lab_func()
    for idx in range(ntime):
        if idx > 0:
            positions = dt_pos_func(positions, dt)
            labels = dt_lab_func(labels, dt)
        yield positions.copy(), labels.copy()


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
