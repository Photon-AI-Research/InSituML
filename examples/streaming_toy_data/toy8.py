from typing import Callable

import numpy as np
import torch as T


def generate(labels: str, tot_dataset_size: int) -> tuple[T.Tensor, T.Tensor]:

    assert tot_dataset_size % 8 == 0, "tot_dataset_size is not a multiple of 8"

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

    label_maps = {
        "all": [0, 1, 2, 3, 4, 5, 6, 7],
        "some": [0, 0, 0, 0, 1, 1, 2, 3],
        "none": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    rng = np.random.default_rng(seed=0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.empty((N, 2))
    onehot_labels = np.zeros((N, 8))
    n = N // 8

    for i, v in enumerate(verts):
        pos[i * n : (i + 1) * n, :] = rng.normal(loc=v, scale=0.2, size=(n, 2))
        onehot_labels[i * n : (i + 1) * n, mapping[i]] = 1.0

    return T.tensor(pos, dtype=T.float), T.tensor(onehot_labels, dtype=T.float)


def generate_time_dependent(
    labels: str,
    tot_dataset_size: int,
    fctn: Callable,
    t: np.ndarray = np.arange(10),
) -> tuple[T.Tensor, T.Tensor]:

    """
    Parameters
    ----------
    labels, tot_dataset_size :
        See generate()
    fctn :
        Map scalar time to length-2 array of additive adjustments for
        positions.
    t : (n_time,)
        Time axis.
    """

    pos, onehot_labels = generate(labels, tot_dataset_size)
    pos_t, onehot_labels_t = pos, onehot_labels

    for ti in t:
        pos_t = T.cat([pos_t, pos + fctn(ti)])
        onehot_labels_t = T.cat([onehot_labels_t, (ti + 1) * onehot_labels])

    return pos_t, onehot_labels_t


generate_timeDependent = generate_time_dependent
