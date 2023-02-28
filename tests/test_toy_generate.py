import itertools
from pathlib import Path
import subprocess as sp

import pytest
import torch as T

from insituml.toy_data import generate


@pytest.mark.parametrize(
    "time_func_mode, label_kind",
    itertools.product(("abs", "rel"), ("all", "some", "none")),
)
def test_td_api(time_func_mode, label_kind):
    npoints = 16
    nsteps = 20
    dt = 4.0

    X, Y = generate.generate_toy8(
        label_kind=label_kind,
        npoints=npoints,
        seed=None,
        scale=0.1,
    )

    ps, ls = generate.td_arrays(
        X,
        Y,
        time_func_mode=time_func_mode,
        time=T.linspace(0, nsteps - 1, nsteps) * dt,
    )

    assert ps.shape == (nsteps, npoints, 2)
    assert ls.shape == (nsteps, npoints, 8)


def test_td_time_func_mode():
    """
    Note that abs and rel mode are only equal for time_{x,y}_func linear in t.
    """
    npoints = 32
    nsteps = 20
    dt = 4.0

    def gen(time_func_mode):
        Xt, Yt = generate.td_arrays(
            *generate.generate_toy8(
                label_kind="all",
                npoints=npoints,
                seed=123,
            ),
            time_func_mode=time_func_mode,
            time=T.linspace(0, nsteps - 1, nsteps) * dt,
            time_x_func=lambda X, t: X + 2 * t,
            time_y_func=lambda X, t: X + (X > 0) * 2 * t,
        )
        return Xt, Yt

    for aa, bb in zip(gen("abs"), gen("rel")):
        T.testing.assert_close(aa, bb)


def test_tdds_api():
    npoints = 32
    nsteps = 20
    dt = 4.0

    X, Y = generate.generate_toy8(label_kind="all", npoints=npoints, seed=123)

    generate.TimeDependentTensorDataset(
        X,
        Y,
        dt=dt,
        time_x_func=lambda x, t: x + T.sin(2 * T.tensor(t)),
        time_y_func=lambda x, t: x + (x > 0) * T.cos(2 * T.tensor(t)) ** 2,
    )

    # pos args
    generate.TimeDependentTensorDataset(X, Y)


def test_ttds():
    npoints = 32
    nsteps = 20
    dt = 4.0
    X, Y = generate.generate_toy8(label_kind="all", npoints=npoints, seed=123)

    def td_arrays():
        Xt, Yt = generate.td_arrays(
            X,
            Y,
            time_func_mode="abs",
            time=T.linspace(0, nsteps - 1, nsteps) * dt,
            time_x_func=lambda X, t: X + T.sin(2 * t),
            time_y_func=lambda X, t: X + (X > 0) * T.cos(2 * t) ** 2,
        )
        return Xt, Yt

    # When t is scaler, we must use T.some_function(T.tensor(t)) ... ok.
    def tdds_arrays():
        ds = generate.TimeDependentTensorDataset(
            X,
            Y,
            dt=dt,
            time_x_func=lambda x, t: x + T.sin(2 * T.tensor(t)),
            time_y_func=lambda x, t: x + (x > 0) * T.cos(2 * T.tensor(t)) ** 2,
        )
        return generate.tdds_arrays(ds=ds, nsteps=nsteps, batch_size=npoints)

    for aa, bb in zip(td_arrays(), tdds_arrays()):
        T.testing.assert_close(aa, bb)


@pytest.mark.examples
def test_run_examples():
    path = Path(__file__).parent / "../examples/streaming_toy_data/"
    for script in path.glob("*.py"):
        sp.run(script, check=True)
