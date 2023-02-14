import itertools

import pytest
import torch as T

import data_gen


@pytest.mark.parametrize(
    "time_func_mode, label_kind",
    itertools.product(("abs", "rel"), ("all", "some", "none")),
)
def test_functions_api(time_func_mode, label_kind):
    npoints = 16
    nsteps = 20
    dt = 4.0

    ps, ls = data_gen.td_arrays(
        xy_func=lambda: data_gen.generate_toy8(
            label_kind=label_kind,
            npoints=npoints,
            seed=None,
            scale=0.1,
        ),
        time_func_mode=time_func_mode,
        time=T.linspace(0, nsteps - 1, nsteps) * dt,
    )

    assert ps.shape == (nsteps, npoints, 2)
    assert ls.shape == (nsteps, npoints, 8)


def test_time_func_mode():
    npoints = 32
    nsteps = 20
    dt = 4.0

    def gen(time_func_mode):
        ps, ls = data_gen.td_arrays(
            xy_func=lambda: data_gen.generate_toy8(
                label_kind="all",
                npoints=npoints,
                seed=123,
            ),
            time_func_mode=time_func_mode,
            time=T.linspace(0, nsteps - 1, nsteps) * dt,
        )
        return ps, ls

    for aa, bb in zip(gen("abs"), gen("rel")):
        T.testing.assert_close(aa, bb)


@pytest.mark.parametrize("cycle", [True, False])
def test_time_dependent_dataset(cycle):
    npoints = 32
    nsteps = 20
    dt = 4.0
    xy_func = lambda: data_gen.generate_toy8(
        label_kind="all",
        npoints=npoints,
        seed=123,
    )

    def td_arrays():
        ps, ls = data_gen.td_arrays(
            xy_func=xy_func,
            time_func_mode="abs",
            time=T.linspace(0, nsteps - 1, nsteps) * dt,
        )
        return ps, ls

    def tdds_arrays():
        ds = data_gen.TimeDependentDataset(
            xy_func=xy_func,
            dt=dt,
            cycle=cycle,
        )
        return data_gen.tdds_arrays(ds=ds, nsteps=nsteps, batch_size=npoints)

    for aa, bb in zip(td_arrays(), tdds_arrays()):
        T.testing.assert_close(aa, bb)
