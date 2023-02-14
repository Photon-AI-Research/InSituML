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

    ps, ls = data_gen.generate_td_array(
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
        ps, ls = data_gen.generate_td_array(
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
def test_toy_iter_dataset(cycle):
    npoints = 32
    nsteps = 20
    dt = 4.0
    xy_func = lambda: data_gen.generate_toy8(
        label_kind="all",
        npoints=npoints,
        seed=123,
    )

    def gen_functional():
        ps, ls = data_gen.generate_td_array(
            xy_func=xy_func,
            time_func_mode="abs",
            time=T.linspace(0, nsteps - 1, nsteps) * dt,
        )
        return ps, ls

    def gen_toy_iter_dataset():
        ds = data_gen.ToyIterDataset(
            xy_func=xy_func,
            dt=dt,
            cycle=cycle,
        )
        return data_gen.arrays_from_itr(data_gen.iter_ds(ds, npoints, nsteps))

    for aa, bb in zip(gen_functional(), gen_toy_iter_dataset()):
        T.testing.assert_close(aa, bb)
