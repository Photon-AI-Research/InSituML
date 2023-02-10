import itertools

import pytest
import torch as T

import data_gen


@pytest.mark.parametrize(
    "time_func_mode, label_kind",
    itertools.product(("abs", "rel"), ("all", "some", "none")),
)
def test_data_gen_functions_api(time_func_mode, label_kind):
    npoints = 16
    ntime = 20
    dt = 4.0

    ps, ls = data_gen.generate_td_array(
        pos_lab_func=lambda: data_gen.generate_toy8(
            label_kind=label_kind,
            npoints=npoints,
            seed=None,
            scale=0.1,
        ),
        time_func_mode=time_func_mode,
        time=T.linspace(0, ntime - 1, ntime) * dt,
    )

    assert ps.shape == (ntime, npoints, 2)
    assert ls.shape == (ntime, npoints, 8)


def test_data_gen_time_func_mode():
    npoints = 32
    ntime = 20
    dt = 4.0

    def gen(time_func_mode):
        ps, ls = data_gen.generate_td_array(
            pos_lab_func=lambda: data_gen.generate_toy8(
                label_kind="all",
                npoints=npoints,
                seed=123,
            ),
            time_func_mode=time_func_mode,
            time=T.linspace(0, ntime - 1, ntime) * dt,
        )
        return ps, ls

    for aa, bb in zip(gen("abs"), gen("rel")):
        T.testing.assert_close(aa, bb)
