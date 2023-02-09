import itertools

import pytest
import numpy as np

import data_gen


@pytest.mark.parametrize(
    "func_mode, label_kind",
    itertools.product(("abs", "rel"), ("all", "some", "none")),
)
def test_np_data_gen_api(func_mode, label_kind):
    npoints = 16
    ntime = 20
    dt = 4.0

    ps, ls = data_gen.generate_td_array(
        pos_lab_func=lambda: data_gen.generate_toy8(
            label_kind=label_kind,
            npoints=npoints,
            rng=np.random.default_rng(123),
        ),
        time_func_mode=func_mode,
        time=np.linspace(0, 19, ntime) * dt,
    )

    assert ps.shape == (ntime, npoints, 2)
    assert ls.shape == (ntime, npoints, 8)


def test_np_data_gen_func_mode():
    npoints = 32
    ntime = 20
    dt = 4.0

    def gen(func_mode):
        ps, ls = data_gen.generate_td_array(
            pos_lab_func=lambda: data_gen.generate_toy8(
                label_kind="all",
                npoints=npoints,
                rng=np.random.default_rng(123),
            ),
            time_func_mode=func_mode,
            time=np.linspace(0, 19, ntime) * dt,
        )
        return ps, ls

    for aa, bb in zip(gen("abs"), gen("rel")):
        np.testing.assert_allclose(aa, bb)
