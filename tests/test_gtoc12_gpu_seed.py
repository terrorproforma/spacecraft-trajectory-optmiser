"""GPU seed parity with the independent NumPy Lambert/Kepler reference."""

from __future__ import annotations

import ctypes as ct
import os
from dataclasses import replace

import numpy as np
import pytest
from test_gtoc12_gpu_discretisation import synthetic_boundary

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.gpu_discretisation import _DoublePointer, _pointer
from spacepdhcg.gtoc12.low_thrust import DU_KM, TU_S, VU_KM_S, _ballistic_reference

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires serialized native GPU seed runtime",
)


@pytest.mark.parametrize(
    "free_dep,free_arr", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("case", ["transfer", "long", "collinear", "hyperbolic", "dense"])
def test_seed_matches_lambert_kepler_reference(case, free_dep, free_arr):
    boundary = replace(
        synthetic_boundary(), free_departure_vinf=free_dep, free_arrival_vinf=free_arr
    )
    if case == "long":
        boundary = replace(
            boundary,
            departure_velocity=-boundary.departure_velocity,
            arrival_velocity=-boundary.arrival_velocity,
        )
    elif case == "collinear":
        boundary = replace(boundary, arrival_position=boundary.departure_position * 1.2)
    elif case == "hyperbolic":
        boundary = replace(boundary, arrival_epoch=boundary.departure_epoch + 5)
    days = np.linspace(0, boundary.duration_days, 2001 if case == "dense" else 19)
    times = np.ascontiguousarray(days * C.DAY_S / TU_S)
    b = np.ascontiguousarray(
        np.r_[
            boundary.departure_position / DU_KM,
            boundary.departure_velocity / VU_KM_S,
            boundary.arrival_position / DU_KM,
            boundary.arrival_velocity / VU_KM_S,
        ]
    )
    expected_states, expected_controls = _ballistic_reference(
        boundary, times, boundary.initial_mass
    )
    states, controls = np.empty_like(expected_states), np.empty_like(expected_controls)
    library = ct.CDLL(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"])
    launch = library.spacepdhcg_gtoc12_seed_evaluate_host
    launch.argtypes = [
        ct.c_int,
        _DoublePointer,
        _DoublePointer,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        _DoublePointer,
        _DoublePointer,
    ]
    launch.restype = ct.c_int
    code = launch(
        len(times),
        _pointer(times),
        _pointer(b),
        free_dep,
        free_arr,
        C.MAX_VINF_EARTH_KM_S / VU_KM_S,
        _pointer(states),
        _pointer(controls),
    )
    assert code == 0
    np.testing.assert_allclose(states, expected_states, atol=5e-12, rtol=5e-12)
    np.testing.assert_array_equal(controls, expected_controls)
