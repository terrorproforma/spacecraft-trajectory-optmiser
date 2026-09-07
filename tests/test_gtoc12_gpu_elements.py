"""GPU-built ephemerides preserve screening and fixed-order retiming decisions."""

import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import AsteroidCatalogue, load_catalogue
from spacepdhcg.gtoc12.ephemeris import asteroid_state, earth_state
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.search import SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("source,target", [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)])
def test_elements_match_cpu_ephemeris_screening(source, target):
    # Circular, inclined, high-eccentricity and retrograde orbits, including
    # epochs on both sides of the element reference and partial final batches.
    cat = AsteroidCatalogue(
        np.arange(1, 4),
        np.full(3, 64328.0),
        np.array([2.2, 2.8, 3.1]) * C.AU_KM,
        np.array([0.0, 0.2, 0.85]),
        np.array([0.0, 0.6, 2.9]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.3, 0.7]),
        np.array([0.0, 2.0, 5.0]),
        "synthetic",
    )
    departures = np.array([62000.0, 64328.0, 66000.0, 68000.0])
    tofs = np.array([60.0, 180.0, 400.0, 700.0, 900.0])
    dep, tof = np.repeat(departures, len(tofs)), np.tile(tofs, len(departures))

    def state(body, epochs):
        return earth_state(epochs) if body == 0 else asteroid_state(cat, body, epochs)

    r1, v1 = state(source, dep)
    r2, v2 = state(target, dep + tof)
    with using_lambert_backend("cuda", maximum_batch_size=7) as gpu:
        reference = gpu.screen_hops(
            r1,
            v1,
            r2,
            v2,
            dep,
            tof,
            departure_allowance_km_s=6.0 if source == 0 else 0.0,
            arrival_allowance_km_s=6.0 if target == 0 else 0.0,
        )
        actual, feasible = gpu.leg_table(cat, source, target, departures, tofs)
        np.testing.assert_array_equal(feasible.ravel(), reference.feasible)
        np.testing.assert_allclose(actual.ravel(), reference.total_delta_v, rtol=2e-9, atol=2e-8)
        assert gpu.telemetry["completed_element_hops"] == len(dep)
        # Scratch-buffer reuse must leave the original interface reusable.
        repeat = gpu.screen_hops(r1, v1, r2, v2, dep, tof)
        assert np.all(repeat.feasible == reference.feasible)


def test_retimer_does_not_compute_cpu_states(monkeypatch):
    cat = load_catalogue()
    with using_lambert_backend("cuda") as gpu:
        retimer = Retimer(cat, SearchSettings())
        monkeypatch.setattr(retimer, "_state", lambda *args: pytest.fail("CPU ephemeris used"))
        dv, feasible = retimer.leg_table(45738, 25792, "deploy_hop")
        assert np.count_nonzero(feasible) > 0
        assert gpu.telemetry["completed_element_hops"] == dv.size
        count = gpu.evaluations
        assert retimer.leg_table(45738, 25792, "deploy_hop")[0] is dv
        assert gpu.evaluations == count


def test_invalid_elements_and_times_recover():
    cat = load_catalogue()
    with using_lambert_backend("cuda") as gpu:
        dv, feasible = gpu.leg_table(cat, 1, 2, [np.nan], [100.0, -1.0, 0.0])
        assert not feasible.any() and np.isinf(dv).all()
        with pytest.raises(ValueError, match="vectors"):
            gpu.leg_table(cat, 1, 2, [[64328.0]], [100.0])
        dv, feasible = gpu.leg_table(cat, 1, 2, [64328.0], [100.0])
        assert feasible.any() and np.isfinite(dv).all()
