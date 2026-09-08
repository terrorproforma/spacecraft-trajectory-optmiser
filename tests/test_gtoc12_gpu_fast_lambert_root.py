"""Interpolated Lambert roots retain branch choice and independent orbit closure."""
import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.ephemeris import propagate_kepler
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA"
)


@pytest.mark.parametrize("count", [31, 16385])
@pytest.mark.parametrize("iterations", [32, 64, 256])
@pytest.mark.parametrize("scan_samples", [16, 256])
def test_fast_root_matches_reference_and_closes_orbit(
    monkeypatch, count, iterations, scan_samples
):
    rng = np.random.default_rng(581)
    r1, r2 = (rng.normal(size=(count, 3)) for _ in range(2))
    for positions in (r1, r2):
        positions *= (C.AU_KM * rng.uniform(1, 3, count) / np.linalg.norm(
            positions, axis=1
        ))[:, None]
    days = rng.uniform(150, 800, count)
    days[:2] = [0, -1]
    r2[2] = r1[2]
    results = []
    with GpuLambert(count) as gpu:
        native = gpu.evaluate_hops

        def evaluate(*args):
            gpu.hops["lambert"]["iterations"] = iterations
            return native(*args)

        monkeypatch.setattr(gpu, "evaluate_hops", evaluate)
        for enabled in (0, 1, None):
            if enabled is None:
                monkeypatch.delenv("SPACEPDHCG_TEST_GTOC12_FAST_LAMBERT_ROOT")
            else:
                monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_FAST_LAMBERT_ROOT", str(enabled))
            results.append(gpu.screen_hops(
                r1, [0, 0, 0], r2, [0, 0, 0], 64328, days, scan_samples=scan_samples
            ))
    before, after, default = results
    np.testing.assert_array_equal(default.feasible, after.feasible)
    np.testing.assert_array_equal(default.departure_velocity, after.departure_velocity)
    np.testing.assert_array_equal(before.feasible, after.feasible)
    for name in (
        "departure_velocity", "arrival_velocity", "departure_delta_v", "arrival_delta_v"
    ):
        np.testing.assert_allclose(
            getattr(after, name), getattr(before, name), atol=1e-8, rtol=1e-9
        )
    valid = after.feasible
    if iterations >= 64:
        assert valid.any()
        if scan_samples == 256:
            assert valid[3:].all()
    if not valid.any():
        return
    position, velocity = propagate_kepler(
        r1[valid], after.departure_velocity[valid], days[valid] * C.DAY_S
    )
    assert np.max(np.linalg.norm(position - r2[valid], axis=1), initial=0) < 0.01
    assert np.max(
        np.linalg.norm(velocity - after.arrival_velocity[valid], axis=1), initial=0
    ) < 1e-8
