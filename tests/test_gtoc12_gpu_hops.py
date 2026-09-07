"""Combined GPU screening preserves CPU costs and removes host numerical work."""

import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12 import screening
from spacepdhcg.gtoc12.gpu_lambert import HOP_REQUEST, HOP_RESULT, GpuLambert
from spacepdhcg.gtoc12.lambert import completed_branch_requests, using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("allowance", [0.0, 6.0, 1e6])
def test_gpu_hops_costs_selection_and_reuse(monkeypatch, allowance):
    rng = np.random.default_rng(188)
    n = 257
    r1, r2 = (rng.normal(size=(n, 3)) * C.AU_KM for _ in range(2))
    b1, b2 = (rng.normal(size=(n, 3)) * 20 for _ in range(2))
    tof = rng.uniform(100, 900, n)
    tof[:2] = [0, -1]
    r2[2] = r1[2]
    kwargs = dict(departure_allowance_km_s=allowance, arrival_allowance_km_s=allowance)
    before_cpu = completed_branch_requests()
    expected = screening.lambert_hops(r1, b1, r2, b2, 64328.0, tof, **kwargs)

    assert completed_branch_requests() - before_cpu == 2 * n

    def forbidden(*args, **kwargs):
        raise AssertionError("host Lambert or numerical cost selection used")

    monkeypatch.setattr(screening, "lambert_batch", forbidden)
    monkeypatch.setattr(screening, "_credit", forbidden)
    before_gpu = completed_branch_requests()
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        for _ in range(2):
            actual = screening.lambert_hops(r1, b1, r2, b2, 64328.0, tof, **kwargs)
            np.testing.assert_array_equal(actual.feasible, expected.feasible)
            for name in [
                "departure_delta_v",
                "arrival_delta_v",
                "departure_velocity",
                "arrival_velocity",
            ]:
                np.testing.assert_allclose(
                    getattr(actual, name), getattr(expected, name), atol=1e-8, rtol=1e-9
                )
        assert gpu.batches == 18 and gpu.evaluations == 4 * n
    assert completed_branch_requests() - before_gpu == 4 * n
    assert HOP_REQUEST.itemsize == 160 and HOP_RESULT.itemsize == 72


def test_gpu_hops_invalid_body_and_allowance():
    r1 = np.array([[C.AU_KM, 0, 0]], dtype=float)
    r2 = np.array([[0, 2 * C.AU_KM, 0]], dtype=float)
    with GpuLambert(8) as gpu:
        bad = gpu.screen_hops(r1, [np.nan, 0, 0], r2, [0, 0, 0], 64328, 300)
        assert not bad.feasible.any()
        assert np.isinf(bad.total_delta_v).all() and np.isnan(bad.departure_velocity).all()
        for allowance in [-1, np.inf, np.nan]:
            with pytest.raises(ValueError, match="allowances"):
                gpu.screen_hops(
                    r1, [0, 0, 0], r2, [0, 0, 0], 64328, 300, arrival_allowance_km_s=allowance
                )
        assert gpu.screen_hops(r1, [0, 0, 0], r2, [0, 0, 0], 64328, 300).feasible.all()
        empty = gpu.screen_hops(np.empty((0, 3)), [0, 0, 0], np.empty((0, 3)), [0, 0, 0], [], [])
        assert len(empty.feasible) == 0


def test_gpu_workspace_retries_after_recreation_failure(monkeypatch):
    r1 = np.array([[C.AU_KM, 0, 0]], dtype=float)
    r2 = np.array([[0, 2 * C.AU_KM, 0]], dtype=float)
    with GpuLambert(8) as gpu:
        gpu.screen_hops(r1, [0, 0, 0], r2, [0, 0, 0], 64328, 300)
        create = gpu.create
        monkeypatch.setattr(gpu, "create", lambda *args: 1)
        with pytest.raises(RuntimeError, match="native status"):
            gpu.screen_hops(r1, [0, 0, 0], r2, [0, 0, 0], 64328, 300, scan_samples=127)
        assert not gpu.handle.value
        monkeypatch.setattr(gpu, "create", create)
        assert gpu.screen_hops(r1, [0, 0, 0], r2, [0, 0, 0], 64328, 300).feasible.all()
