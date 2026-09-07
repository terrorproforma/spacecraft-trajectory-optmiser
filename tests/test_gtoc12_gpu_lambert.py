"""CUDA candidate screening: CPU parity, independent Kepler closure and scopes."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.ephemeris import propagate_kepler
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert
from spacepdhcg.gtoc12.lambert import lambert_batch, using_lambert_backend

GPU = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def fixture(n=129):
    rng = np.random.default_rng(178)
    r1 = rng.normal(size=(n, 3))
    r1 *= C.AU_KM * np.linspace(1, 3, n)[:, None] / np.linalg.norm(r1, axis=1)[:, None]
    r2 = rng.normal(size=(n, 3))
    r2 *= C.AU_KM * np.linspace(3, 1, n)[:, None] / np.linalg.norm(r2, axis=1)[:, None]
    tof = rng.uniform(150, 800, n) * C.DAY_S
    return r1, r2, tof


@GPU
@pytest.mark.parametrize("scan", [127, 256, 8192])
@pytest.mark.parametrize("long_way", [False, True])
def test_cuda_lambert_matches_cpu_and_closes_orbits(scan, long_way):
    r1, r2, tof = fixture()
    expected = lambert_batch(r1, r2, tof, long_way=long_way, scan_samples=scan)
    with GpuLambert(31) as gpu:
        actual = gpu.solve(r1, r2, tof, C.MU_SUN_KM3_S2, long_way=long_way, scan_samples=scan)
        assert gpu.batches == 5 and gpu.evaluations == len(r1)
    np.testing.assert_array_equal(actual.feasible, expected.feasible)
    for name in ["departure_velocity", "arrival_velocity"]:
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-9, atol=1e-9
        )
    r, v = propagate_kepler(
        r1[actual.feasible], actual.departure_velocity[actual.feasible], tof[actual.feasible]
    )
    assert np.max(np.linalg.norm(r - r2[actual.feasible], axis=1)) < 0.01
    assert np.max(np.linalg.norm(v - actual.arrival_velocity[actual.feasible], axis=1)) < 1e-8


@GPU
def test_scope_reuses_gpu_and_restores_cpu(monkeypatch):
    from spacepdhcg.gtoc12 import lambert

    r1, r2, tof = fixture(33)
    expected = lambert_batch(r1, r2, tof, scan_samples=256)
    cpu = lambert._residual

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU Lambert arithmetic was invoked")

    with using_lambert_backend("cuda", maximum_batch_size=32) as gpu:
        monkeypatch.setattr(lambert, "_residual", forbidden)
        for _ in range(2):
            actual = lambert_batch(r1, r2, tof, scan_samples=256)
            np.testing.assert_array_equal(actual.feasible, expected.feasible)
        assert gpu.batches == 4 and gpu.evaluations == 66
        with using_lambert_backend("numpy"):
            with pytest.raises(AssertionError, match="CPU Lambert"):
                lambert_batch(r1, r2, tof, scan_samples=256)
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError, match="creating thread"):
                pool.submit(gpu.solve, r1, r2, tof, C.MU_SUN_KM3_S2).result()
    monkeypatch.setattr(lambert, "_residual", cpu)
    restored = lambert_batch(r1, r2, tof, scan_samples=256)
    np.testing.assert_array_equal(restored.departure_velocity, expected.departure_velocity)


@GPU
def test_invalid_requests_never_become_feasible_and_buffers_recover():
    r1, r2, tof = fixture(7)
    r1[0] = np.nan
    r2[1] = r1[1]
    tof[2:5] = [0, -1, np.inf]
    r1[5] = 0
    with GpuLambert(8) as gpu:
        actual = gpu.solve(r1, r2, tof, C.MU_SUN_KM3_S2, scan_samples=256)
        assert not actual.feasible[:6].any() and actual.feasible[6]
        assert np.isnan(actual.departure_velocity[:6]).all()
        for bad in [float("inf"), float("nan"), 0, -1]:
            with pytest.raises(ValueError):
                gpu.solve(r1, r2, tof, bad)
        with pytest.raises(ValueError):
            gpu.solve(r1, r2, tof, C.MU_SUN_KM3_S2, scan_samples=0)
        r1, r2, tof = fixture(7)
        assert gpu.solve(r1, r2, tof, C.MU_SUN_KM3_S2, scan_samples=127).feasible.all()
    with pytest.raises(RuntimeError, match="closed"):
        gpu.solve(r1, r2, tof, C.MU_SUN_KM3_S2)


def test_screening_cli_options_and_worker_guard():
    from argparse import Namespace

    from spacepdhcg.cli import build_parser
    from spacepdhcg.gtoc12.cli import _with_screening_backend

    parsed = build_parser().parse_args(
        ["gtoc12", "run", "--run-id", "test", "--output", "unused", "--screening-backend", "cuda"]
    )
    assert parsed.screening_backend == "cuda"

    @_with_screening_backend
    def body(args):
        raise AssertionError("must reject configuration before running")

    with pytest.raises(ValueError, match="workers 1"):
        body(Namespace(screening_backend="cuda", workers=2))
    with pytest.raises(ValueError, match="backend"):
        with using_lambert_backend("bogus"):
            pass
