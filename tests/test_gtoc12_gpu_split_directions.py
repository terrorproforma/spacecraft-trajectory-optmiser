"""Parallel short/long scans preserve every observable hop result field."""
import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.gpu_lambert import HOP_RESULT, GpuLambert

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA"
)


@pytest.mark.parametrize("count", [31, 257, 1024, 1025, 16385])
@pytest.mark.parametrize("samples", [16, 31, 32, 33, 256, 1024])
@pytest.mark.parametrize("fast_root", [0, 1])
def test_split_directions_match_every_field(monkeypatch, count, samples, fast_root):
    rng = np.random.default_rng(601)
    r1, r2 = (rng.normal(size=(count, 3)) * C.AU_KM for _ in range(2))
    days = rng.uniform(50, 1200, count)
    days[:2] = [0, -1]
    r2[2] = r1[2]
    velocity = rng.normal(size=(count, 3)) * 20
    velocity[3, 0] = np.nan
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_FAST_LAMBERT_ROOT", str(fast_root))
    outputs = []
    with GpuLambert(count) as gpu:
        gpu.screen_hops(
            r1, velocity, r2, [0, 0, 0], 64328, days,
            departure_allowance_km_s=6, scan_samples=samples,
        )
        for enabled in (0, 1):
            monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_PARALLEL_DIRECTIONS", str(enabled))
            output = np.empty(count, dtype=HOP_RESULT)
            gpu._check(gpu.evaluate_hops(
                gpu.handle, gpu.hops.ctypes.data, count, output.ctypes.data, count
            ))
            outputs.append(output)
    for name in HOP_RESULT.names:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name], err_msg=name)
    assert outputs[1]["feasible"].any()
