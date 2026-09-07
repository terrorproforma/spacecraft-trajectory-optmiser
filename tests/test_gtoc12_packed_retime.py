"""Packed output preserves legacy C APIs, array bounds and detached results."""

import ctypes as ct
import os

import numpy as np
import pytest
from test_gtoc12_gpu_retime import fixture

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.gpu_retime import STAGE
from spacepdhcg.gtoc12.screening import exhaust_velocity_km_s

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("cooperative", [False, True])
def test_packed_output_legacy_api_bounds_and_reuse(monkeypatch, cooperative):
    timer, visits = fixture(monkeypatch, 8, cooperative)
    count = len(visits)
    masses = [2500.0] * (count - 1)
    with lambert.using_lambert_backend("cuda") as gpu:
        expected = timer._dp(visits, masses, 0.15)
        assert expected is not None
        ws = gpu.retime_workspace
        native = ws.evaluate
        captured = []

        def capture(*args):
            captured.append(
                np.frombuffer(ct.string_at(args[1], (count - 1) * STAGE.itemsize), STAGE).copy()
            )
            return native(*args)

        monkeypatch.setattr(ws, "evaluate", capture)
        timer._dp(visits, masses, 0.15)
        params = captured[-1]
        detached = None
        for graph in [False, True]:
            ws._check(ws.set_graph(ws.handle, int(graph)))
            for suffix, optional in [("host", 0), ("path_host", 1), ("swept_path_host", 3)]:
                call = getattr(gpu.library, "spacepdhcg_gtoc12_retime_" + suffix)
                call.argtypes = native.argtypes[:9] + [ct.c_void_p] * optional
                call.restype = ct.c_int
                buffers = [
                    np.full(count + 2, -777, np.int32),
                    np.full(count + 2, -777, np.int32),
                    np.full(count + 1, -777.0),
                    np.full(count + 1, -777.0),
                    np.full(count + 1, 231, np.uint8),
                ]
                outputs = [b[1:-1] for b in buffers]
                objective, feasible = ct.c_double(), ct.c_int32()
                ws._check(
                    call(
                        ws.handle,
                        params.ctypes.data,
                        0.15,
                        0.6,
                        exhaust_velocity_km_s(),
                        outputs[0].ctypes.data,
                        outputs[1].ctypes.data,
                        ct.byref(objective),
                        ct.byref(feasible),
                        *[a.ctypes.data for a in outputs[2 : 2 + optional]],
                    )
                )
                assert feasible.value == 1
                result = (outputs[0].tolist(), outputs[1].tolist(), objective.value)
                assert result == expected
                if detached is None:
                    detached = (result, outputs[0], outputs[0].copy())
                assert result == detached[0]
                np.testing.assert_array_equal(detached[1], detached[2])
                for b in buffers:
                    sentinel = 231 if b.dtype == np.uint8 else -777
                    assert b[0] == sentinel and b[-1] == sentinel
                for a in outputs[2 + optional :]:
                    assert (a == (231 if a.dtype == np.uint8 else -777)).all()
                if optional == 3:
                    assert np.isfinite(outputs[2]).all()
                    assert (outputs[4] == 1).all()
                # Invalid controls must leave every caller buffer untouched.
                saved = [b.copy() for b in buffers]
                assert (
                    call(
                        ws.handle,
                        params.ctypes.data,
                        -1.0,
                        0.6,
                        exhaust_velocity_km_s(),
                        outputs[0].ctypes.data,
                        outputs[1].ctypes.data,
                        ct.byref(objective),
                        ct.byref(feasible),
                        *[a.ctypes.data for a in outputs[2 : 2 + optional]],
                    )
                    == 1
                )
                for actual, before in zip(buffers, saved, strict=True):
                    np.testing.assert_array_equal(actual, before)
