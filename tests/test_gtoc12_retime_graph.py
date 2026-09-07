"""Graph replay must read fresh policies, controls and table workspaces."""

import ctypes as ct
import os
from dataclasses import replace

import numpy as np
import pytest
from test_gtoc12_gpu_retime import compare, fixture

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.gpu_retime import STAGE

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def stats(ws):
    builds, launches = ct.c_uint64(), ct.c_uint64()
    ws._check(ws.graph_stats(ws.handle, ct.byref(builds), ct.byref(launches)))
    return builds.value, launches.value


@pytest.mark.parametrize("cooperative", [False, True])
def test_graph_and_ordinary_launches_follow_changed_policies(monkeypatch, cooperative):
    timer, visits = fixture(monkeypatch, 3, cooperative)
    masses = [2900.0 - j * 300 for j in range(len(visits) - 1)]
    with lambert.using_lambert_backend("cuda") as gpu:
        for iteration in range(4):
            if iteration == 1:
                timer.ban(1, 2, 0.15)
                timer.weights[1] = 0.4
                timer.inflations[(1, 0)] = 1.3
                masses = [m * 0.93 for m in masses]
            if iteration == 2:
                timer.settings = replace(
                    timer.settings, hop_inflation_slope=None, return_tof_model=False
                )
                visits[1] = replace(visits[1], pinned_arrival=timer.lattice.epochs[25])
            if iteration == 3:
                visits[1] = replace(visits[1], pinned_arrival=None)
            results = []
            for enabled in [False, True, False, True]:
                gpu.retime_cuda_graph = enabled
                results.append(compare(monkeypatch, gpu, timer, visits, masses, iteration * 0.15))
            assert all(result == results[0] for result in results)
            assert stats(gpu.retime_workspace) == (1, (iteration + 1) * 2)
        assert gpu.retime_workspace.set_graph(gpu.retime_workspace.handle, 2) == 1
        assert stats(gpu.retime_workspace) == (1, 8)
        # A replacement snapshot owns new addresses and must rebuild its graph.
        key = (0, 1, "earth_out")
        dv, ok = timer._tables[key]
        timer._tables[key] = dv.copy(), np.zeros_like(ok)
        assert compare(monkeypatch, gpu, timer, visits, masses, 0.15) is None
        assert stats(gpu.retime_workspace) == (1, 1)
        timer._tables[key] = dv, ok
        assert compare(monkeypatch, gpu, timer, visits, masses, 0.15) is not None
        assert stats(gpu.retime_workspace) == (1, 1)


def test_graph_controls_and_selected_path_match_ordinary_launches(monkeypatch):
    timer, visits = fixture(monkeypatch, 5)
    with lambert.using_lambert_backend("cuda") as gpu:
        timer._dp(visits, [2500.0] * 4, 0.15)
        ws = gpu.retime_workspace
        native = ws.evaluate
        captured = []

        def capture(*args):
            captured.append(np.frombuffer(ct.string_at(args[1], 4 * STAGE.itemsize), STAGE).copy())
            return native(*args)

        monkeypatch.setattr(ws, "evaluate", capture)
        timer._dp(visits, [2500.0] * 4, 0.15)
        params = captured[-1]
        for price, thrust, exhaust in [(0.0, 0.6, 40.0), (0.3, 0.4, 25.0), (1.0, 0.8, 50.0)]:
            results = []
            for enabled in [False, True]:
                ws._check(ws.set_graph(ws.handle, int(enabled)))
                a, d = np.empty(5, np.int32), np.empty(5, np.int32)
                dv, swept, ok = np.empty(4), np.empty(4), np.empty(4, np.uint8)
                objective, feasible = ct.c_double(), ct.c_int32()
                ws._check(
                    native(
                        ws.handle,
                        params.ctypes.data,
                        price,
                        thrust,
                        exhaust,
                        a.ctypes.data,
                        d.ctypes.data,
                        ct.byref(objective),
                        ct.byref(feasible),
                        dv.ctypes.data,
                        swept.ctypes.data,
                        ok.ctypes.data,
                    )
                )
                results.append((a, d, dv, swept, ok, objective.value, feasible.value))
            for actual, expected in zip(results[0], results[1], strict=True):
                np.testing.assert_array_equal(actual, expected)
        assert stats(ws) == (1, 5)
