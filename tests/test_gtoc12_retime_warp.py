"""Flight-time reduction keeps serial tie order and prefix-stop semantics."""

import ctypes as ct
import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.gpu_retime import STAGE, GpuRetime

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize(
    "candidates,cutoff,nan_index,winner",
    [
        ([31, 32], None, None, 31),
        ([1, 33], None, None, 1),
        ([32, 64], None, None, 32),
        ([31, 64], 32, None, 31),
        ([64], 32, None, None),
        ([31, 32], None, 31, 32),
    ],
)
def test_warp_ties_multiple_tiles_and_unsorted_prefix(candidates, cutoff, nan_index, winner):
    n, nt = 83, 65  # Partial final block and more than two warp-sized TOF tiles.
    epochs = np.arange(n, dtype=np.float64)
    tofs = np.arange(1, nt + 1, dtype=np.float64)
    shifts = np.arange(1, nt + 1, dtype=np.int32)
    if cutoff is not None:
        shifts[cutoff] = n  # Old serial loop must ignore all later entries.
    dv, feasible = np.zeros((n, nt)), np.zeros((n, nt), np.uint8)
    swept, sweep_ok = np.full((n, nt), np.nan), np.ones((n, nt), np.uint8)
    for k in candidates:
        departure = n - 1 - shifts[k]
        if departure >= 0:
            feasible[departure, k] = 1
            if k == nan_index:
                dv[departure, k] = np.nan
                swept[departure, k] = 1.0  # NaN objective must lose, even when measured.
    params = np.zeros(1, STAGE)
    params["tofs"], params["pinned_next"] = nt, n - 1
    params["earliest_collect"], params["mass"] = -np.inf, 1000.0
    params["ratio_limit"], params["flat"] = 1.0, 1.0
    with lambert.using_lambert_backend("cuda") as gpu:
        ws = GpuRetime(gpu.library, gpu.device_id, gpu)
        try:
            ws._check(
                ws.create(
                    gpu.device_id,
                    n,
                    1,
                    n * nt,
                    nt,
                    epochs.ctypes.data,
                    dv.ctypes.data,
                    feasible.ctypes.data,
                    swept.ctypes.data,
                    sweep_ok.ctypes.data,
                    tofs.ctypes.data,
                    shifts.ctypes.data,
                    ct.byref(ws.handle),
                )
            )
            for graph in [False, True, False, True]:
                ws._check(ws.set_graph(ws.handle, int(graph)))
                a, d = np.empty(2, np.int32), np.empty(2, np.int32)
                path, measured, ok = np.empty(1), np.empty(1), np.empty(1, np.uint8)
                objective, accepted = ct.c_double(), ct.c_int32()
                ws._check(
                    ws.evaluate(
                        ws.handle,
                        params.ctypes.data,
                        0.15,
                        0.6,
                        39.2266,
                        a.ctypes.data,
                        d.ctypes.data,
                        ct.byref(objective),
                        ct.byref(accepted),
                        path.ctypes.data,
                        measured.ctypes.data,
                        ok.ctypes.data,
                    )
                )
                assert accepted.value == (winner is not None)
                if winner is not None:
                    assert objective.value == 0.0
                    assert a.tolist() == d.tolist() == [n - 1 - shifts[winner], n - 1]
                    assert path[0] == 0.0 and ok[0] == 1
                else:
                    assert objective.value == -np.inf
        finally:
            ws.close()
