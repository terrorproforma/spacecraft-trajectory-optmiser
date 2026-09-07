"""Final-epoch reduction preserves first ties across warps and strided tiles."""

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
    "candidates", [[31, 32, 256], [127, 128], [128, 129], [255, 256, 512], [512], []]
)
def test_finish_first_epoch_tie_and_partial_tiles(candidates):
    n = 513
    epochs = np.arange(n, dtype=np.float64)
    tofs, shifts = np.ones(1), np.ones(1, np.int32)
    dv, ok = np.zeros(n), np.zeros(n, np.uint8)
    swept, sweep_ok = np.full(n, np.nan), np.ones(n, np.uint8)
    for candidate in candidates:
        ok[candidate - 1] = 1
    params = np.zeros(1, STAGE)
    params["tofs"], params["pinned_next"] = 1, -1
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
                    n,
                    1,
                    epochs.ctypes.data,
                    dv.ctypes.data,
                    ok.ctypes.data,
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
                path, measured, accepted_path = np.empty(1), np.empty(1), np.empty(1, np.uint8)
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
                        accepted_path.ctypes.data,
                    )
                )
                assert accepted.value == bool(candidates)
                if candidates:
                    winner = min(candidates)
                    assert a.tolist() == d.tolist() == [winner - 1, winner]
                    assert objective.value == 0.0 and path[0] == 0.0 and accepted_path[0] == 1
                else:
                    assert objective.value == -np.inf
        finally:
            ws.close()
