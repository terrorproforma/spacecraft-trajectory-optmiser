"""Resident Lambert options preserve values, selection and ownership."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_options import GpuResidentOptions
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("source,target", [(0, 45738), (45738, 0), (45738, 25792)])
@pytest.mark.parametrize("ordered", [False, True])
def test_resident_generation_selection_and_scratch_lifetime(source, target, ordered):
    cat = load_catalogue()
    settings = SearchSettings()
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        retained = []
        for count in (1, 33, 1025, 7):
            i = np.arange(count)
            departures = 64328.125 + (i % 101) * 11.3
            tofs = 60.375 + (i % 23) * 27.1
            tofs[i % 19 == 0] = 0.0
            departures[i % 37 == 0] = np.nan
            expected = gpu.paired_options(
                cat, source, target, departures, tofs, sort_returns=ordered
            )
            before = gpu.telemetry.get("compact_option_download_bytes", 0)
            table = gpu.paired_options(
                cat, source, target, departures, tofs, sort_returns=ordered, resident=True
            )
            assert isinstance(table, GpuResidentOptions) and len(table) == len(expected)
            assert gpu.telemetry["compact_option_download_bytes"] - before == 4
            retained.append((table, expected))
        # All producer scratch has been overwritten/grown; old tables must remain exact.
        for table, expected in retained:
            for first in (False, True):
                for mass in (600.0, 2000.0, 4000.0):
                    actual = gpu.select_collection(table, mass, 68000.0, settings, first=first)
                    reference = gpu.select_collection(
                        expected, mass, 68000.0, settings, first=first
                    )
                    assert actual == reference
            assert table._host is None
            with pytest.raises(RuntimeError, match="invalid"):
                gpu.select_collection(
                    table, 2000.0, 68000.0, replace(settings, wait_penalty=np.nan)
                )
            assert gpu.select_collection(table, 2000.0, 68000.0, settings) == gpu.select_collection(
                expected, 2000.0, 68000.0, settings
            )
            np.testing.assert_array_equal(table.read(), np.asarray(expected).reshape(-1, 3))
        with ThreadPoolExecutor(max_workers=1) as worker:
            with pytest.raises(RuntimeError):
                worker.submit(retained[-1][0].read).result()
        retained[0][0].close()
        with pytest.raises(RuntimeError, match="closed"):
            gpu.select_collection(retained[0][0], 2000.0, 68000.0, settings)
    assert all(not table.handle.value for table, _ in retained)


def test_return_pruning_uses_resident_selection_without_host_read(monkeypatch):
    cat = load_catalogue()
    settings = SearchSettings()
    search = RouteSearch(cat, [45738, 25792], settings)
    with using_lambert_backend("cuda") as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_RESIDENT_OPTIONS", "0")
        baseline = [search._return_feasible(45738, mass) for mass in (600.0, 2000.0, 4000.0)]
        search._return_cache.clear()
        monkeypatch.delenv("SPACEPDHCG_TEST_GTOC12_RESIDENT_OPTIONS")
        monkeypatch.setattr(GpuResidentOptions, "read", lambda *a: pytest.fail("host table read"))
        monkeypatch.setattr(search, "_feasible", lambda *a: pytest.fail("CPU return pruning"))
        assert [
            search._return_feasible(45738, mass) for mass in (600.0, 2000.0, 4000.0)
        ] == baseline
        assert gpu.telemetry["completed_return_feasibility_queries"] == 3
        assert gpu.telemetry["resident_option_selection_download_bytes"] == 3 * 40
