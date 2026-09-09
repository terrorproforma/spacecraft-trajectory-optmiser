"""Resident collection tables avoid downloads and survive cache eviction."""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12 import collectdp
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_collect_tables import GpuCollectTable
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("capacity", [0, 1, 20000])
def test_resident_tour_never_downloads_tables(monkeypatch, capacity):
    source = (
        Path(__file__).resolve().parents[1]
        / "results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json"
    )
    fixture = json.loads(source.read_text())["fixtures"]["6"]
    settings = collectdp.CollectDPSettings(**fixture["settings"])
    catalogue = load_catalogue()
    options = fixture["kwargs"]

    def solve(table):
        return collectdp.plan_collect_tour(
            table,
            fixture["deployed"],
            fixture["camp"],
            fixture["camp_epoch"],
            fixture["mass_after_deploys"],
            **options,
        )

    with using_lambert_backend("cuda") as gpu:
        gpu.collect_tables_resident = False
        expected = solve(collectdp.CollectPairTable(catalogue, settings))
        assert expected is not None
        gpu.collect_tables_resident = True
        table = collectdp.CollectPairTable(
            catalogue, dataclasses.replace(settings, cache_pairs=capacity)
        )
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                pytest.fail("resident DP used a host table or host ephemeris")

            patch.setattr(GpuCollectTable, "read", forbidden)
            patch.setattr(table, "hop", forbidden)
            patch.setattr(table, "earth_return", forbidden)
            patch.setattr(collectdp, "asteroid_state", forbidden)
            patch.setattr(collectdp, "earth_state", forbidden)
            actual = solve(table)
            repeated = solve(table)
        assert actual is not None and repeated is not None
        assert actual.order == expected.order == repeated.order
        assert actual.collect_epochs == expected.collect_epochs == repeated.collect_epochs
        assert actual.return_departure == expected.return_departure == repeated.return_departure
        assert actual.return_tof == expected.return_tof == repeated.return_tof
        np.testing.assert_allclose(actual.hops, expected.hops, rtol=0, atol=0)
        np.testing.assert_allclose(
            actual.hop_propellant_kg, expected.hop_propellant_kg, rtol=0, atol=1e-9
        )
        assert actual.objective_kg == pytest.approx(expected.objective_kg, abs=1e-9)
        assert gpu.telemetry.get("collect_table_download_bytes", 0) == 0
        assert gpu.telemetry.get("collect_dp_rebinds", 0) >= 2
        assert gpu.telemetry["collect_dp_allocations"] == 1
        assert len(gpu.collect_table_cache) <= capacity
        table.release_caches()
        assert not gpu.collect_table_cache


def test_legacy_result_buffer_stays_496_bytes(monkeypatch):
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "0")
    import ctypes as ct

    from spacepdhcg.gtoc12.gpu_collect_dp import GpuCollectDP

    original = GpuCollectDP.solve
    calls = []

    def checked(self, masses, camp_mass, price, *args):
        calls.append(True)
        values = np.ascontiguousarray(masses, dtype=np.float64)
        buffer = (ct.c_ubyte * 528)(*([0xA5] * 528))
        legacy = self.gpu.library.spacepdhcg_collect_solve
        legacy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_double, ct.c_void_p]
        legacy.restype = ct.c_int
        assert legacy(self.handle, values.ctypes.data, camp_mass, price, buffer) == 0
        assert bytes(buffer)[496:] == bytes([0xA5] * 32)
        return original(self, masses, camp_mass, price, *args)

    monkeypatch.setattr(GpuCollectDP, "solve", checked)
    test_resident_tour_never_downloads_tables(monkeypatch, 1)
    assert calls
