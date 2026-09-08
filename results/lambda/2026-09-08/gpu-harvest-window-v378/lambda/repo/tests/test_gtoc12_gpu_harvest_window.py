"""Resident harvest ranking matches host pricing of the same float32 costs."""

import dataclasses
import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import collectdp, constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_collect_tables import GpuCollectTable
from spacepdhcg.gtoc12.hopcalib import InflationFit
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("model", ["flat", "ratio", "fit"])
@pytest.mark.parametrize("maximum_tof", [0.0, 60.0 - 2e-9, 60.0, 180.0])
def test_harvest_minimum_without_table_download(monkeypatch, model, maximum_tof):
    settings = collectdp.CollectDPSettings()
    if model == "ratio":
        settings = dataclasses.replace(settings, hop_inflation_slope=0.65)
    elif model == "fit":
        settings = dataclasses.replace(
            settings, inflation_fit=InflationFit((1.1, 0.4, -0.03, 0.02, 0.05), 0.65, 0.85)
        )
    table = collectdp.CollectPairTable(load_catalogue(), settings)
    a, b = 57530, 53410
    window = (C.MISSION_END_MJD - 1300, C.MISSION_END_MJD - 400)
    lo, hi = map(table.index_at_or_after, window)
    keep = table.tofs <= maximum_tof + 1e-9
    with using_lambert_backend("cuda") as gpu:
        arrays = {
            pair: table._resident_table(*pair, table.tofs, table.epochs[-1]).read()
            for pair in [(a, b), (b, a)]
        }
        downloaded = gpu.telemetry["collect_table_download_bytes"]
        monkeypatch.setattr(
            GpuCollectTable, "read", lambda *args: pytest.fail("downloaded a full table")
        )
        # Changed masses and repeated queries must use current inputs and retained scratch.
        for mass in [1800.0, 900.0, 100000.0, 1800.0]:
            expected = float("inf")
            if np.any(keep):
                for pair, values in arrays.items():
                    costs = table.hop_propellant(
                        values[lo:hi][:, keep].astype(np.float64),
                        mass,
                        table.tofs[keep],
                        pair=pair,
                        epochs=table.epochs[lo:hi],
                    )
                    expected = min(expected, float(np.min(costs)))
            actual = table.harvest_window_cost(a, b, mass, window=window, max_tof_days=maximum_tof)
            assert actual == pytest.approx(expected, rel=0, abs=1e-9)
        assert gpu.telemetry["collect_table_download_bytes"] == downloaded
        assert gpu.telemetry.get("collect_window_download_bytes", 0) == (64 if np.any(keep) else 0)
        assert table.harvest_window_cost(
            a, b, 1800.0, window=(window[0], window[0]), max_tof_days=180
        ) == float("inf")
