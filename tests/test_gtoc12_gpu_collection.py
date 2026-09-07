"""CUDA collection pricing retains the scalar search's decisions."""

import ctypes as ct
import os
from dataclasses import replace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_collection import Query, Result
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.screening import thrust_authority_km_s
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("slope", [None, 0.0, 0.65])
def test_collection_selection_matches_reference(monkeypatch, slope):
    rng = np.random.default_rng(196)
    settings = SearchSettings(hop_inflation_slope=slope)
    search = RouteSearch(load_catalogue(), [1, 2], settings)
    epoch = C.MISSION_END_MJD - 100
    options = [
        tuple(x)
        for x in np.column_stack(
            (
                rng.uniform(0, 12, 3073),
                epoch - rng.uniform(100, 1000, 3073),
                rng.choice([90.0, 180.0, 360.0, 720.0], 3073),
            )
        )
    ]
    monkeypatch.setattr(search, "_collect_hop_options", lambda *a: options)
    cases = [(600.0, 1.0, np.inf), (2000.0, 4.0, 350.0), (4000.0, 16.0, 100.0), (2000.0, 1.0, -1.0)]
    expected = [search._best_collect_hop(1, 2, epoch, *case) for case in cases]
    with using_lambert_backend("cuda") as gpu:
        monkeypatch.setattr(search, "_feasible", lambda *a: pytest.fail("scalar feasibility used"))
        monkeypatch.setattr(search, "_propellant", lambda *a: pytest.fail("scalar pricing used"))
        for case, (cost, hop) in zip(cases, expected, strict=True):
            actual_cost, actual_hop = search._best_collect_hop(1, 2, epoch, *case)
            assert actual_hop == hop
            assert actual_cost == pytest.approx(cost, abs=1e-10, rel=1e-13)
        assert gpu.telemetry["completed_collection_options"] == len(options) * len(cases)


def test_collection_ties_boundaries_empty_and_invalid_recovery():
    settings = SearchSettings(wait_penalty=0.0)
    epoch = 68000.0
    mass = 2000.0
    authority = float(thrust_authority_km_s(mass, 180.0, 1.0))
    limit = settings.earth_return_authority_ratio * authority
    with using_lambert_backend("cuda") as gpu:
        rows = [(0.0, 66000.0, 180.0), (0.0, 66500.0, 180.0), (0.0, 66500.0, 360.0)]
        cost, hop = gpu.select_collection(rows, mass, epoch, settings)
        assert cost == 0 and hop == rows[1]  # later departure, then first identical tie
        rows = [(np.nextafter(limit, np.inf), 66000.0, 180.0), (limit, 66100.0, 180.0)]
        assert gpu.select_collection(rows, mass, epoch, settings, first=True)[1] == rows[1]
        assert gpu.select_collection([], mass, epoch, settings) == (np.inf, None)
        for bad_rows, bad_mass in [([(np.nan, 66000.0, 180.0)], mass), (rows, 0.0)]:
            with pytest.raises(RuntimeError, match="invalid"):
                gpu.select_collection(bad_rows, bad_mass, epoch, settings)
        assert gpu.select_collection(rows, mass, epoch, settings, first=True)[1] == rows[1]
        assert gpu.select_collection(
            [(0.0, 67000.0, 180.0)], mass, epoch, settings, max_span=999.0
        ) == (np.inf, None)
        with pytest.raises(RuntimeError, match="invalid"):
            gpu.select_collection(rows, mass, epoch, replace(settings, wait_penalty=np.nan))
    assert ct.sizeof(Query) == 120 and ct.sizeof(Result) == 16
