"""CUDA DP parity across pricing, ties and cooperative scheduling constraints."""

import os
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.retiming import Retimer, RetimeSettings, Visit
from spacepdhcg.gtoc12.search import SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def fixture(monkeypatch, seed, cooperative=False, tied=False):
    settings = RetimeSettings(
        step_days=30.0,
        hop_tof_days=(60.0, 240.0),
        earth_tof_days=(360.0, 900.0),
        camp_max_days=180.0,
        long_camp_max_days=720.0,
        return_tof_model=True,
        orphan_credit=0.4,
    )
    retimer = Retimer(None, SearchSettings(), settings, weights={1: 0.7, 2: 1.2})
    if cooperative:
        visits = [
            Visit(0, False, False, "earth_out"),
            Visit(1, True, False, "collect_hop"),
            Visit(2, False, True, "earth_return", foreign_deploy_epoch=65000.0),
            Visit(0, False, False, ""),
        ]
    else:
        visits = [
            Visit(0, False, False, "earth_out"),
            Visit(1, True, False, "deploy_hop"),
            Visit(2, True, True, "collect_hop"),
            Visit(1, False, True, "earth_return"),
            Visit(0, False, False, ""),
        ]
    rng = np.random.default_rng(seed)
    for visit, nxt in pairwise(visits):
        shape = (retimer.lattice.count, len(retimer._tofs(visit.role_out)))
        dv = np.zeros(shape) if tied else rng.uniform(0.2, 4.0, shape)
        feasible = np.ones(shape, dtype=bool) if tied else rng.random(shape) > 0.08
        retimer._tables[(visit.body, nxt.body, visit.role_out)] = dv, feasible
    nret = len(retimer._tofs("earth_return"))
    swept = np.full((retimer.lattice.count, nret), np.nan)
    ok = np.ones_like(swept, dtype=bool)
    if not tied:
        measured = rng.random(swept.shape) < 0.25
        swept[measured] = rng.uniform(0.85, 1.5, measured.sum())
        ok[rng.random(ok.shape) < 0.1] = False
    override = (swept, ok)
    monkeypatch.setattr(retimer, "_return_override", lambda body: override)
    retimer.inflations[(1, 2)] = 1.08
    retimer.inflations[(1, 0)] = 0.97
    return retimer, visits


def compare(monkeypatch, gpu, retimer, visits, masses, price):
    with monkeypatch.context() as reference:
        reference.setattr(lambert, "cuda_retime_dp", lambda *args: NotImplemented)
        expected = retimer._dp(visits, masses, price)
    actual = retimer._dp(visits, masses, price)
    if expected is None:
        assert actual is None
    else:
        assert actual[:2] == expected[:2]
        assert actual[2] == pytest.approx(expected[2], abs=1e-9, rel=0)
    return actual


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("cooperative", [False, True])
def test_weighted_pricing_and_measured_returns_match(monkeypatch, seed, cooperative):
    retimer, visits = fixture(monkeypatch, seed, cooperative)
    masses = [2900.0 - j * 300 for j in range(len(visits) - 1)]
    with lambert.using_lambert_backend("cuda") as gpu:
        for price in [0.0, 0.01, 0.15, 1.0]:
            assert compare(monkeypatch, gpu, retimer, visits, masses, price) is not None
        assert gpu.telemetry["retime_table_uploads"] == 1
        assert gpu.telemetry["completed_retime_dp_calls"] == 4
        assert gpu.telemetry["gpu_used"]
        retimer.ban(1, 2, 0.15)
        compare(monkeypatch, gpu, retimer, visits, [m * 0.95 for m in masses], 0.15)
        assert gpu.telemetry["retime_table_uploads"] == 1


def test_ties_pins_infeasibility_and_table_replacement(monkeypatch):
    retimer, visits = fixture(monkeypatch, 0, tied=True)
    masses = [2500.0] * 4
    with lambert.using_lambert_backend("cuda") as gpu:
        expected = compare(monkeypatch, gpu, retimer, visits, masses, 0.0)
        visits[1] = replace(visits[1], pinned_arrival=retimer.lattice.epochs[20])
        assert compare(monkeypatch, gpu, retimer, visits, masses, 0.0) is not None
        visits[1] = replace(visits[1], pinned_arrival=retimer.lattice.epochs[20] + 1.0)
        assert compare(monkeypatch, gpu, retimer, visits, masses, 0.0) is None
        visits[1] = replace(visits[1], pinned_arrival=None)
        key = (0, 1, "earth_out")
        dv, ok = retimer._tables[key]
        retimer._tables[key] = dv.copy(), np.zeros_like(ok)
        assert compare(monkeypatch, gpu, retimer, visits, masses, 0.0) is None
        retimer._tables[key] = dv, ok
        assert compare(monkeypatch, gpu, retimer, visits, masses, 0.0) == expected
        assert gpu.telemetry["retime_table_uploads"] == 3


def test_flat_models_and_impossible_camp(monkeypatch):
    retimer, visits = fixture(monkeypatch, 7)
    retimer.settings = replace(retimer.settings, hop_inflation_slope=None, return_tof_model=False)
    with lambert.using_lambert_backend("cuda") as gpu:
        compare(monkeypatch, gpu, retimer, visits, [2500.0] * 4, 0.1)
        retimer.settings = replace(retimer.settings, long_camp_max_days=60.0)
        assert compare(monkeypatch, gpu, retimer, visits, [2500.0] * 4, 0.1) is None
