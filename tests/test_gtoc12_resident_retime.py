"""Real ephemeris grids must stay resident without changing selected schedules."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.retiming import Retimer, RetimeSettings, build_visits, orders_of
from spacepdhcg.gtoc12.search import RoutePlan, SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def mission():
    source = (
        Path(__file__).resolve().parents[1]
        / "results/lambda/2026-09-08/gpu-retime-dp-v220/h100/mission-v223/plan.json"
    )
    return RoutePlan.from_summary(json.loads(source.read_text()))


@pytest.mark.parametrize("step", [15.0, 30.0])
def test_resident_schedule_matches_cpu_reference_without_host_tables(monkeypatch, step):
    plan, cat = mission(), load_catalogue()
    settings = RetimeSettings(step_days=step)
    with lambert.using_lambert_backend("cuda") as gpu:
        timer = Retimer(cat, SearchSettings(), settings)
        native = timer.retime(plan)
        assert native.plan is not None
        assert not timer._tables
        assert gpu.telemetry["retime_table_uploads"] == 0
        assert gpu.telemetry["retime_resident_builds"] == 1
        assert native.lambert_evaluations == 2 * gpu.telemetry["retime_resident_cells"]
        visits = build_visits(*orders_of(plan))
        masses = timer._plan_masses(plan)
        selected = timer._dp(visits, masses, 0.15)
        assert selected is not None
        a, d, value = selected
        arrivals, departures = timer.lattice.epochs[a].tolist(), timer.lattice.epochs[d].tolist()
        path = lambert.cuda_retime_path_values(timer, visits, arrivals, departures).copy()
        assert (
            lambert.cuda_retime_path_values(timer, visits, arrivals, [x + 1 for x in departures])
            is None
        )
        assert lambert.cuda_retime_path_values(Retimer(cat), visits, arrivals, departures) is None
        # Force independent host-table and CPU-DP evaluation, including forward
        # bookkeeping. The CUDA ephemeris/Lambert operator stays identical.
        monkeypatch.setattr(lambert, "cuda_retime_dp", lambda *args: NotImplemented)
        monkeypatch.setattr(lambert, "cuda_retime_path_values", lambda *args: None)
        reference = Retimer(cat, SearchSettings(), settings)
        cpu = reference.retime(plan)
        assert cpu.plan is not None
        assert native.plan.summary() == cpu.plan.summary()
        result = reference._dp(visits, masses, 0.15)
        assert result[:2] == (a, d)
        assert abs(result[2] - value) < 1e-9
        for j, visit in enumerate(visits[:-1]):
            table, _ = reference.leg_table(visit.body, visits[j + 1].body, visit.role_out)
            tofs = reference._tofs(visit.role_out)
            k = round((arrivals[j + 1] - departures[j] - tofs[0]) / step)
            assert path[j] == table[d[j], k]


def test_resident_cache_rebuilds_after_release_and_changed_floor():
    plan, cat = mission(), load_catalogue()
    with lambert.using_lambert_backend("cuda") as gpu:
        timer = Retimer(cat)
        visits = build_visits(*orders_of(plan))
        masses = timer._plan_masses(plan)
        first = timer._dp(visits, masses, 0.15)
        assert first is not None
        assert timer._dp(visits, masses, 0.3) is not None
        assert gpu.telemetry["retime_resident_builds"] == 1
        timer.release_caches()
        a, d, _ = first
        assert (
            lambert.cuda_retime_path_values(
                timer, visits, timer.lattice.epochs[a].tolist(), timer.lattice.epochs[d].tolist()
            )
            is None
        )
        assert timer._dp(visits, masses, 0.15) == first
        assert gpu.telemetry["retime_resident_builds"] == 2
        timer.protect_earth_leg(plan)
        assert timer._dp(visits, masses, 0.15) is not None
        assert gpu.telemetry["retime_resident_builds"] == 3
        # A custom table must switch back to the established immutable-table path.
        pair = visits[0].body, visits[1].body, visits[0].role_out
        shape = timer.lattice.count, len(timer._tofs(pair[2]))
        timer._tables[pair] = (np.full(shape, np.inf), np.zeros(shape, dtype=bool))
        assert timer._dp(visits, masses, 0.15) is None
        assert gpu.telemetry["retime_table_uploads"] == 1


def test_resident_invalid_pinned_visit_does_not_return_previous_path():
    plan, cat = mission(), load_catalogue()
    with lambert.using_lambert_backend("cuda"):
        timer = Retimer(cat)
        visits = build_visits(*orders_of(plan))
        masses = timer._plan_masses(plan)
        a, d, _ = timer._dp(visits, masses, 0.15)
        bad = [*visits]
        bad[1] = replace(bad[1], pinned_arrival=timer.lattice.epochs[0] + 0.5)
        assert timer._dp(bad, masses, 0.15) is None
        assert (
            lambert.cuda_retime_path_values(
                timer, visits, timer.lattice.epochs[a].tolist(), timer.lattice.epochs[d].tolist()
            )
            is None
        )
