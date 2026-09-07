"""Resident return masks must match the complete CPU grid, including rejected cells."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_retime import SWEEP_CELL
from spacepdhcg.gtoc12.retiming import Retimer, build_visits, orders_of
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12.search import RoutePlan

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def fixture():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/lambda/2026-09-08/gpu-retime-dp-v220/h100/mission-v223/plan.json"
    plan = RoutePlan.from_summary(json.loads(path.read_text()))
    timer = Retimer(load_catalogue())
    visits = build_visits(*orders_of(plan))
    body = visits[-2].body
    k = timer.lattice.index(plan.legs[-1].departure_epoch)
    # Unsorted inputs expose nearest-neighbour tie order; off-grid samples and
    # unattempted cells must not participate. These are synthetic measurements.
    departures = [*timer.lattice.epochs[[k + 2, k, k + 4]], timer.lattice.epochs[k] + 0.5]
    tofs = np.array([450.0, 420.0, 480.0, 435.5])
    attempted = np.ones((4, 4), dtype=bool)
    attempted[2, 2] = False
    certified = attempted.copy()
    certified[1, 0] = False
    measured = np.arange(16, dtype=float).reshape(4, 4) / 10 + 5.0
    measured[1, 1] = np.nan
    return (
        plan,
        timer,
        visits,
        ReturnSweep(
            body,
            1400.0,
            np.array(departures),
            tofs,
            attempted,
            certified,
            measured,
            np.full((4, 4), np.inf),
        ),
    )


def exported(gpu, timer, visits):
    nt = len(timer._tofs("earth_return"))
    offset = sum(timer.lattice.count * len(timer._tofs(v.role_out)) for v in visits[:-2])
    inflation = np.empty((timer.lattice.count, nt), dtype=np.float64)
    ok = np.empty(inflation.shape, dtype=np.uint8)
    ws = gpu.retime_workspace
    ws._check(ws.read_sweep(ws.handle, offset, nt, inflation.ctypes.data, ok.ctypes.data))
    return inflation, ok.astype(bool), offset


@pytest.mark.parametrize("policy", ["mixed", "all_refused", "unattempted"])
def test_resident_return_grid_and_schedule_match_cpu(monkeypatch, policy):
    plan, timer, visits, sweep = fixture()
    if policy == "all_refused":
        sweep.certified[:] = False
    if policy == "unattempted":
        sweep.attempted[:] = False
    timer.set_return_sweep(sweep)
    masses = timer._plan_masses(plan)
    with lambert.using_lambert_backend("cuda") as gpu:
        result = timer._dp(visits, masses, 0.15)
        inflation, ok, _ = exported(gpu, timer, visits)
        assert not timer._tables and not timer._return_tables
        native_forward = None
        if result is not None:
            a, d, _ = result
            arrivals, departures = (
                timer.lattice.epochs[a].tolist(),
                timer.lattice.epochs[d].tolist(),
            )
            native_forward = timer._forward(visits, arrivals, departures)
            assert not timer._tables and not timer._return_tables
            assert (
                lambert.cuda_retime_path_values(
                    timer, visits, arrivals, [x + 1 for x in departures], True
                )
                is None
            )
        reference = Retimer(timer.catalogue)
        reference.set_return_sweep(copy.deepcopy(sweep))
        expected = reference._return_override(sweep.asteroid)
        if expected is None:
            assert np.isnan(inflation).all() and ok.all()
        else:
            np.testing.assert_array_equal(ok, expected[1])
            np.testing.assert_allclose(inflation, expected[0], rtol=1e-14, atol=0, equal_nan=True)
        monkeypatch.setattr(lambert, "cuda_retime_dp", lambda *args: NotImplemented)
        cpu = reference._dp(visits, masses, 0.15)
        assert (cpu is None) == (result is None)
        if cpu is not None:
            assert cpu[:2] == result[:2]
            assert abs(cpu[2] - result[2]) < 1e-9
            forward = reference._forward(visits, arrivals, departures)
            assert forward[2] == native_forward[2]
            np.testing.assert_allclose(forward[1], native_forward[1], rtol=1e-14, atol=1e-12)
            if forward[0] is not None:
                assert (
                    abs(forward[0].total_collected_kg - native_forward[0].total_collected_kg) < 1e-9
                )
        assert gpu.telemetry["retime_resident_builds"] == 1
        assert gpu.telemetry["retime_table_uploads"] == 0


def test_refusal_and_empty_sweep_update_without_rebuilding_transfers():
    plan, timer, visits, sweep = fixture()
    masses = timer._plan_masses(plan)
    timer.set_return_sweep(sweep)
    with lambert.using_lambert_backend("cuda") as gpu:
        result = timer._dp(visits, masses, 0.15)
        _, mask, offset = exported(gpu, timer, visits)
        assert timer.refuse_return(sweep.asteroid, sweep.departures[0], sweep.tofs[0])
        if result is not None:
            a, d, _ = result
            assert (
                lambert.cuda_retime_path_values(
                    timer,
                    visits,
                    timer.lattice.epochs[a].tolist(),
                    timer.lattice.epochs[d].tolist(),
                    True,
                )
                is None
            )
        timer._dp(visits, masses, 0.15)
        after, revised, _ = exported(gpu, timer, visits)
        assert not np.array_equal(mask, revised)
        assert gpu.telemetry["retime_resident_builds"] == 1
        assert gpu.telemetry["retime_sweep_updates"] == 2
        # Native validation must reject malformed sample coordinates atomically.
        bad = np.array([(-1, 0, 1.0, 1, 0)], dtype=SWEEP_CELL)
        ws = gpu.retime_workspace
        nt = len(timer._tofs("earth_return"))
        assert ws.set_sweep(ws.handle, offset, nt, 1, bad.ctypes.data, 2) == 1
        same, same_ok, _ = exported(gpu, timer, visits)
        np.testing.assert_array_equal(same, after)
        np.testing.assert_array_equal(same_ok, revised)
        sweep.attempted[:] = False
        timer.set_return_sweep(sweep)
        timer._dp(visits, masses, 0.15)
        clear, clear_ok, _ = exported(gpu, timer, visits)
        assert np.isnan(clear).all() and clear_ok.all()
        assert gpu.telemetry["retime_resident_builds"] == 1
        assert gpu.telemetry["retime_sweep_updates"] == 3
