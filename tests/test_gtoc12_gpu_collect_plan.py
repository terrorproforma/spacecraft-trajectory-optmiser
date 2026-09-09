"""Native mining and burn-policy selection agree with independent existing passes."""

import ctypes as ct
import dataclasses
import os

import numpy as np
import pytest
from test_gtoc12_collectdp import T0, _FakeTable

from spacepdhcg.gtoc12 import collectdp
from spacepdhcg.gtoc12.gpu_collect_dp import Inputs, PlanInputs, PlanResult, Policy
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def case(k=3, n=41):
    ids = list(range(11, 11 + k))
    rng = np.random.default_rng(813 + k)
    table = _FakeTable(
        ids,
        {(a, b): rng.uniform(0.1, 0.9, n) for a in ids for b in ids if a != b},
        n_t=n,
        tofs=(60.0, 120.0, 180.0),
    )
    return table, [(a, T0 + 17.3 * i) for i, a in enumerate(ids)]


@pytest.mark.parametrize("k", [1, 3, 8, 9])
@pytest.mark.parametrize("burn", [None, 0.0, 37.25, -5.0, np.nan])
def test_native_schedule_preserves_complete_tour_and_diagnostics(monkeypatch, k, burn):
    table, deployed = case(k)
    weights = {a: 0.7 + i * 0.023 for i, (a, _) in enumerate(deployed)}
    args = (table, deployed, deployed[-1][0], table.epochs[3], 1800.0)
    with using_lambert_backend("cuda") as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "0")
        expected = collectdp.plan_collect_tour(*args, weights=weights, burn_per_hop=burn)
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "1")
        before = gpu.telemetry.get("completed_collection_dp_passes", 0)
        actual = collectdp.plan_collect_tour(*args, weights=weights, burn_per_hop=burn)
        assert actual is not None and expected is not None
        assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
        assert gpu.telemetry["completed_collection_dp_passes"] - before in (1, 2)
        assert gpu.telemetry["native_collection_plans"] == 1
        if k == 9 and burn is None:
            assert len(actual.hops) >= 8


def test_heavy_failure_uses_nominal_burn_on_device(monkeypatch):
    table, deployed = case(2)
    mass = 1800.0
    heavy = mass + sum(
        collectdp.C.maximum_collected_mass(table.epochs[-1] - t) for _, t in deployed
    )
    # Separate the heavy and nominal-burn return authority thresholds.
    ratio = table.settings.return_authority_ratio
    a = ratio * collectdp.thrust_authority_km_s(heavy, table.return_tofs[0], 1.0)
    b = ratio * collectdp.thrust_authority_km_s(heavy - 0.06 * mass, table.return_tofs[0], 1.0)
    table.return_dv = float((a + b) / 2)
    args = (table, deployed, deployed[-1][0], table.epochs[0], mass)
    with using_lambert_backend("cuda") as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "0")
        assert collectdp.plan_collect_tour(*args, burn_per_hop=0.0) is None
        expected = collectdp.plan_collect_tour(*args)
        assert expected is not None
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "1")
        before = gpu.telemetry.get("completed_collection_dp_passes", 0)
        actual = collectdp.plan_collect_tour(*args)
        assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
        assert gpu.telemetry["completed_collection_dp_passes"] - before == 2
        assert "pass1_objective_kg" not in actual.diagnostics


def test_native_entry_does_not_prepare_host_mining_or_subsets(monkeypatch):
    table, deployed = case()
    with using_lambert_backend("cuda") as gpu:

        def forbidden(*args, **kwargs):
            pytest.fail("native plan entered the host subset/pass preparation")

        monkeypatch.setattr(collectdp, "_solve_collect_dp", forbidden)
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "1")
        assert collectdp.plan_collect_tour(table, deployed, 11, table.epochs[0], 1800.0) is not None
        assert gpu.collect_dp_workspace.data["mined"].size == 0
        assert gpu.telemetry["collection_plan_result_download_bytes"] == ct.sizeof(PlanResult)


@pytest.mark.parametrize(
    "invalid", ["abi", "epochs", "deploy", "weights", "year", "floor", "null", "reserved"]
)
def test_invalid_plan_update_preserves_previous_problem(monkeypatch, invalid):
    table, deployed = case()
    with using_lambert_backend("cuda") as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "1")
        collectdp.plan_collect_tour(table, deployed, 11, table.epochs[0], 1800.0)
        ws = gpu.collect_dp_workspace
        update = gpu.library.spacepdhcg_collect_update_plan
        update.argtypes = [
            ct.c_void_p,
            ct.POINTER(Policy),
            ct.POINTER(Inputs),
            ct.POINTER(PlanInputs),
        ]
        update.restype = ct.c_int
        solve = gpu.library.spacepdhcg_collect_solve_plan
        solve.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.POINTER(PlanResult),
        ]
        solve.restype = ct.c_int
        expected = PlanResult()
        assert solve(ws.handle, 1800.0, 1.0, np.nan, ct.byref(expected)) == 0
        metadata = PlanInputs.from_buffer_copy(bytes(ws.plan_inputs))
        changed = [a.copy() for a in ws.plan_data]
        metadata.epochs, metadata.deploy_epochs, metadata.weights = [a.ctypes.data for a in changed]
        if invalid == "abi":
            metadata.abi_version = 2
        if invalid == "epochs":
            changed[0][1] = changed[0][0]
        if invalid == "deploy":
            changed[1][0] = np.nan
        if invalid == "weights":
            changed[2][0] = np.inf
        if invalid == "year":
            metadata.year_days = 0.0
        if invalid == "floor":
            metadata.floor_mass = -1.0
        if invalid == "null":
            metadata.weights = None
        if invalid == "reserved":
            metadata.reserved = 1
        packed = Inputs(
            *(None if n == "mined" else ws.data[n].ctypes.data for n, _ in Inputs._fields_)
        )
        assert update(ws.handle, ct.byref(ws.policy), ct.byref(packed), ct.byref(metadata)) == 1
        actual = PlanResult()
        assert solve(ws.handle, 1800.0, 1.0, np.nan, ct.byref(actual)) == 0
        assert bytes(actual) == bytes(expected)
        # Invalid solve leaves the caller's output untouched.
        assert solve(ws.handle, 1800.0, 1.0, np.inf, ct.byref(actual)) == 1
        assert bytes(actual) == bytes(expected)


def test_legacy_update_disables_plan_until_metadata_is_rebound(monkeypatch):
    table, deployed = case()
    with using_lambert_backend("cuda") as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN", "1")
        collectdp.plan_collect_tour(table, deployed, 11, table.epochs[0], 1800.0)
        ws = gpu.collect_dp_workspace
        data = dict(ws.data, mined=np.zeros((len(deployed), len(table.epochs))))
        packed = Inputs(*(data[n].ctypes.data for n, _ in Inputs._fields_))
        legacy = gpu.library.spacepdhcg_collect_update
        legacy.argtypes = [ct.c_void_p, ct.POINTER(Policy), ct.POINTER(Inputs)]
        legacy.restype = ct.c_int
        assert legacy(ws.handle, ct.byref(ws.policy), ct.byref(packed)) == 0
        result = PlanResult()
        solve = gpu.library.spacepdhcg_collect_solve_plan
        assert solve(ws.handle, 1800.0, 1.0, np.nan, ct.byref(result)) == 1
        collectdp.plan_collect_tour(table, deployed, 11, table.epochs[0], 1800.0)
        assert gpu.telemetry["collect_dp_allocations"] == 1
        assert gpu.telemetry["collect_dp_rebinds"] == 1
