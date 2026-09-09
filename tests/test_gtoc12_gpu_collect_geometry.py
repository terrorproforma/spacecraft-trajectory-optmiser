"""Native pair phase grids retain the reference model and workspace contracts."""

import ctypes as ct
import dataclasses
import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import collectdp
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_collect_dp import GeometryInputs, Inputs, PlanInputs, PlanResult, Policy
from spacepdhcg.gtoc12.harvestphase import HarvestPhasePrior
from spacepdhcg.gtoc12.hopcalib import InflationFit
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def case(monkeypatch, fit=True, phase=True):
    cat = load_catalogue()
    ids = [int(a) for a in cat.ids[[2, 117, 1031, 2998]]]
    settings = collectdp.CollectDPSettings(
        inflation_fit=InflationFit((1.1, 0.2, 0.1, 0.2, 0.3), 0.5) if fit else None,
        harvest_phase=HarvestPhasePrior(4.8, 2.7, 0.13, 0.17, 0.1) if phase else None,
        phase_weight=0.7,
    )
    table = collectdp.CollectPairTable(cat, settings)
    table.epochs = table.epochs[:71]
    monkeypatch.setattr(table, "hop", lambda *_: np.full((71, len(table.tofs)), 0.2))
    monkeypatch.setattr(
        table, "earth_return", lambda *_: np.full((71, len(table.return_tofs)), 0.3)
    )
    deployed = [(a, table.epochs[0] + i * 7.3) for i, a in enumerate(ids)]
    return table, deployed


def read_geometry(ws):
    k, n = ws.policy.k, ws.policy.n
    arrays = [np.empty((k, k)), np.empty((k, k, n)), np.empty((k, k, n))]
    read = ws.gpu.library.spacepdhcg_collect_read_geometry
    read.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32, *([ct.c_void_p] * 3)]
    read.restype = ct.c_int
    assert read(ws.handle, k, n, *(a.ctypes.data for a in arrays)) == 0
    return arrays


@pytest.mark.parametrize("fit,phase", [(False, False), (True, False), (False, True), (True, True)])
def test_native_grids_and_tours_match_reference(monkeypatch, fit, phase):
    table, deployed = case(monkeypatch, fit, phase)
    banned = {(deployed[0][0], deployed[1][0])}
    with using_lambert_backend("cuda") as gpu:
        gpu.collect_tables_resident = False
        for offset in (0, 3, 12):
            args = (table, deployed, deployed[-1][0], table.epochs[offset], 1800.0)
            monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_GEOMETRY", "0")
            expected = collectdp.plan_collect_tour(*args, banned_pairs=banned)
            reference = read_geometry(gpu.collect_dp_workspace)
            monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_GEOMETRY", "1")
            original = table.pair_geometry
            with monkeypatch.context() as patch:

                def only_diagnostic(source, target, epochs, original=original):
                    assert len(epochs) == 1, "native plan constructed a host phase grid"
                    return original(source, target, epochs)

                patch.setattr(table, "pair_geometry", only_diagnostic)
                actual = collectdp.plan_collect_tour(*args, banned_pairs=banned)
            assert actual is not None and expected is not None
            for a, b in zip(read_geometry(gpu.collect_dp_workspace), reference, strict=True):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            assert actual.order == expected.order
            assert actual.collect_epochs == expected.collect_epochs
            assert actual.hops == expected.hops
            assert actual.return_departure == expected.return_departure
            assert actual.return_tof == expected.return_tof
            assert actual.objective_kg == pytest.approx(expected.objective_kg, rel=0, abs=1e-10)
            np.testing.assert_allclose(
                actual.hop_propellant_kg, expected.hop_propellant_kg, rtol=0, atol=1e-10
            )
            assert all(
                gpu.collect_dp_workspace.data[n].size == 0
                for n in ("geometry_a", "geometry_l", "penalty")
            )
        assert gpu.telemetry["collection_geometry_plans"] == 3
        assert gpu.telemetry["collection_geometry_upload_bytes"] == 3 * 4 * 5 * 8


@pytest.mark.parametrize("invalid", ["abi", "reserved", "null", "a", "nan", "mu", "weight"])
def test_invalid_geometry_update_leaves_plan_and_output_unchanged(monkeypatch, invalid):
    table, deployed = case(monkeypatch)
    with using_lambert_backend("cuda") as gpu:
        gpu.collect_tables_resident = False
        collectdp.plan_collect_tour(table, deployed, deployed[-1][0], table.epochs[0], 1800.0)
        ws = gpu.collect_dp_workspace
        expected = PlanResult()
        assert (
            gpu.library.spacepdhcg_collect_solve_plan(
                ws.handle, 1800.0, 1.0, np.nan, ct.byref(expected)
            )
            == 0
        )
        g = GeometryInputs.from_buffer_copy(bytes(ws.geometry_inputs))
        changed = ws.geometry_data.copy()
        g.elements = changed.ctypes.data
        if invalid == "abi":
            g.abi_version = 2
        if invalid == "reserved":
            g.reserved = 1
        if invalid == "null":
            g.elements = None
        if invalid == "a":
            changed[0, 1] = 0.0
        if invalid == "nan":
            changed[0, 4] = np.nan
        if invalid == "mu":
            g.mu = -1.0
        if invalid == "weight":
            g.phase_weight = -1.0
        packed = Inputs(
            *(
                None
                if n in {"mined", "geometry_a", "geometry_l", "penalty"}
                else ws.data[n].ctypes.data
                for n, _ in Inputs._fields_
            )
        )
        update = gpu.library.spacepdhcg_collect_update_plan_geometry
        update.argtypes = [
            ct.c_void_p,
            ct.POINTER(Policy),
            ct.POINTER(Inputs),
            ct.POINTER(PlanInputs),
            ct.POINTER(GeometryInputs),
        ]
        update.restype = ct.c_int
        assert (
            update(
                ws.handle,
                ct.byref(ws.policy),
                ct.byref(packed),
                ct.byref(ws.plan_inputs),
                ct.byref(g),
            )
            == 1
        )
        actual = PlanResult()
        assert (
            gpu.library.spacepdhcg_collect_solve_plan(
                ws.handle, 1800.0, 1.0, np.nan, ct.byref(actual)
            )
            == 0
        )
        assert bytes(actual) == bytes(expected)


def test_custom_phase_policy_keeps_its_semantics(monkeypatch):
    table, deployed = case(monkeypatch)

    class CustomPrior:
        def penalty_kg(self, phase):
            return np.full_like(phase, 2.73)

    table.settings = dataclasses.replace(table.settings, harvest_phase=CustomPrior())
    with using_lambert_backend("cuda") as gpu:
        gpu.collect_tables_resident = False
        result = collectdp.plan_collect_tour(
            table, deployed, deployed[-1][0], table.epochs[0], 1800.0
        )
        assert result is not None
        assert gpu.collect_dp_workspace.geometry_inputs is None
        assert gpu.telemetry.get("collection_geometry_plans", 0) == 0
