"""Completion packing and native parity; proxy acceptance is not certification."""

from __future__ import annotations

import dataclasses
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_gtoc12_completion_costs import fixture

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.gpu_completion import (
    CANDIDATE,
    DEPLOY,
    ENVIRONMENT,
    LEG,
    LEG_RESULT,
    POLICY,
    RESULT,
    STATS,
    GpuCompletion,
    finish_many,
    pack_inputs,
)
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import PlannedLeg

requires_gpu = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires explicitly enabled, serialized CUDA tests",
)


def request(search, partial, tour, use_table=True):
    legs = search._tour_forward_legs(partial, tour)
    assert legs is not None
    return partial, dict(partial.deployed), dict(tour.collect_epochs), legs, use_table


def test_completion_abi_and_packed_model_provenance():
    assert [a.itemsize for a in (POLICY, CANDIDATE, DEPLOY, LEG, RESULT, LEG_RESULT, STATS)] == [
        80,
        24,
        24,
        128,
        48,
        72,
        48,
    ]
    search, partial, tour = fixture()
    inputs = pack_inputs(search, [request(search, partial, tour)])
    policy, candidates, deploys, legs = inputs
    assert policy["thrust"][0] == C.THRUST_MAX_N
    assert candidates["partial_mass"][0] == partial.mass
    assert list(deploys["deploy_epoch"]) == [epoch for _, epoch in partial.deployed]
    hop = legs[legs["role"] == 3][0]
    assert hop["model"] == 2
    assert list(hop["fit"]) == list(search.collect_table.settings.inflation_fit.coefficients)
    assert hop["delta_a_au"] == 0.02 and hop["delta_longitude_rad"] == 0.3
    assert legs[-1]["model"] == 5
    search, partial, tour = fixture(swept=True)
    swept = pack_inputs(search, [request(search, partial, tour)])[3]
    assert swept[-1]["model"] == 4 and swept[-1]["flat"] == 0.83
    heuristic = pack_inputs(search, [request(search, partial, tour, False)])[3]
    assert heuristic[heuristic["role"] == 3]["model"].tolist() == [0]
    assert heuristic[-1]["model"] == 3


def test_completion_explicit_mode_needs_scope_and_supported_methods(monkeypatch):
    search, partial, tour = fixture()
    rows = [request(search, partial, tour)]
    monkeypatch.setenv(ENVIRONMENT, "0")
    assert finish_many(search, rows) is None
    monkeypatch.setenv(ENVIRONMENT, "1")
    with (
        using_lambert_backend("numpy"),
        pytest.raises(RuntimeError, match="requires using_lambert_backend"),
    ):
        finish_many(search, rows)
    monkeypatch.setattr(search, "_propellant", lambda *args: 0.0)
    with pytest.raises(ValueError, match="overridden _propellant"):
        pack_inputs(search, rows)
    with pytest.raises(RuntimeError, match="native completion ABI"):
        GpuCompletion(SimpleNamespace(library=SimpleNamespace()), 2, 4, 6)


@pytest.mark.parametrize("epoch", [123, np.float64(123.0), float("nan"), float("inf")])
def test_completion_epoch_metadata_restriction_is_explicit(epoch):
    search, partial, tour = fixture()
    row = request(search, partial, tour)
    row[1][11] = epoch
    with pytest.raises(ValueError, match="finite exact Python float epochs"):
        pack_inputs(search, [row])


def test_batched_completion_cannot_bypass_custom_tour_validation(monkeypatch):
    search, partial, tour = fixture()
    row = request(search, partial, tour)
    monkeypatch.setattr(search, "_plan_from_tour", lambda *args: None)
    with pytest.raises(ValueError, match="overridden _plan_from_tour"):
        pack_inputs(search, [row])


@requires_gpu
@pytest.mark.parametrize("model", ["fit", "flat", "ratio", "certified"])
def test_native_completion_matches_scalar_for_ragged_batches_and_workspace_reuse(
    monkeypatch, model
):
    search, partial, tour = fixture(fit=model == "fit", swept=model == "certified")
    if model == "ratio":
        search.settings = dataclasses.replace(search.settings, hop_inflation_slope=0.3)
    rows = []
    for index in range(259):
        p, t = deepcopy(partial), deepcopy(tour)
        p.mass = 650.0 + 4.0 * (index % 200)
        row = request(search, p, t)
        if index % 17 == 0:
            row[2].pop(11)
        elif index % 19 == 0:
            row[2][11] = row[1][11]
        elif index % 23 == 0:
            row[3][-1] = dataclasses.replace(row[3][-1], delta_v_proxy_km_s=100.0)
        elif index % 29 == 0:
            # An extra camp changes the ragged segment length without changing fuel/cargo.
            row[3].insert(0, PlannedLeg(12, 12, p.epoch, p.epoch, 0.0, 7.0, "camp"))
        rows.append(row)
    expected = []
    for p, deploy, collect, legs, use_table in rows:
        plan = search._finish_cpu(p, deploy, collect, legs, use_collect_table=use_table)
        expected.append((plan, search.last_failure if plan is None else ""))
    assert any(plan is not None for plan, _ in expected)
    assert any(plan is None for plan, _ in expected)
    monkeypatch.setenv(ENVIRONMENT, "1")
    with using_lambert_backend("cuda", maximum_batch_size=7) as gpu:
        actual = search._finish_many(rows)
        workspace = gpu.completion_workspace
        pointers = [a.ctypes.data for a in workspace.inputs]
        # A smaller subsequent call must overwrite metadata and reuse all allocations.
        repeated = search._finish_many(rows[:3])
        assert gpu.completion_workspace is workspace
        assert [a.ctypes.data for a in workspace.inputs] == pointers
        assert gpu.telemetry["completion_batches"] == 2
        assert gpu.telemetry["completion_candidates"] == 262
        for (got, failure), (wanted, reason) in zip(
            actual + repeated, expected + expected[:3], strict=True
        ):
            assert failure == reason
            assert (got is None) == (wanted is None)
            if got is None:
                continue
            assert (
                got.deploy_epochs == wanted.deploy_epochs
                and got.collect_epochs == wanted.collect_epochs
            )
            assert list(got.collected_mass) == list(wanted.collected_mass)
            assert got.collected_mass == pytest.approx(wanted.collected_mass, rel=2e-14, abs=2e-12)
            assert got.final_mass_proxy_kg == pytest.approx(
                wanted.final_mass_proxy_kg, rel=2e-14, abs=2e-11
            )
            assert got.propellant_proxy_kg == pytest.approx(
                wanted.propellant_proxy_kg, rel=2e-14, abs=2e-11
            )
            assert len(got.legs) == len(wanted.legs)
            for a, b in zip(got.legs, wanted.legs, strict=True):
                assert dataclasses.replace(a, inflation=b.inflation) == b
                assert a.inflation == pytest.approx(b.inflation, rel=2e-14, abs=2e-12)
