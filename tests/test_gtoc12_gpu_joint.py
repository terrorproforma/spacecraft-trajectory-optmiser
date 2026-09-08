"""Continuous-itinerary CUDA parity; these surrogate checks do not certify physics.

Run serially on an idle GPU with SPACEPDHCG_GTOC12_GPU_TESTS=1 and
SPACEPDHCG_GTOC12_CUDA_LIBRARY pointing at the candidate core. The incumbent
directory can be supplied as SPACEPDHCG_GTOC12_JOINT_ARCHIVE; catalogue/bonus
files use the usual checksum-verified SPACEPDHCG_GTOC12_DATA location.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import replace
from itertools import pairwise, product
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.bundles import (
    ClusterPricingSettings,
    cluster_retime_settings,
    cluster_search_settings,
)
from spacepdhcg.gtoc12.data import data_available, load_bonus_table, load_catalogue
from spacepdhcg.gtoc12.gpu_joint import COST, POLICY, RESULT, STAGE, VISIT, evaluate_joint
from spacepdhcg.gtoc12.jointopt import JointItinerary, MeasuredLeg, route_from_summary
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.retiming import Retimer, Visit, visits_of
from spacepdhcg.gtoc12.screening import exhaust_velocity_km_s

JOINT_FLAG = "SPACEPDHCG_TEST_GTOC12_JOINT_BATCH"
ARCHIVE = Path(
    os.environ.get(
        "SPACEPDHCG_GTOC12_JOINT_ARCHIVE",
        str(
            Path(__file__).resolve().parents[1]
            / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
        ),
    )
)
requires_gpu = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires explicitly enabled, serialized CUDA tests",
)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setenv(JOINT_FLAG, "1")
    # A deliberately partial geometry batch exercises retained Lambert storage.
    with using_lambert_backend("cuda", maximum_batch_size=97) as backend:
        yield backend


@pytest.fixture(scope="module")
def catalogue_and_bonus():
    if not data_available("GTOC12_Asteroids_Data.txt", "bonus_coefficients.txt"):
        pytest.skip("pinned GTOC12 catalogue/bonus data not fetched")
    return load_catalogue(), load_bonus_table()


@pytest.fixture(params=("ship-01.json", "ship-02.json"))
def incumbent(request, catalogue_and_bonus):
    path = ARCHIVE / request.param
    if not path.is_file():
        pytest.skip(f"incumbent summary unavailable: {path}")
    catalogue, bonus = catalogue_and_bonus
    route = route_from_summary(json.loads(path.read_text(encoding="utf-8")))
    weights = {body: bonus.for_asteroid(body) for body in route.collected_mass}
    pricing = ClusterPricingSettings(collect_dp_inflation_fit="")
    retimer = Retimer(
        catalogue,
        cluster_search_settings(pricing, 40),
        cluster_retime_settings(pricing, last=True),
        weights,
    )
    joint = JointItinerary(catalogue, retimer, weights=weights)
    assert route.certified and joint.learn(route) == len(route.legs)
    assert joint.measured and retimer.inflations
    visits, arrivals, departures = visits_of(route.plan)
    # visits_of is shared with the scalar path; foreign donors live in the plan.
    visits = [
        replace(v, foreign_deploy_epoch=route.plan.foreign_deploy_epochs.get(v.body))
        if v.collect and v.body not in route.plan.deploy_epochs
        else v
        for v in visits
    ]
    return joint, visits, np.asarray(arrivals), np.asarray(departures)


def _synthetic(mode="mixed", *, camp_only=False):
    pricing = ClusterPricingSettings(collect_dp_inflation_fit="")
    settings = replace(
        cluster_retime_settings(pricing, last=True),
        camp_max_days=1200.0,
        hop_inflation_slope=0.65 if mode in ("ratio", "mixed") else None,
        return_tof_model=mode in ("return", "mixed"),
    )
    weights = {1: 0.7, 2: 1.2}
    retimer = Retimer(None, cluster_search_settings(pricing, 40), settings, weights)
    retimer.inflations.update({(0, 1): 0.92, (1, 2): 1.08, (2, 1): 1.05, (1, 0): 0.96})
    joint = JointItinerary(None, retimer, weights=weights)
    if camp_only:
        visits = [
            Visit(0, False, False, "earth_out"),
            Visit(1, True, True, "earth_return"),
            Visit(0, False, False, ""),
        ]
        arr = C.MISSION_START_MJD + np.array([100.0, 600.0, 1955.0])
        dep = C.MISSION_START_MJD + np.array([100.0, 1400.0, 1955.0])
        costs = [1.4, 1.2]
    else:
        visits = [
            Visit(0, False, False, "earth_out"),
            Visit(1, True, False, "deploy_hop"),
            Visit(2, True, True, "collect_hop"),
            Visit(1, False, True, "earth_return"),
            Visit(0, False, False, ""),
        ]
        arr = C.MISSION_START_MJD + np.array([100.0, 600.0, 780.0, 1980.0, 2535.0])
        dep = C.MISSION_START_MJD + np.array([100.0, 600.0, 1800.0, 1980.0, 2535.0])
        costs = [1.4, 0.6, 0.7, 1.2]
    return joint, visits, arr, dep, costs


def _moves(visits, arr, dep, delta=3.0):
    arrivals, departures = [arr.copy()], [dep.copy()]
    for shift in JointItinerary.moves(len(visits), delta):
        a, d = arr.copy(), dep.copy()
        for j, (da, dd) in shift.items():
            a[j] += da
            d[j] += dd
        arrivals.append(a)
        departures.append(d)
    return np.asarray(arrivals), np.asarray(departures)


def _cache(joint, visits, arrivals, departures, costs):
    for arr, dep in zip(np.atleast_2d(arrivals), np.atleast_2d(departures), strict=True):
        for j, (visit, nxt) in enumerate(pairwise(visits)):
            joint._lambert[joint.key(visit.body, nxt.body, dep[j], arr[j + 1])] = costs[j]


def _warm_geometry(gpu, joint, visits, arrivals, departures):
    """Supply identical real costs to both evaluators, isolating their arithmetic."""
    for j, (visit, nxt) in enumerate(pairwise(visits)):
        hops = gpu.paired_hops(
            joint.catalogue,
            visit.body,
            nxt.body,
            departures[:, j],
            arrivals[:, j + 1] - departures[:, j],
        )
        for row, (cost, feasible) in enumerate(zip(hops.total_delta_v, hops.feasible, strict=True)):
            key = joint.key(visit.body, nxt.body, departures[row, j], arrivals[row, j + 1])
            joint._lambert.setdefault(key, float(cost) if feasible else math.inf)


def _scalar(monkeypatch, joint, visits, arrivals, departures):
    with monkeypatch.context() as scalar:
        scalar.setenv(JOINT_FLAG, "0")
        return [
            joint.evaluate(visits, arr, dep)
            for arr, dep in zip(np.atleast_2d(arrivals), np.atleast_2d(departures), strict=True)
        ]


def _same(actual, expected, *, atol=5e-9, rtol=5e-12):
    assert actual.feasible == expected.feasible
    assert actual.failure == expected.failure
    assert actual.measured_legs == expected.measured_legs
    fields = ("objective", "weighted_kg", "collected_kg", "spare_kg", "propellant_kg")
    np.testing.assert_allclose(
        [getattr(actual, f) for f in fields],
        [getattr(expected, f) for f in fields],
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(actual.masses, expected.masses, atol=atol, rtol=rtol)
    if expected.plan is None:
        assert actual.plan is None
        return
    a, e = actual.plan, expected.plan
    assert a.deploy_epochs == e.deploy_epochs
    assert a.collect_epochs == e.collect_epochs
    assert a.foreign_deploy_epochs == e.foreign_deploy_epochs
    assert list(a.collected_mass) == list(e.collected_mass)
    np.testing.assert_allclose(
        list(a.collected_mass.values()), list(e.collected_mass.values()), atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        [a.propellant_proxy_kg, a.final_mass_proxy_kg],
        [e.propellant_proxy_kg, e.final_mass_proxy_kg],
        atol=atol,
        rtol=rtol,
    )
    for a_leg, e_leg in zip(a.legs, e.legs, strict=True):
        assert (a_leg.from_id, a_leg.to_id, a_leg.role) == (
            e_leg.from_id,
            e_leg.to_id,
            e_leg.role,
        )
        assert (a_leg.departure_epoch, a_leg.arrival_epoch) == (
            e_leg.departure_epoch,
            e_leg.arrival_epoch,
        )
        np.testing.assert_allclose(
            [a_leg.delta_v_proxy_km_s, a_leg.inflation],
            [e_leg.delta_v_proxy_km_s, e_leg.inflation],
            atol=atol,
            rtol=rtol,
        )


def _compare(monkeypatch, joint, visits, arr, dep, costs):
    arrivals, departures = np.atleast_2d(arr), np.atleast_2d(dep)
    _cache(joint, visits, arrivals, departures, costs)
    expected = _scalar(monkeypatch, joint, visits, arrivals, departures)
    actual = evaluate_joint(joint, visits, arrivals, departures)
    assert actual is not None and len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        _same(a, e)
    assert joint.lambert_evaluations == 0, "synthetic cases must use their supplied costs"
    return actual


def test_joint_c_abi_layout():
    # Sizes and offsets are the ordinary C ABI in gtoc12_joint_c_api.h, not a packed ABI.
    for dtype, size, first_double, offset in (
        (VISIT, 48, "foreign_epoch", 16),
        (STAGE, 72, "tof_min", 16),
        (COST, 32, "lambert", 8),
        (POLICY, 128, "mission_start", 0),
        (RESULT, 64, "objective", 16),
    ):
        assert dtype.itemsize == size and dtype.alignment == 8
        assert dtype.fields[first_double][1] == offset
    assert POLICY.fields["free_earth_leg"][1] == 112
    assert RESULT.fields["final_mass"][1] == 56


def test_nearby_exact_epoch_keys_keep_costs_and_measurements_separate(monkeypatch):
    joint, visits, arr, dep, costs = _synthetic(camp_only=True)
    nearby_arr, nearby_dep = arr.copy(), dep.copy()
    nearby_arr[0] += 2e-6
    nearby_dep[0] += 2e-6
    original = joint.key(0, 1, dep[0], arr[1])
    nearby = joint.key(0, 1, nearby_dep[0], nearby_arr[1])
    assert round(dep[0], 5) == round(nearby_dep[0], 5) and original != nearby
    _cache(joint, visits, arr, dep, costs)
    _cache(joint, visits, nearby_arr, nearby_dep, [1.8, costs[1]])
    joint.measured[original] = MeasuredLeg(1.1, C.MAX_INITIAL_MASS_KG, costs[0])
    evaluations = _scalar(
        monkeypatch, joint, visits, np.array([arr, nearby_arr]), np.array([dep, nearby_dep])
    )
    assert all(e.feasible for e in evaluations)
    assert [e.measured_legs for e in evaluations] == [1, 0]
    assert joint._lambert[original] == costs[0] and joint._lambert[nearby] == 1.8
    assert evaluations[0].objective > evaluations[1].objective
    assert joint.lambert_evaluations == 0


def test_disabled_cuda_joint_streams_cpu_moves(monkeypatch):
    from spacepdhcg.gtoc12 import gpu_joint

    joint, visits, arr, dep, costs = _synthetic()
    arrivals, departures = _moves(visits, arr, dep)
    _cache(joint, visits, arrivals, departures, costs)
    monkeypatch.setenv(JOINT_FLAG, "0")
    original_moves = joint.moves

    def streaming_moves(n, delta):
        started = joint.evaluations
        for index, move in enumerate(original_moves(n, delta)):
            yield move
            # Each yielded move must finish before the next one is constructed.
            assert joint.evaluations == started + index + 1

    def no_batch(*args, **kwargs):
        pytest.fail("disabled joint backend should not receive a trial matrix")

    monkeypatch.setattr(joint, "moves", streaming_moves)
    monkeypatch.setattr(gpu_joint, "evaluate_joint", no_batch)
    result = joint.optimise_epochs(visits, arr, dep, mesh=(3.0,), max_moves=1)
    assert result[3] == 1 and joint.lambert_evaluations == 0


@requires_gpu
def test_nearby_finite_rows_keep_distinct_cached_leg_costs(monkeypatch, gpu):
    joint, visits, arr, dep, costs = _synthetic()
    arrivals, departures = np.tile(arr, (2, 1)), np.tile(dep, (2, 1))
    arrivals[1, 0] += 2e-6
    departures[1, 0] += 2e-6
    departures[1, -2] += 2e-6
    # Row zero stops at its first authority gate. Its unused return leg must
    # never alias row one's nearby but distinct return epoch.
    _cache(joint, visits, arrivals[0], departures[0], [100.0, *costs[1:]])
    _cache(joint, visits, arrivals[1], departures[1], [*costs[:-1], 0.9])
    first_return = joint.key(1, 0, departures[0, -2], arrivals[0, -1])
    second_return = joint.key(1, 0, departures[1, -2], arrivals[1, -1])
    assert first_return != second_return
    assert round(first_return[2], 5) == round(second_return[2], 5)
    expected = _scalar(monkeypatch, joint, visits, arrivals, departures)
    actual = evaluate_joint(joint, visits, arrivals, departures)
    assert [e.failure for e in actual] == ["leg_authority", ""]
    for a, e in zip(actual, expected, strict=True):
        _same(a, e)
    assert actual[1].plan.legs[-1].delta_v_proxy_km_s == 0.9
    assert joint.lambert_evaluations == 0


@requires_gpu
def test_initial_workspace_reserves_full_neighbourhood(monkeypatch, gpu):
    joint, visits, arr, dep, costs = _synthetic()
    arrivals, departures = _moves(visits, arr, dep)
    _cache(joint, visits, arrivals, departures, costs)
    assert joint.evaluate(visits, arr, dep).feasible
    workspace = gpu.joint_workspace
    assert workspace.capacity >= len(arrivals) - 1
    handle = workspace.handle.value
    expected = _scalar(monkeypatch, joint, visits, arrivals[1:], departures[1:])
    actual = evaluate_joint(joint, visits, arrivals[1:], departures[1:])
    assert gpu.joint_workspace is workspace and workspace.handle.value == handle
    for a, e in zip(actual, expected, strict=True):
        _same(a, e)


@requires_gpu
def test_incumbent_mesh_arithmetic_and_retained_workspace(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    arrivals, departures = _moves(visits, arr, dep)
    _warm_geometry(gpu, joint, visits, arrivals, departures)
    geometry_calls = gpu.evaluations
    expected = _scalar(monkeypatch, joint, visits, arrivals, departures)
    assert expected[0].feasible, expected[0].failure
    assert expected[0].measured_legs == len(visits) - 1
    actual = evaluate_joint(joint, visits, arrivals, departures)
    for a, e in zip(actual, expected, strict=True):
        _same(a, e)
    workspace = gpu.joint_workspace
    handle = workspace.handle.value
    shorter = evaluate_joint(joint, visits, arrivals[:3], departures[:3])
    assert gpu.joint_workspace is workspace and workspace.handle.value == handle
    for a, e in zip(shorter, expected[:3], strict=True):
        _same(a, e)
    assert evaluate_joint(joint, visits, arrivals[:0], departures[:0]) == []
    assert gpu.evaluations == geometry_calls and joint.lambert_evaluations == 0
    assert gpu.telemetry["completed_joint_evaluations"] == len(arrivals) + 3


@requires_gpu
def test_empty_cache_scalar_cuda_matches_batched_cuda(monkeypatch, gpu, incumbent):
    scalar_joint, visits, arr, dep = incumbent
    batch_joint = deepcopy(scalar_joint)
    arrivals, departures = (x[:17] for x in _moves(visits, arr, dep))
    assert not scalar_joint._lambert and not batch_joint._lambert
    expected = _scalar(monkeypatch, scalar_joint, visits, arrivals, departures)
    before = gpu.telemetry.get("completed_element_hops", 0)
    actual = evaluate_joint(batch_joint, visits, arrivals, departures)
    assert scalar_joint.lambert_evaluations > 0 and batch_joint.lambert_evaluations > 0
    assert gpu.telemetry["completed_element_hops"] > before
    for a, e in zip(actual, expected, strict=True):
        # This comparison also includes GPU versus NumPy orbital propagation.
        _same(a, e, atol=5e-5, rtol=2e-8)
    for key in scalar_joint._lambert.keys() & batch_joint._lambert.keys():
        np.testing.assert_allclose(
            batch_joint._lambert[key], scalar_joint._lambert[key], atol=2e-8, rtol=2e-9
        )


@requires_gpu
@pytest.mark.parametrize("mode", ["flat", "ratio", "return", "mixed"])
def test_all_inflation_modes_and_pair_calibration(monkeypatch, gpu, mode):
    joint, visits, arr, dep, costs = _synthetic(mode)
    arrivals, departures = _moves(visits, arr, dep)
    result = _compare(monkeypatch, joint, visits, arrivals, departures, costs)
    assert result[0].feasible and result[0].weighted_kg != result[0].collected_kg
    assert any(leg.inflation != 1.0 for leg in result[0].plan.legs if leg.role != "camp")


@requires_gpu
def test_epoch_gate_order_pins_and_mining(monkeypatch, gpu):
    joint, visits, arr, dep, costs = _synthetic()
    visits[1] = replace(visits[1], pinned_arrival=arr[1])
    arrivals, departures = np.tile(arr, (9, 1)), np.tile(dep, (9, 1))
    arrivals[1, 0] = departures[1, 0] = C.MISSION_START_MJD - 1
    arrivals[1, -1] = departures[1, -1] = C.MISSION_END_MJD + 1  # launch wins
    arrivals[2, -1] = departures[2, -1] = C.MISSION_END_MJD + 1
    departures[3, 0] += 1
    departures[4, 1] = arr[1] - 1
    departures[5, 1] = arr[1] + joint.dwell_limit(visits[1]) + 1
    arrivals[6, 1] += 1
    departures[6, 1] += 1
    arrivals[7, 2] = dep[1] + joint.tof_limits("deploy_hop")[0] - 1
    departures[8, 2] = arr[2] + 300
    arrivals[8, 3] = departures[8, 3] = departures[8, 2] + 180
    arrivals[8, -1] = departures[8, -1] = departures[8, 3] + 555
    result = _compare(monkeypatch, joint, visits, arrivals, departures, costs)
    assert [r.failure for r in result] == [
        "",
        "launch_before_window",
        "return_after_window",
        "earth_visits_have_no_dwell",
        "negative_dwell",
        "dwell_too_long",
        "pinned_arrival_moved",
        "tof_outside_limits",
        "stay_too_short",
    ]


@requires_gpu
@pytest.mark.parametrize("failure", ["double_deploy", "double_collect", "collect_without_deploy"])
def test_structural_failures_precede_mining_and_geometry(monkeypatch, gpu, failure):
    joint, visits, arr, dep, _ = _synthetic()
    if failure == "double_deploy":
        visits[2] = replace(visits[2], body=visits[1].body)
    elif failure == "double_collect":
        visits[1] = replace(visits[1], collect=True)  # zero stay, but the duplicate wins
    else:
        visits[1] = replace(visits[1], deploy=False)
    expected = _scalar(monkeypatch, joint, visits, arr, dep)[0]
    actual = evaluate_joint(joint, visits, arr[None], dep[None])[0]
    assert actual.failure == failure
    _same(actual, expected)
    assert not joint._lambert and gpu.evaluations == 0  # catalogue=None must never be touched


@requires_gpu
@pytest.mark.parametrize("foreign", [False, True])
def test_cooperative_foreign_and_final_own_donor(monkeypatch, gpu, foreign):
    joint, visits, arr, dep, costs = _synthetic()
    if foreign:
        visits[2] = replace(visits[2], deploy=False, foreign_deploy_epoch=arr[2] - 100)
        actual = _compare(monkeypatch, joint, visits, arr, dep, costs)[0]
        assert actual.feasible and actual.plan.foreign_deploy_epochs == {2: arr[2] - 100}
        assert 2 not in actual.plan.deploy_epochs
    else:
        # A collect preceding its own future deployment uses the FINAL deploy map,
        # even though the ordered scan originally recorded a foreign donor.
        visits[1] = replace(
            visits[1], deploy=False, collect=True, foreign_deploy_epoch=arr[1] - 800
        )
        visits[3] = replace(visits[3], deploy=True, collect=False)
        actual = _compare(monkeypatch, joint, visits, arr, dep, costs)[0]
        assert actual.failure == "stay_too_short"


@requires_gpu
@pytest.mark.parametrize("lambert_cost", [math.inf, 0.0, 2.0])
@pytest.mark.parametrize("mass_offset", [60.0, 60.000001])
def test_measured_mass_tolerance_and_proxy_translation(monkeypatch, gpu, lambert_cost, mass_offset):
    joint, visits, arr, dep, costs = _synthetic(camp_only=True)
    costs[0] = lambert_cost
    key = joint.key(0, 1, dep[0], arr[1])
    joint.measured[key] = MeasuredLeg(2.0, C.MAX_INITIAL_MASS_KG + mass_offset, 2.0)
    actual = _compare(monkeypatch, joint, visits, arr, dep, costs)[0]
    accepted = mass_offset <= joint.settings.measured_mass_tolerance_kg
    assert actual.measured_legs == int(accepted)
    if math.isinf(lambert_cost) and not accepted:
        assert actual.failure == "leg_infeasible"
    else:
        assert actual.feasible
    if accepted:
        first = actual.plan.legs[0]
        assert first.delta_v_proxy_km_s == (lambert_cost if math.isfinite(lambert_cost) else 2.0)
        assert first.inflation == (
            2.0 / lambert_cost if lambert_cost > 0 and math.isfinite(lambert_cost) else 1.0
        )


@requires_gpu
@pytest.mark.parametrize(
    "failure", ["leg_infeasible", "leg_authority", "earth_out_unmeasured_below_floor"]
)
def test_leg_gates_and_earth_screening_override(monkeypatch, gpu, failure):
    joint, visits, arr, dep, costs = _synthetic(camp_only=True)
    if failure == "leg_infeasible":
        costs[0] = math.inf
    elif failure == "leg_authority":
        costs[0] = 100.0
    else:
        joint.free_earth_leg = True
        joint.retimer.earth_out_tof_floor = arr[1] - dep[0] + 30
    actual = _compare(monkeypatch, joint, visits, arr, dep, costs)[0]
    assert actual.failure == failure
    if failure == "earth_out_unmeasured_below_floor":
        joint._screen_earth_out = True
        joint.earth_out_inflation = 0.5
        actual = _compare(monkeypatch, joint, visits, arr, dep, costs)[0]
        assert actual.feasible and actual.plan.legs[0].inflation == 0.5
        joint.earth_out_inflation = None
        assert _compare(monkeypatch, joint, visits, arr, dep, costs)[0].feasible


@requires_gpu
@pytest.mark.parametrize(
    "remaining, failure",
    [(535.0, "mass_below_dry_plus_collected"), (510.0, "mass_below_dry")],
)
def test_payload_sizing_failure_diagnostics(monkeypatch, gpu, remaining, failure):
    joint, visits, arr, dep, costs = _synthetic(camp_only=True)
    costs[:] = [100.0, 0.0]
    dv = -exhaust_velocity_km_s() * math.log(remaining / C.MAX_INITIAL_MASS_KG)
    joint.measured[joint.key(0, 1, dep[0], arr[1])] = MeasuredLeg(dv, C.MAX_INITIAL_MASS_KG, 100.0)
    actual = _compare(monkeypatch, joint, visits, arr, dep, costs)[0]
    assert actual.failure == failure and actual.plan is None
    assert actual.objective == -math.inf and actual.weighted_kg == actual.collected_kg == 0
    if failure == "mass_below_dry_plus_collected":
        # Five kg remain missing after every pass; three resizings precede the
        # fourth returned pass. Its diagnostic masses must not be discarded.
        payload = C.maximum_collected_mass(dep[1] - arr[1]) - 3 * 1.02 * 5.0
        assert actual.spare_kg == pytest.approx(-5.0, abs=1e-9)
        assert actual.masses == pytest.approx([3000.0, 495.0 + payload], abs=1e-9)
        assert actual.propellant_kg == pytest.approx(2465.0, abs=1e-9)
        assert actual.measured_legs == 1
    else:
        assert actual.masses == [] and actual.measured_legs == 0 and actual.spare_kg == 0


@requires_gpu
def test_first_winning_tie_threshold_and_only_winner_materialization(monkeypatch, gpu):
    from spacepdhcg.gtoc12 import search

    joint, visits, arr, dep, costs = _synthetic()
    arrivals, departures = np.tile(arr, (5, 1)), np.tile(dep, (5, 1))
    arrivals[0, 0] = departures[0, 0] = C.MISSION_START_MJD - 1
    arrivals[2:4, 2] -= 3  # equal, strictly better mining stays
    _cache(joint, visits, arrivals, departures, costs)
    expected = _scalar(monkeypatch, joint, visits, arrivals, departures)
    assert expected[2].objective == expected[3].objective > expected[1].objective + 1e-9
    materialized = []
    route_plan = search.RoutePlan

    def record_plan(*args, **kwargs):
        plan = route_plan(*args, **kwargs)
        materialized.append(plan)
        return plan

    monkeypatch.setattr(search, "RoutePlan", record_plan)
    winner, actual = evaluate_joint(
        joint, visits, arrivals, departures, minimum_objective=expected[1].objective
    )
    assert winner == 2 and len(materialized) == 1
    _same(actual, expected[2])
    assert evaluate_joint(
        joint, visits, arrivals, departures, minimum_objective=actual.objective
    ) == (None, None)
    assert len(materialized) == 1


@requires_gpu
def test_two_accepted_mesh_moves_match_scalar_controller(monkeypatch, gpu):
    joint, visits, arr, dep, costs = _synthetic()
    delta = 3.0
    # Every epoch reachable within the bounded two-move search has one of these
    # offsets. No callback overrides or geometry work enter the controller test.
    for j, (visit, nxt) in enumerate(pairwise(visits)):
        for da, dd in product((-2 * delta, -delta, 0.0, delta, 2 * delta), repeat=2):
            joint._lambert[joint.key(visit.body, nxt.body, dep[j] + dd, arr[j + 1] + da)] = costs[j]
    scalar_joint = deepcopy(joint)
    with monkeypatch.context() as scalar:
        scalar.setenv(JOINT_FLAG, "0")
        expected = scalar_joint.optimise_epochs(visits, arr, dep, mesh=(delta,), max_moves=2)
    actual = joint.optimise_epochs(visits, arr, dep, mesh=(delta,), max_moves=2)
    assert actual[3] == expected[3] == 2
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    _same(actual[2], expected[2])
    assert joint.lambert_evaluations == scalar_joint.lambert_evaluations == 0
    assert gpu.telemetry["completed_joint_batches"] == 3  # initial plus two neighbourhoods


@requires_gpu
def test_shape_checks_and_custom_override_rejection(monkeypatch, gpu):
    joint, visits, arr, dep, _ = _synthetic()
    with pytest.raises(ValueError, match="matching"):
        evaluate_joint(joint, visits, arr, dep)
    with pytest.raises(ValueError, match="one arrival"):
        joint.evaluate(visits, arr[:-1], dep)
    with monkeypatch.context() as overridden:
        overridden.setattr(joint, "_forward", lambda *args: None)
        with pytest.raises(ValueError, match="overridden _forward"):
            evaluate_joint(joint, visits, arr[None], dep[None])
    assert gpu.joint_workspace is None and gpu.evaluations == 0
