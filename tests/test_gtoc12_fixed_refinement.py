"""Fixed-cargo mass events, failure retention, finite claims and fleet identity."""

from __future__ import annotations

import ctypes
import dataclasses
import json
import math
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import fixed_refinement as f
from spacepdhcg.gtoc12 import pipeline as p
from spacepdhcg.gtoc12.refinement_admission import (
    AdmissionCandidate,
    CompletionObservation,
    FixedCargoRequest,
    RefinementAdmissionQueue,
)


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only tests cannot load a native solver")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(p, "body_state", lambda catalogue, body, epoch: (np.zeros(3), np.zeros(3)))


@pytest.fixture
def camp_plan():
    return p.RoutePlan(
        (
            p.PlannedLeg(0, 1, 64328, 64828, 1, 1, "earth_out"),
            p.PlannedLeg(1, 2, 64828, 65028, 1, 1, "deploy_hop"),
            p.PlannedLeg(2, 2, 65028, 65628, 0, 1, "camp"),
            p.PlannedLeg(2, 1, 65628, 66628, 1, 1, "collect_hop"),
            p.PlannedLeg(1, 0, 66628, 67328, 1, 1, "earth_return"),
        ),
        {1: 64828, 2: 65028},
        {1: 66628, 2: 65628},
        {1: 30.0, 2: 10.0},
        5000,
        100,
    )


def settings():
    return p.ScvxSettings(
        discretisation_backend="cuda",
        assembly_backend="cuda",
        convex_solver_backend="qoco",
        outer_loop_backend="cuda",
    )


class LegRunner:
    def __init__(self, *, burn=100, failure=None, exception=None, master=True):
        self.burn, self.failure, self.exception, self.master = burn, failure, exception, master
        self.boundaries, self.closed, self.master_called = [], False, False

    def solve(self, leg, boundary, index):
        self.boundaries.append(boundary)
        if index == self.exception:
            raise RuntimeError("native failure fixture")
        ok = index != self.failure
        item = p.RefinedLeg(
            leg,
            SimpleNamespace(deterministic_id=index),
            SimpleNamespace(feasible=ok),
            None,
            None,
            ok,
            boundary.initial_mass,
            boundary.initial_mass - self.burn,
        )
        return item, {"leg": index, "status": "converged" if ok else "failed"}

    def certify_route(self, plan, refined):
        self.master_called = True
        return self.master

    @property
    def telemetry(self):
        return {"submitted": len(self.boundaries)}

    def close(self):
        self.closed = True


def test_fixed_cargo_preserves_exact_mass_events_even_when_proxy_failed(camp_plan):
    plan = camp_plan
    original = deepcopy(plan.summary())
    runner = LegRunner()
    result = f.refine_fixed(
        plan, None, plan.collected_mass, settings=settings(), runner_factory=lambda _: runner
    )
    assert not plan.feasible
    assert result.certified and result.master_certified and result.passes == 1
    assert [x.initial_mass for x in runner.boundaries] == [3000, 2860, 2730, 2660]
    assert [x.minimum_final_mass for x in runner.boundaries] == [500, 500, 510, 540]
    assert result.final_mass_kg == 2520
    assert result.collected_mass == {1: 30.0, 2: 10.0}
    assert plan.summary() == original
    assert runner.closed and runner.master_called


@pytest.mark.parametrize("mode", ["leg", "exception", "mass", "master"])
def test_failed_refinement_never_shrinks_retimes_retries_or_hides_failure(camp_plan, mode):
    options = {
        "leg": {"failure": 1},
        "exception": {"exception": 1},
        "mass": {"burn": 650},
        "master": {"master": False},
    }[mode]
    runner = LegRunner(**options)
    original = deepcopy(camp_plan.summary())
    result = f.refine_fixed(
        camp_plan,
        None,
        camp_plan.collected_mass,
        settings=settings(),
        runner_factory=lambda _: runner,
    )
    assert not result.certified and result.failures and result.passes == 1
    assert camp_plan.summary() == original
    assert result.collected_mass == camp_plan.collected_mass
    assert len(runner.boundaries) == (2 if mode in {"leg", "exception"} else 4)
    assert runner.closed
    if mode == "mass":
        assert result.final_mass_kg == 320
        assert result.failures[-1]["status"] == "fixed_cargo_mass_deficit"
        assert result.failures[-1]["cargo_resized"] is False
        assert not runner.master_called


def test_expired_deadline_submits_no_leg_and_closes_the_runner(camp_plan):
    runner = LegRunner()
    result = f.refine_fixed(
        camp_plan,
        None,
        camp_plan.collected_mass,
        settings=settings(),
        deadline=0,
        runner_factory=lambda _: runner,
    )
    assert not runner.boundaries and runner.closed
    assert not result.certified and result.failures[0]["status"] == "campaign_deadline_before_leg"


def test_fixed_refiner_detects_prescription_mutation_even_after_a_failed_leg(camp_plan):
    runner = LegRunner(failure=0)

    def corrupt(index, item, record):
        camp_plan.collected_mass[1] -= 1

    with pytest.raises(AssertionError, match="mutated"):
        f.refine_fixed(
            camp_plan,
            None,
            camp_plan.collected_mass,
            settings=settings(),
            runner_factory=lambda _: runner,
            on_leg=corrupt,
        )
    assert runner.closed


def test_direct_refiner_rejects_disconnected_flights_before_runner_creation(camp_plan):
    legs = list(camp_plan.legs)
    legs[3] = dataclasses.replace(legs[3], from_id=99)
    disconnected = dataclasses.replace(camp_plan, legs=tuple(legs))
    created = []
    with pytest.raises(ValueError, match="discontinuous"):
        f.refine_fixed(
            disconnected,
            None,
            disconnected.collected_mass,
            settings=settings(),
            runner_factory=lambda value: created.append(value),
        )
    assert not created


def test_native_leg_record_preserves_actual_certification_backend(camp_plan):
    leg = camp_plan.legs[0]
    boundary = p.LegBoundary(
        leg.departure_epoch,
        np.zeros(3),
        np.zeros(3),
        leg.arrival_epoch,
        np.zeros(3),
        np.zeros(3),
        3000,
    )
    records = {}
    certificate = p.LegCertificate(0, 0, 2900, 1, 0, 0)
    state = SimpleNamespace(certificate=certificate, solution=None, certification_backend="cuda")
    runner = object.__new__(f.FrozenLegRunner)
    runner.registry = SimpleNamespace(
        records=records,
        register=lambda request, value: records.update({request.deterministic_id: state}),
    )
    result = SimpleNamespace(
        feasible=True, status=SimpleNamespace(value="converged"), diagnostic=""
    )
    runner.scheduler = SimpleNamespace(run=lambda requests: [result])
    runner.certifier = SimpleNamespace(
        certify=lambda value: SimpleNamespace(accepted=True, diagnostic="")
    )
    refined, record = runner.solve(leg, boundary, 0)
    assert refined.certification_backend == record["certification_backend"] == "cuda"
    assert refined.certified


def queued(plan, gain=5.0, uncertain=True, replay=False):
    request = FixedCargoRequest.from_summary(plan.summary())
    observed = CompletionObservation(
        request.sha256,
        "a" * 64,
        "b" * 64,
        "mass_below_dry_plus_collected" if uncertain else "",
        1,
        1,
        (4,),
        500 + request.cargo_kg + (-1 if uncertain else 1),
        request.cargo_kg,
        True,
    )
    queue = RefinementAdmissionQueue(replay_controls=replay)
    assert (
        queue.offer(AdmissionCandidate(request, observed, gain, 15000, 23)) == "refinement_eligible"
    )
    return queue, request


def full_check(route, request_sha):
    return {
        "request_sha256": request_sha,
        "result_sha256": "d" * 64,
        "independent": {
            "ok": True,
            "result_sha256": "d" * 64,
            "ships": 23,
            "total_mass_kg": 15000.0,
            "weighted_score_fixed_bonus_kg": 12811.0,
        },
        "official": {"ok": True, "result_sha256": "d" * 64},
    }


def invoke_queue(plan, *, refine=None, verify=full_check, deadline=None, replay=False):
    queue, request = queued(plan, gain=0 if replay else 5, replay=replay)
    records = []

    def simple_refine(prescribed, catalogue, cargo, **kwargs):
        return f.refine_fixed(
            prescribed, catalogue, cargo, **kwargs, runner_factory=lambda _: LegRunner()
        )

    outcomes = f.run_refinement_queue(
        queue,
        lambda key: plan,
        catalogue=None,
        settings=settings(),
        verify_full_fleet=verify,
        incumbent_weighted_kg=12810.0,
        deadline=time.perf_counter() + 10 if deadline is None else deadline,
        on_result=records.append,
        refine=simple_refine if refine is None else refine,
    )
    assert outcomes == records
    return queue, request, outcomes


def test_claimed_uncertain_route_must_use_fixed_refiner_and_actual_fleet_context(camp_plan):
    original = deepcopy(camp_plan.summary())
    calls = []

    def verify(route, key):
        assert route.certified and route.master_certified and route.passes == 1
        assert FixedCargoRequest.from_summary(route.plan.summary()).sha256 == key
        calls.append(key)
        return full_check(route, key)

    queue, request, outcomes = invoke_queue(camp_plan, verify=verify)
    assert queue.claimed == 1 and calls == [request.sha256]
    assert outcomes[0]["status"] == "verified_gain" and outcomes[0]["accepted"]
    assert camp_plan.summary() == original


@pytest.mark.parametrize(
    "failure",
    ["solver", "shrink", "retime", "leg", "official", "request", "result", "weight", "raw"],
)
def test_claimed_failure_is_retained_without_fleet_promotion_or_budget_refund(camp_plan, failure):
    original = deepcopy(camp_plan.summary())
    checks = []

    def refine(plan, catalogue, cargo, **kwargs):
        if failure == "solver":
            raise RuntimeError("retained native exception")
        route = f.refine_fixed(
            plan, catalogue, cargo, **kwargs, runner_factory=lambda _: LegRunner()
        )
        if failure == "shrink":
            route.collected_mass[1] -= 1
        elif failure == "retime":
            route.plan = dataclasses.replace(
                route.plan,
                legs=(
                    *route.plan.legs[:-1],
                    dataclasses.replace(route.plan.legs[-1], arrival_epoch=67329),
                ),
            )
        elif failure == "leg":
            route.legs.pop()
        return route

    def verify(route, key):
        checks.append(key)
        result = full_check(route, key)
        if failure == "official":
            result["official"]["ok"] = False
        elif failure == "request":
            result["request_sha256"] = "e" * 64
        elif failure == "result":
            result["official"]["result_sha256"] = "e" * 64
        elif failure == "weight":
            result["independent"]["weighted_score_fixed_bonus_kg"] = math.nan
        elif failure == "raw":
            result["independent"]["total_mass_kg"] = 1000.0
        return result

    queue, _, outcomes = invoke_queue(camp_plan, refine=refine, verify=verify)
    assert queue.claimed == 1 and queue.claim_next() is None
    assert len(outcomes) == 1 and not outcomes[0]["accepted"]
    assert outcomes[0]["status"] != "verified_gain"
    assert len(checks) == (0 if failure in {"solver", "shrink", "retime", "leg"} else 1)
    assert camp_plan.summary() == original


def test_wrong_captured_plan_is_not_refined(camp_plan):
    queue, _ = queued(camp_plan)
    changed = deepcopy(camp_plan)
    changed.collected_mass[1] -= 1
    calls = []
    rows = f.run_refinement_queue(
        queue,
        lambda key: changed,
        catalogue=None,
        settings=settings(),
        verify_full_fleet=full_check,
        incumbent_weighted_kg=12810,
        deadline=time.perf_counter() + 10,
        on_result=lambda x: None,
        refine=lambda *args, **kwargs: calls.append(args),
    )
    assert not calls and queue.claimed == 1
    assert "does not match" in rows[0]["error"]


def test_control_replay_and_expired_queue_cannot_trigger_refinement(camp_plan):
    with pytest.raises(RuntimeError, match="cannot launch"):
        invoke_queue(camp_plan, replay=True)
    queue, _, rows = invoke_queue(camp_plan, deadline=0)
    assert not rows and queue.claimed == 0


@pytest.mark.parametrize("encoding", ["summary", "bytes"])
def test_captured_mapping_rebuilds_failed_proxy_for_fixed_cargo_path(camp_plan, encoding):
    queue, request = queued(camp_plan)
    captured = camp_plan.summary()
    assert not captured["feasible_proxy"]
    if encoding == "bytes":
        captured = json.dumps(captured).encode()

    def refine(plan, catalogue, cargo, **kwargs):
        assert not plan.feasible
        return f.refine_fixed(
            plan, catalogue, cargo, **kwargs, runner_factory=lambda _: LegRunner()
        )

    records = f.run_refinement_queue(
        queue,
        {request.sha256: captured},
        catalogue=None,
        settings=settings(),
        verify_full_fleet=full_check,
        incumbent_weighted_kg=12810,
        deadline=time.perf_counter() + 10,
        on_result=lambda x: None,
        refine=refine,
    )
    assert len(records) == 1 and records[0]["status"] == "verified_gain"
