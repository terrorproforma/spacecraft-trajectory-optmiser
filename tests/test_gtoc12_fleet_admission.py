"""Fleet accounting and production admission; no trajectory solve or propagation."""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import bundles as b
from spacepdhcg.gtoc12 import memory
from spacepdhcg.gtoc12.refinement_admission import (
    CompletionObservation,
    FixedCargoRequest,
    FleetBudgetContext,
    RefinementAdmissionQueue,
)
from spacepdhcg.gtoc12.solution import Event, ShipTrajectory, Solution, StateLine, format_solution


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("policy tests cannot load native libraries")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(memory, "release_heap", lambda: False)


def context(*, other_mass=200.0, weights=None, raw=False):
    """Synthetic checker-bound ledger, not a physically certified fixture.

    Reports are stubs solely to exercise identity/accounting; these tests do not
    certify the arbitrary state vectors or claim a mission score.
    """
    weights = weights or {1: 1.0, 2: 1.0, 3: 1.0, **{a: 2.0 for a in range(4, 105)}}
    masses = [10.0, other_mass, other_mass]
    ships = []
    for ship_id, mass in enumerate(masses, 1):

        def event(body, epoch, before, after):
            return Event(
                body,
                StateLine(epoch, np.ones(3), np.ones(3), before),
                StateLine(epoch, np.ones(3), np.ones(3), after),
            )

        ships.append(
            ShipTrajectory(
                ship_id,
                [
                    event(0, 64328, 3000, 3000),
                    event(ship_id, 65000, 2000, 1960),
                    event(ship_id, 66000, 1500, 1500 + mass),
                    event(-3, 66500, 1000 + mass, 1000),
                ],
            )
        )
    result = format_solution(Solution(ships)).encode("ascii")
    digest = hashlib.sha256(result).hexdigest()
    checked = {
        "ok": True,
        "ships": 3,
        "result_sha256": digest,
        "total_mass_kg": sum(masses),
        "weighted_score_fixed_bonus_kg": sum(weights[a] * m for a, m in enumerate(masses, 1)),
    }
    return (
        FleetBudgetContext.from_verified_result(
            result, independent=checked, official=checked, weights=None if raw else weights
        ),
        result,
        checked,
        weights,
    )


def plan(body=4, cargo=9.0, *, departure=66000):
    return b.RoutePlan(
        (
            b.PlannedLeg(0, body, 64328, 65000, 1, 1, "earth_out"),
            b.PlannedLeg(body, 0, departure, 66500, 1, 1, "earth_return"),
        ),
        {body: 65000},
        {body: departure},
        {body: cargo},
        500,
        501 + cargo,
    )


def envelope(route, *, deficit=1.0, salt="a"):
    request = FixedCargoRequest.from_summary(route.summary())
    observation = CompletionObservation(
        request.sha256,
        "a" * 64,
        hashlib.sha256(salt.encode()).hexdigest(),
        "mass_below_dry_plus_collected" if deficit > 0 else "",
        1,
        1,
        (4,),
        500 + request.cargo_kg - deficit,
        request.cargo_kg,
        True,
    )
    return SimpleNamespace(
        request=request, observation=observation, plan_summary=json.dumps(route.summary()).encode()
    )


def test_raw_negative_weighted_positive_uses_real_fleet_headroom():
    fleet, _, _, _ = context()
    request = FixedCargoRequest.from_summary(plan().summary())
    cost = fleet.replacement(1, request)
    assert cost.raw_gain_kg == -1
    assert cost.objective_gain_kg == 8
    assert cost.proposed_fleet_raw_kg == 409
    assert cost.score_kind == "weighted_fixed_bonus_kg"
    assert not cost.blocker
    queue = RefinementAdmissionQueue()
    assert queue.completion_consumer(fleet, 1)(envelope(plan())) == "refinement_eligible"
    selected = queue.shortlist()[0]
    assert selected.proposed_fleet_raw_kg == 409
    assert selected.weighted_gain_kg == 8
    assert queue.replacement_ship(selected.request.sha256) == 1


def test_positive_weighted_gain_cannot_spend_more_than_global_raw_budget():
    fleet, _, _, _ = context(other_mass=147.5)  # 305 kg passes; 304 kg does not.
    assert (
        fleet.replacement(1, FixedCargoRequest.from_summary(plan().summary())).blocker
        == "raw_fleet_ship_limit"
    )
    queue = RefinementAdmissionQueue()
    assert queue.completion_consumer(fleet, 1)(envelope(plan())) == "raw_fleet_ship_limit"
    assert not queue.retained() and not queue.plan_lookup


def test_raw_objective_is_explicit_and_no_missing_weight_falls_back():
    raw, _, _, _ = context(raw=True)
    assert raw.score_kind == "raw_kg"
    with pytest.raises(ValueError, match="requires a weighted objective"):
        RefinementAdmissionQueue().completion_consumer(raw, 1)
    assert (
        raw.replacement(1, FixedCargoRequest.from_summary(plan().summary())).blocker
        == "no_objective_improvement"
    )
    fleet, _, _, _ = context(weights={1: 1, 2: 1, 3: 1})
    with pytest.raises(ValueError, match="missing fixed bonus"):
        fleet.replacement(1, FixedCargoRequest.from_summary(plan().summary()))


def test_retained_asteroid_conflict_rejected_even_with_large_gain():
    fleet, _, _, _ = context()
    conflict = plan(2, 20)
    queue = RefinementAdmissionQueue()
    assert "conflicts" in queue.completion_consumer(fleet, 1)(envelope(conflict))
    assert not queue.shortlist()
    assert b.refine_candidates(
        [conflict, plan()],
        set(),
        b.ClusterPricingSettings(),
        fleet_context=fleet,
        replacement_ship=1,
    ) == [plan()]


@pytest.mark.parametrize("field,value", [("ok", False), ("result_sha256", "a" * 64), ("ships", 4)])
def test_context_requires_each_actual_result_binding(field, value):
    _, result, checked, weights = context()
    for changed in ("independent", "official"):
        reports = {"independent": dict(checked), "official": dict(checked)}
        reports[changed][field] = value
        with pytest.raises(ValueError):
            FleetBudgetContext.from_verified_result(result, **reports, weights=weights)


def test_context_derives_ledgers_and_freezes_inputs():
    fleet, result, checked, weights = context()
    weights[4] = 100
    checked["total_mass_kg"] = 999
    assert (
        fleet.replacement(1, FixedCargoRequest.from_summary(plan().summary())).objective_gain_kg
        == 8
    )
    assert fleet.raw_kg == 410
    with pytest.raises(ValueError, match="ledger differs"):
        FleetBudgetContext.from_verified_result(
            result, independent=checked, official=checked, weights=weights
        )
    with pytest.raises(TypeError):
        fleet._weights[4] = 3


def test_capture_storage_is_bounded_and_keeps_chosen_proxy_identity():
    fleet, _, _, _ = context()
    queue = RefinementAdmissionQueue()
    consume = queue.completion_consumer(fleet, 1)
    for body in range(4, 104):
        consume(envelope(plan(body), deficit=-1, salt=str(body)))
    assert len(queue.retained()) == len(queue.plan_lookup) == 48
    key = queue.retained()[0].request.sha256
    original = json.loads(queue.plan_lookup[key])
    better = b.RoutePlan.from_summary(original)
    newer = envelope(better, deficit=-1, salt="000")
    consume(newer)
    chosen = next(x for x in queue.retained() if x.request.sha256 == key)
    assert chosen.request == FixedCargoRequest.from_summary(json.loads(queue.plan_lookup[key]))
    with pytest.raises(TypeError):
        queue.plan_lookup[key] = b"corrupt"
    assert queue.claim_next() is not None
    with pytest.raises(RuntimeError, match="closed"):
        consume(newer)


def test_earlier_native_failures_and_changed_capture_identity_do_not_escalate():
    fleet, _, _, _ = context()
    queue = RefinementAdmissionQueue()
    consume = queue.completion_consumer(fleet, 1)
    observed = envelope(plan())
    observed.observation = dataclasses.replace(observed.observation, failure="leg_authority")
    assert consume(observed) == "earlier_completion_gate"
    observed = envelope(plan())
    observed.plan_summary = json.dumps(plan(cargo=8).summary()).encode()
    assert consume(observed) == "captured_prescription_mismatch"
    assert not queue.plan_lookup


def test_shortlist_keeps_depth_then_schedule_diversity_with_same_earth_seed():
    shallow = plan()
    timed = plan(departure=66010)
    deep = b.RoutePlan(
        (
            shallow.legs[0],
            b.PlannedLeg(4, 5, 65000, 65200, 1, 1, "deploy_hop"),
            b.PlannedLeg(5, 4, 66000, 66200, 1, 1, "collect_hop"),
            b.PlannedLeg(4, 0, 66200, 67000, 1, 1, "earth_return"),
        ),
        {4: 65000, 5: 65200},
        {5: 66000, 4: 66200},
        {4: 9, 5: 9},
        500,
        550,
    )
    candidates = [shallow, deepcopy(shallow), timed, deep]
    assert b.refine_candidates(candidates, set(), b.ClusterPricingSettings(refine_top=2)) == [
        shallow,
        deep,
    ]
    assert b.refine_candidates(candidates, set(), b.ClusterPricingSettings(refine_top=3)) == [
        shallow,
        deep,
        timed,
    ]
    assert b.refine_candidates(candidates, set(), b.ClusterPricingSettings(refine_top=0)) == []


@pytest.mark.parametrize("fleet_passes", [True, False])
def test_context_queue_reaches_fixed_executor_but_only_both_fleet_checks_can_promote(fleet_passes):
    from spacepdhcg.gtoc12.fixed_refinement import run_refinement_queue

    fleet, _, _, _ = context()
    queue = RefinementAdmissionQueue()
    observed = envelope(plan())
    assert queue.completion_consumer(fleet, 1)(observed) == "refinement_eligible"
    calls = []

    def refine(route, catalogue, cargo, **kwargs):
        calls.append(("refine", dict(cargo)))
        return refined(route)

    def verify(route, key):
        calls.append(("both_checkers", queue.replacement_ship(key)))
        assert (
            FixedCargoRequest.from_summary(route.plan.summary(), route.collected_mass).sha256 == key
        )
        digest = "c" * 64
        return {
            "request_sha256": key,
            "result_sha256": digest,
            "independent": {
                "ok": True,
                "result_sha256": digest,
                "ships": 3,
                "total_mass_kg": 409,
                "weighted_score_fixed_bonus_kg": 418,
            },
            "official": {"ok": fleet_passes, "result_sha256": digest},
        }

    journal = []
    outcomes = run_refinement_queue(
        queue,
        queue.plan_lookup,
        catalogue=None,
        settings=SimpleNamespace(outer_loop_backend="cuda"),
        verify_full_fleet=verify,
        incumbent_weighted_kg=fleet.objective_kg,
        deadline=time.perf_counter() + 2,
        on_result=journal.append,
        refine=refine,
    )
    assert calls == [("refine", {4: 9}), ("both_checkers", 1)]
    assert len(outcomes) == queue.claimed == len(journal) == 1
    assert outcomes[0]["accepted"] is fleet_passes
    assert fleet.raw_kg == 410  # Testing proposals never mutates the incumbent.


def install_pricing(monkeypatch, candidates, observed):
    earth = b.EarthLeg(4, 64328, 672, 1, 100)
    monkeypatch.setattr(b, "certify_earth_legs", lambda *a, **kw: ([earth], []))

    class Search:
        def __init__(self, *args, **kwargs):
            observed["search_args"] = kwargs
            self.collect_dp_stats = self.substitution_stats = self.chain_tour_stats = {}
            self.asteroid_prices = {}
            self.collect_dp_used = False

        def run(self):
            observed["search_calls"] = observed.get("search_calls", 0) + 1
            return SimpleNamespace(
                candidates=candidates, best_by_depth={}, wall_seconds=0, failures=[]
            )

        def release_caches(self):
            return {}

    monkeypatch.setattr(b, "RouteSearch", Search)

    def forbidden(*args, **kwargs):
        pytest.fail("fixed replacements must not enter stock shrinking/retiming/repair")

    for name in ("refine_route", "Retimer", "_repair_orphans", "improve_and_certify"):
        monkeypatch.setattr(b, name, forbidden)


def refined(route, *, certified=True):
    return SimpleNamespace(
        plan=route,
        collected_mass=dict(route.collected_mass),
        certified=certified,
        master_certified=certified,
        legs=[SimpleNamespace(certified=certified) for _ in route.legs],
        failures=[],
    )


def test_real_cluster_entry_uses_fixed_refiner_keeps_alternatives_and_budget(monkeypatch):
    from spacepdhcg.gtoc12 import fixed_refinement

    fleet, _, _, _ = context()
    candidates = [plan(), plan(departure=66010), plan(departure=66020)]
    observed = {}
    install_pricing(monkeypatch, candidates, observed)
    calls = []

    def fixed(route, catalogue, cargo, *, settings, deadline):
        calls.append((route, dict(cargo), settings, deadline))
        return refined(route)

    monkeypatch.setattr(fixed_refinement, "refine_fixed", fixed)
    settings = b.ClusterPricingSettings(ships=1, refine_top=2, search_retries=10)
    result = b.price_cluster(
        None,
        np.array([1, 2, 3, 4, 5]),
        settings=settings,
        scvx=SimpleNamespace(outer_loop_backend="cuda"),
        fleet_context=fleet,
        replacement_ship=1,
    )
    assert observed["search_calls"] == 1 and len(calls) == 2
    assert observed["search_args"]["excluded"] == {2, 3}
    assert len(result.ships) == 1 and len(result.ships[0].variants) == 2
    assert all(x[1] == {4: 9} for x in calls)
    assert (
        result.ships[0].report["replacement"]["scope"]
        == "certified_columns_require_full_fleet_verification"
    )
    assert [x.summary() for x in candidates] == [
        plan().summary(),
        plan(departure=66010).summary(),
        plan(departure=66020).summary(),
    ]


@pytest.mark.parametrize("failure", ["shrink", "retime", "exception", "partial_certificate"])
def test_cluster_fixed_attempt_failures_consume_slots_without_acceptance(monkeypatch, failure):
    fleet, _, _, _ = context()
    candidates = [plan(), plan(departure=66010), plan(departure=66020)]
    originals = [deepcopy(x.summary()) for x in candidates]
    observed = {}
    install_pricing(monkeypatch, candidates, observed)
    calls = []

    def refuse(route):
        calls.append(route)
        if failure == "exception":
            raise RuntimeError("native rejected fixture")
        if failure == "shrink":
            route.collected_mass[4] = 8
        if failure == "retime":
            route = dataclasses.replace(route, collect_epochs={4: 65999})
        result = refined(route)
        if failure == "partial_certificate":
            result.legs.pop()
        return result

    result = b.price_cluster(
        None,
        np.array([1, 2, 3, 4, 5]),
        settings=b.ClusterPricingSettings(ships=1, refine_top=2, search_retries=10),
        refine=refuse,
        fleet_context=fleet,
        replacement_ship=1,
    )
    assert len(calls) == 2 and observed["search_calls"] == 1
    assert len(result.rejected) == 2 and not result.ships
    assert [x.summary() for x in candidates] == originals
