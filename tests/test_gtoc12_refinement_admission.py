"""A fuel estimate may buy a solver slot, never a certificate or a larger budget."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest

from spacepdhcg.gtoc12.refinement_admission import (
    AdmissionCandidate,
    AdmissionPolicy,
    CompletionObservation,
    FixedCargoRequest,
    RefinementAdmissionQueue,
    promotion_blockers,
)


def prescription(body=101, cargo=10.0):
    return {
        "legs": [
            {"from": 0, "to": body, "t0": 64328.0, "tf": 65000.0, "role": "earth_out"},
            {"from": body, "to": 0, "t0": 66000.0, "tf": 66500.0, "role": "earth_return"},
        ],
        "deploy_epochs": {body: 65000.0},
        "collect_epochs": {body: 66000.0},
        "collected_mass_kg": {body: cargo},
    }


def candidate(body=101, deficit=1.0, gain=3.0):
    request = FixedCargoRequest.from_summary(prescription(body))
    readback = hashlib.sha256(f"{body}/{deficit}".encode()).hexdigest()
    observation = CompletionObservation(
        request.sha256,
        "a" * 64,
        readback,
        "mass_below_dry_plus_collected" if deficit > 0 else "",
        1,
        1,
        (4,),
        510.0 - deficit,
        10.0,
        True,
    )
    return AdmissionCandidate(request, observation, gain, 15000.0, 23)


def test_uncertainty_buys_two_finite_slots_and_never_means_acceptance():
    queue = RefinementAdmissionQueue()
    rows = [candidate(i, deficit=float(i), gain=100.0 - i) for i in range(1, 7)]
    for row in reversed(rows):
        assert queue.offer(row) == "refinement_eligible"
    assert queue.shortlist() == tuple(rows[:2])
    assert queue.claim_next() == rows[0]
    # A failed solve still consumes a slot. There is no success-dependent refund.
    assert queue.claim_next() == rows[1]
    assert queue.claim_next() is None
    assert queue.claimed == 2
    assert all(x.observation.failure for x in queue.shortlist())
    with pytest.raises(RuntimeError, match="closed"):
        queue.offer(candidate(9))


def test_mixed_queue_reserves_uncertain_slots_without_exceeding_total_budget():
    regular = [candidate(i, deficit=-1.0, gain=float(i)) for i in range(1, 51)]
    uncertain = [candidate(100 + i, deficit=float(i)) for i in range(1, 4)]
    queue = RefinementAdmissionQueue()
    for row in uncertain + regular:
        queue.offer(row)
    selected = queue.shortlist()
    assert selected == (regular[-1], regular[-2], *uncertain[:2])
    assert len(queue._pending) == 48
    assert sum(x.uncertain for x in selected) == 2
    assert [queue.claim_next() for _ in range(4)] == list(selected)
    assert queue.claim_next() is None


def test_weighted_gain_breaks_equal_deficit_ties_but_cannot_hide_large_deficit():
    rows = [candidate(1, 1, 1), candidate(2, 1, 9), candidate(3, 50, 1000)]
    queue = RefinementAdmissionQueue()
    for row in rows:
        queue.offer(row)
    assert queue.shortlist() == (rows[1], rows[0])


def test_exact_prescription_is_deduplicated_across_proxy_models_independent_of_order():
    worse, better = candidate(1, 20), candidate(1, 5)
    selected = []
    for rows in ((worse, better), (better, worse)):
        queue = RefinementAdmissionQueue()
        assert queue.offer(rows[0]) == "refinement_eligible"
        assert queue.offer(rows[1]) == "duplicate_prescription_other_proxy"
        selected.append(queue.shortlist())
    assert selected == [(better,), (better,)]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("failure", "leg_authority", "earlier_completion_gate"),
        ("failure", "stay_too_short", "earlier_completion_gate"),
        ("failure", "uncollected", "earlier_completion_gate"),
        ("failure", "invalid_inflation", "earlier_completion_gate"),
        ("failure", "invalid_mining_stay", "earlier_completion_gate"),
        ("processed_legs", 0, "incomplete_forward_pass"),
        ("stages", (2,), "incomplete_forward_pass"),
        ("stages", (0,), "incomplete_forward_pass"),
        ("final_mass_kg", float("nan"), "invalid_completion_readback"),
        ("finite_nonnegative_costs", False, "invalid_completion_readback"),
        ("collected_kg", 9.0, "invalid_completion_readback"),
        ("request_sha256", "b" * 64, "request_readback_mismatch"),
        ("readback_sha256", "", "missing_readback_provenance"),
        ("failure", "", "inconsistent_mass_classification"),
    ],
)
def test_other_failures_or_unbound_readbacks_never_get_uncertain_slots(field, value, reason):
    row = candidate()
    row = dataclasses.replace(
        row, observation=dataclasses.replace(row.observation, **{field: value})
    )
    queue = RefinementAdmissionQueue()
    assert queue.offer(row) == reason
    assert queue.shortlist() == ()


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan"), float("inf")])
def test_live_admission_requires_actual_weighted_improvement_estimate(gain):
    queue = RefinementAdmissionQueue()
    assert queue.offer(candidate(gain=gain)) != "refinement_eligible"


def test_archival_equal_score_control_can_be_ranked_but_cannot_launch():
    queue = RefinementAdmissionQueue(replay_controls=True)
    assert queue.offer(candidate(gain=0)) == "refinement_eligible"
    assert len(queue.shortlist()) == 1
    with pytest.raises(RuntimeError, match="cannot launch"):
        queue.claim_next()


def test_raw_fleet_rule_is_separate_from_weighted_priority():
    row = dataclasses.replace(candidate(gain=10000), proposed_fleet_raw_kg=1000)
    assert RefinementAdmissionQueue().offer(row) == "raw_fleet_ship_limit"


def test_payload_cannot_be_mutated_through_original_plan_or_a_decoded_copy():
    plan = prescription()
    saved = copy.deepcopy(plan)
    request = FixedCargoRequest.from_summary(plan)
    digest = request.sha256
    plan["collected_mass_kg"][101] = 9.0
    decoded = json.loads(request.payload)
    decoded["cargo"][0][1] = 8.0
    assert request.sha256 == digest
    assert request == FixedCargoRequest.from_summary(saved)
    assert request != FixedCargoRequest.from_summary(plan)


@pytest.mark.parametrize("change", ["overmined", "short_stay", "wrong_event", "foreign", "window"])
def test_invalid_schedule_or_inventory_cannot_be_packaged_for_queue(change):
    plan = prescription()
    if change == "overmined":
        plan["collected_mass_kg"][101] = 1000
    elif change == "short_stay":
        plan["collect_epochs"][101] = 65100
    elif change == "wrong_event":
        plan["collect_epochs"][101] = 66100
    elif change == "foreign":
        plan["foreign_deploy_epochs"] = {102: 65000}
    else:
        plan["legs"][-1]["tf"] = 70000
    with pytest.raises(ValueError):
        FixedCargoRequest.from_summary(plan)


def test_guard_requires_actual_matching_result_and_both_checkers_and_exact_cargo():
    request = candidate().request
    actual = "f" * 64
    independent = {
        "ok": True,
        "result_sha256": actual,
        "weighted_score_fixed_bonus_kg": 12811.0,
        "total_mass_kg": 15000.0,
        "ships": 23,
    }
    official = {"ok": True, "result_sha256": actual}
    common = dict(
        all_legs_certified=True,
        result_sha256=actual,
        independent=independent,
        official=official,
        incumbent_weighted_kg=12810.0,
    )
    assert promotion_blockers(request, request, **common) == ()
    shrunk = FixedCargoRequest.from_summary(prescription(cargo=9.0))
    assert "prescribed_events_or_cargo_changed" in promotion_blockers(request, shrunk, **common)
    delayed = prescription()
    delayed["legs"][-1]["tf"] += 1.0
    changed = FixedCargoRequest.from_summary(delayed)
    assert "prescribed_events_or_cargo_changed" in promotion_blockers(request, changed, **common)
    for field, value, expected in (
        ("official", {**official, "ok": False}, "official_rejected"),
        ("independent", {**independent, "ok": False}, "independent_rejected"),
        ("official", {**official, "result_sha256": "a" * 64}, "official_result_mismatch"),
        ("all_legs_certified", False, "uncertified_route"),
        (
            "independent",
            {**independent, "weighted_score_fixed_bonus_kg": None},
            "no_verified_weighted_improvement",
        ),
        ("independent", {**independent, "total_mass_kg": 1000.0}, "verified_raw_fleet_ship_limit"),
    ):
        assert expected in promotion_blockers(request, request, **{**common, field: value})


@pytest.mark.parametrize("budgets", [(49, 4, 2), (48, 5, 2), (48, 4, 3), (2, 4, 2)])
def test_unreviewed_budget_expansion_is_rejected(budgets):
    with pytest.raises(ValueError):
        AdmissionPolicy(*budgets)
