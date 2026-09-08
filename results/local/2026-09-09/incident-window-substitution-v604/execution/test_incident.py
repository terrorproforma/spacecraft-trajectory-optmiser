"""Behavioral construction/ranking checks; CUDA loading is forbidden."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import incident
import numpy as np
import pytest
from test_driver import (
    HERE,
    camp,  # noqa: F401
    driver,
    frozen_inventory,  # noqa: F401
    no_native_execution,  # noqa: F401
    paired,  # noqa: F401
)


@pytest.mark.parametrize("layout,count", [("paired", 4), ("camp", 2)])
def test_both_ends_of_both_mining_visits_are_included(request, layout, count):
    visits, _, _ = request.getfixturevalue(layout)
    changed = driver.replacement_visits(visits, 2, 99, set())
    indices = incident.incident_indices(changed, 99)
    assert len(indices) == count
    for j in indices:
        assert 99 in (changed[j].body, changed[j + 1].body)
    assert indices == [
        j for j in range(len(changed) - 1) if 99 in (changed[j].body, changed[j + 1].body)
    ]


@pytest.mark.parametrize("field", ["deploy", "collect"])
def test_missing_mining_action_cannot_be_a_complete_incident_case(request, field):
    layout = request.getfixturevalue("paired")
    visits = [replace(v, **{field: False}) if v.body == 2 else v for v in layout[0]]
    with pytest.raises(ValueError, match="exactly one"):
        incident.incident_indices(visits, 2)


def test_no_cpu_geometry_fallback_when_gpu_preflight_skips_a_row(request):
    visits, arr, dep = request.getfixturevalue("paired")
    joint = SimpleNamespace(_lambert={}, key=lambda a, b, d, e: (a, b, d, e))
    costs = incident.cached_leg_costs(joint, visits, arr, dep)
    assert len(costs) == len(visits) - 1
    assert all(math.isinf(value) for value in costs)


@pytest.fixture
def projection_fixture(request):
    summaries, audit, weights, _ = request.getfixturevalue("frozen_inventory")
    driver.activate_source()
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
    )
    from spacepdhcg.gtoc12.data import load_catalogue
    from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
    from spacepdhcg.gtoc12.retiming import Retimer, visits_of

    catalogue = load_catalogue()
    route = route_from_summary(summaries[2])
    policy = ClusterPricingSettings(
        collect_dp_inflation_fit=str(HERE / "source/results/gtoc12/hop_inflation_fit.json")
    )
    retimer = Retimer(
        catalogue,
        cluster_search_settings(policy, len(catalogue.ids)),
        cluster_retime_settings(policy, last=True),
        weights=weights,
    )
    retimer.protect_earth_leg(route.plan)
    joint = JointItinerary(catalogue, retimer, weights=weights)
    joint.learn(route)
    visits, arr, dep = visits_of(route.plan)
    case = {"old": 41045, "new": 34568, "ship": 2, "case": "replacement"}
    changed = driver.replacement_visits(visits, case["old"], case["new"], set())
    for j, leg in enumerate(route.legs):
        joint._lambert[joint.key(changed[j].body, changed[j + 1].body, dep[j], arr[j + 1])] = (
            leg.planned.delta_v_proxy_km_s
        )
    row = {
        "verified_raw_kg": route.total_collected_kg,
        "verified_weighted_kg": sum(weights[a] * m for a, m in route.collected_mass.items()),
    }
    return joint, route, changed, np.asarray(arr), np.asarray(dep), case, weights, row, audit


def diagnostic(fixture):
    joint, route, visits, arr, dep, case, weights, row, audit = fixture
    return incident.ranking_diagnostic(
        joint,
        route,
        visits,
        arr,
        dep,
        case,
        weights,
        row,
        audit["independent"]["total_mass_kg"],
        driver.MIN_FLEET_RAW_KG,
    )


def test_bad_last_incident_leg_is_seen_even_when_first_leg_would_reject(projection_fixture):
    fixture = projection_fixture
    joint, _, visits, arr, dep, case, *_ = fixture
    changed = incident.incident_indices(visits, case["new"])
    original = diagnostic(fixture)
    first, last = changed[0], changed[-1]
    first_key = joint.key(visits[first].body, visits[first + 1].body, dep[first], arr[first + 1])
    last_key = joint.key(visits[last].body, visits[last + 1].body, dep[last], arr[last + 1])
    joint._lambert[first_key] *= 2
    early_failure = diagnostic(fixture)
    joint._lambert[last_key] *= 20
    late_failure = diagnostic(fixture)
    assert [r["leg"] for r in late_failure["incident_legs"]] == changed
    assert (
        late_failure["incident_legs"][0]["authority_normalized_estimate"]
        == early_failure["incident_legs"][0]["authority_normalized_estimate"]
    )
    assert (
        late_failure["incident_legs"][-1]["authority_normalized_estimate"]
        > 10 * original["incident_legs"][-1]["authority_normalized_estimate"]
    )
    assert (
        late_failure["resource_bottleneck_estimate"] > early_failure["resource_bottleneck_estimate"]
    )
    assert incident.diagnostic_key(early_failure) < incident.diagnostic_key(late_failure)


def test_downstream_unchanged_leg_cost_affects_mass_bottleneck(projection_fixture):
    fixture = projection_fixture
    joint, _, visits, arr, dep, case, *_ = fixture
    changed = incident.incident_indices(visits, case["new"])
    before = diagnostic(fixture)
    downstream = len(visits) - 2
    assert downstream not in changed
    key = joint.key(
        visits[downstream].body, visits[downstream + 1].body, dep[downstream], arr[downstream + 1]
    )
    # Remove the measured shortcut solely in this synthetic fixture, then price
    # an expensive unchanged return; production measured data is never altered.
    joint.measured.pop(key)
    joint._lambert[key] = 30.0
    after = diagnostic(fixture)
    assert after["incident_legs"] == before["incident_legs"]
    assert after["mass_projection"]["propellant_kg"] > before["mass_projection"]["propellant_kg"]
    assert after["mass_projection"]["spare_kg"] < before["mass_projection"]["spare_kg"]


def test_changed_body_never_inherits_old_measured_leg(projection_fixture):
    joint, route, visits, arr, dep, case, *_ = projection_fixture
    for j in incident.incident_indices(visits, case["new"]):
        new_key = joint.key(visits[j].body, visits[j + 1].body, dep[j], arr[j + 1])
        old = route.legs[j].planned
        old_key = joint.key(old.from_id, old.to_id, old.departure_epoch, old.arrival_epoch)
        assert new_key not in joint.measured
        assert old_key in joint.measured


def test_changed_collect_epoch_accounts_for_cargo_in_later_incident_mass(projection_fixture):
    fixture = projection_fixture
    joint, route, visits, arr, dep, case, *_ = fixture
    before = diagnostic(fixture)
    seeds = dict(
        (mode, (a, d)) for mode, a, d in incident.epoch_seeds(visits, arr, dep, case["new"])
    )
    later_arr, later_dep = seeds["deploy_+0_collect_+30"]
    for j, cost in enumerate(incident.cached_leg_costs(joint, visits, arr, dep)):
        joint._lambert[
            joint.key(visits[j].body, visits[j + 1].body, later_dep[j], later_arr[j + 1])
        ] = cost
    after = diagnostic((joint, route, visits, later_arr, later_dep, *fixture[5:]))
    assert after["incident_legs"][-1]["reference_mass_kg"] - before["incident_legs"][-1][
        "reference_mass_kg"
    ] == pytest.approx(30 * 10 / 365.25)
    assert after["mining_weighted_gain_kg_estimate"] > before["mining_weighted_gain_kg_estimate"]


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_nonfinite_new_bonus_is_an_error(projection_fixture, missing):
    fixture = list(projection_fixture)
    fixture[6] = dict(fixture[6])
    new = fixture[5]["new"]
    if missing:
        del fixture[6][new]
    else:
        fixture[6][new] = math.nan
    with pytest.raises(ValueError, match="fixed-bonus weights"):
        diagnostic(fixture)


def synthetic_diagnostic(bottleneck, *, complete=True, gain=5, shortfall=0):
    return {
        "complete": complete,
        "resource_bottleneck_estimate": bottleneck,
        "fleet_raw_shortfall_kg_estimate": shortfall,
        "mining_weighted_gain_kg_estimate": gain,
    }


def test_equal_arm_budgets_select_near_threshold_ahead_of_large_bonus():
    cases, reference, diagnostics = [], [], {}
    for slot in range(12):
        control = {"ship": slot // 4 + 1, "old": slot + 1, "case": f"control_{slot}"}
        changed = control | {"case": f"incident_{slot}"}
        cases.extend((control, changed))
        reference.append(control["case"])
        diagnostics[control["case"]] = synthetic_diagnostic(2.1, gain=100)
        diagnostics[changed["case"]] = synthetic_diagnostic(1.04, gain=1)
    arms = incident.choose_arms(cases, diagnostics, reference)
    assert list(arms) == ["historical_priority", "complete_incidence"]
    assert [c["case"] for c in arms["historical_priority"]] == reference
    assert [c["case"] for c in arms["complete_incidence"]] == [f"incident_{i}" for i in range(12)]
    assert len(arms["historical_priority"]) == len(arms["complete_incidence"]) == 12
    assert arms == incident.choose_arms(list(reversed(cases)), diagnostics, reference)


def test_incomplete_or_resource_depleted_diagnostic_cannot_outrank_complete_windows():
    good = synthetic_diagnostic(1.03)
    assert incident.diagnostic_key(good) < incident.diagnostic_key(
        synthetic_diagnostic(0.8, complete=False)
    )
    assert incident.diagnostic_key(good) < incident.diagnostic_key(synthetic_diagnostic(1.3))
    assert incident.diagnostic_key(good) < incident.diagnostic_key(
        synthetic_diagnostic(0.8, shortfall=10)
    )


def test_exact_control_ids_and_case_inventory_are_preserved():
    plan = json.loads((HERE / "plan/plan.json").read_text())
    reference = json.loads((HERE / "reference/v599-plan.json").read_text())
    report = json.loads((HERE / "reference/v599-report.json").read_text())
    assert plan["replacement_cases"] == reference["replacement_cases"]
    assert plan["control_case_ids"] == [r["case"] for r in report["retimings"]]
    assert plan["maximum_retime_calls"] == 24
    assert plan["maximum_full_refinements"] == 4
    assert plan["maximum_initial_epoch_candidates"] <= 12400
    assert all(len(r["incident_legs"]) == 4 for r in plan["construction"])


@pytest.mark.parametrize("omission", ["skipped", "partial", "duplicate"])
def test_comparison_never_claims_complete_grid_when_coverage_is_missing(omission):
    sample = {
        "feasible_surrogate": False,
        "failure": "leg_authority",
        "ranking_estimate": {"complete": True, "near_authority_threshold_estimate": True},
    }
    cases = [{"case": "one"}, {"case": "two"}]
    records = [
        {"case": c["case"], "status": "screened", "samples": [sample, sample]} for c in cases
    ]
    complete = incident.screen_progress(records, cases, 4)
    assert complete["complete"] and complete["completed_rows"] == 4
    if omission == "skipped":
        records[1].update(status="outside_neighbor_union", samples=[])
    elif omission == "partial":
        records[1]["samples"] = [sample]
    else:
        records[1]["case"] = "one"
    partial = incident.screen_progress(records, cases, 4)
    assert not partial["complete"]
    assert partial["native_forward_feasible_rows"] == 0
    assert partial["objective_eligible_native_forward_rows"] == 0
