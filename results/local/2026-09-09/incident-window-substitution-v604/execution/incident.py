"""Ranking diagnostics, never acceptance certificates, for the frozen v604 search."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

SHIFTS_DAYS = (-30.0, -15.0, 0.0, 15.0, 30.0)
PER_ARM_RETIMINGS = 12
MAX_POLISH_SEEDS = 4
MAX_POLISH_ROWS = 2820


def epoch_seeds(visits, arrivals, departures, new):
    """Same two-sided grid for both arms; protect all unrelated visit epochs."""
    seeds = []
    for deploy_shift in SHIFTS_DAYS:
        for collect_shift in SHIFTS_DAYS:
            arr, dep = np.array(arrivals, dtype=float), np.array(departures, dtype=float)
            for j, visit in enumerate(visits):
                if visit.body != new:
                    continue
                if visit.deploy:
                    arr[j] += deploy_shift
                    if not visit.collect:
                        dep[j] += deploy_shift
                if visit.collect:
                    dep[j] += collect_shift
                    if not visit.deploy:
                        arr[j] += collect_shift
            # Only structural invalidity is omitted here. TOF, dwell, authority,
            # minimum mining stay, mass and mission rules stay with frozen native code.
            if (
                np.all(arr[1:] > dep[:-1])
                and np.all(dep >= arr)
                and arr[0] >= 64328
                and arr[-1] <= 69807
            ):
                name = f"deploy_{deploy_shift:+g}_collect_{collect_shift:+g}"
                seeds.append((name, arr, dep))
    return seeds


def incident_indices(visits, new):
    """Both incoming and outgoing legs at both mining visits (camp deduplicated)."""
    actions = [v for v in visits if v.body == new]
    if sum(v.deploy for v in actions) != 1 or sum(v.collect for v in actions) != 1:
        raise ValueError("exactly one own deployment and collection required")
    changed = set()
    for j, visit in enumerate(visits):
        if visit.body == new:
            if j == 0 or j == len(visits) - 1:
                raise ValueError("Earth endpoints cannot be substituted")
            changed.update((j - 1, j))
    return sorted(changed)


def cached_leg_costs(joint, visits, arr, dep):
    """No hidden geometry work: missing native preflight rows remain unrankable."""
    costs = []
    for j in range(len(visits) - 1):
        key = joint.key(visits[j].body, visits[j + 1].body, dep[j], arr[j + 1])
        costs.append(joint._lambert.get(key, math.inf))
    return costs


def mass_projection(joint, visits, arr, dep, costs, quantities):
    """Uncertified full-route mass diagnostic, continuing beyond authority failures.

    This does not call or alter the authoritative forward evaluator. It retains
    its exact measured-key/mass-tolerance and calibrated inflation formulas, but
    does not truncate at an authority violation or trim mining quantities.
    """
    from spacepdhcg.gtoc12 import constants as C
    from spacepdhcg.gtoc12.screening import propellant_for_delta_v

    initial = float(joint.retimer.search_settings.initial_mass)
    mass, spent = initial, 0.0
    for j, visit in enumerate(visits[:-1]):
        if visit.collect:
            mass += quantities[visit.body]
        nxt = visits[j + 1]
        tof = float(arr[j + 1] - dep[j])
        cost = float(costs[j])
        if tof <= 0 or not math.isfinite(cost) or not math.isfinite(mass) or mass <= 0:
            return {"complete": False, "propellant_kg": math.inf, "spare_kg": -math.inf}
        key = joint.key(visit.body, nxt.body, dep[j], arr[j + 1])
        measured = joint.measured.get(key)
        if (
            measured is not None
            and abs(measured.mass_before_kg - mass) <= joint.settings.measured_mass_tolerance_kg
        ):
            effective = measured.delta_v_km_s
        else:
            inflation = float(
                joint.retimer.leg_inflation(visit.role_out, visit.body, nxt.body, cost, mass, tof)
            )
            effective = cost * inflation
        propellant = float(propellant_for_delta_v(mass, effective))
        if not math.isfinite(propellant) or propellant < 0:
            return {"complete": False, "propellant_kg": math.inf, "spare_kg": -math.inf}
        spent += propellant
        mass -= propellant
        if nxt.deploy:
            mass -= C.MINER_MASS_KG
    capacity = initial - C.MINER_MASS_KG * sum(v.deploy for v in visits) - C.DRY_MASS_KG
    if capacity <= 0:
        raise ValueError("no propellant capacity in incumbent inventory")
    return {
        "complete": True,
        "propellant_kg": spent,
        "propellant_capacity_kg": capacity,
        "propellant_utilization": spent / capacity,
        "spare_kg": mass - C.DRY_MASS_KG - sum(quantities.values()),
    }


def ranking_diagnostic(joint, route, visits, arr, dep, case, weights, row, fleet_raw, raw_floor):
    """Price every changed incident leg at reference mass, regardless of first failure."""
    from spacepdhcg.gtoc12 import constants as C
    from spacepdhcg.gtoc12.retiming import Retimer

    deploy = {v.body: float(arr[j]) for j, v in enumerate(visits) if v.deploy}
    collect = {v.body: float(dep[j]) for j, v in enumerate(visits) if v.collect}
    quantities = {a: C.maximum_collected_mass(collect[a] - deploy[a]) for a in collect}
    if any(a not in weights or not math.isfinite(weights[a]) for a in quantities):
        raise ValueError("complete finite fixed-bonus weights are mandatory")
    raw = sum(quantities.values())
    weighted = sum(weights[a] * q for a, q in quantities.items())
    costs = cached_leg_costs(joint, visits, arr, dep)
    changed = set(incident_indices(visits, case["new"]))
    rows, collected_delta = [], 0.0
    if len(route.legs) != len(costs):
        raise ValueError("incumbent and candidate leg topology lengths differ")
    for j, visit in enumerate(visits[:-1]):
        if visit.collect:
            original_body = case["old"] if visit.body == case["new"] else visit.body
            collected_delta += quantities[visit.body] - route.collected_mass[original_body]
        if j not in changed:
            continue
        nxt = visits[j + 1]
        tof = float(arr[j + 1] - dep[j])
        mass = route.legs[j].mass_before + collected_delta
        _, limit = joint.retimer._limits(visit.role_out, visit.body, nxt.body)
        ratio = (
            float(Retimer.authority_ratio(costs[j], mass, tof))
            if tof > 0 and mass > 0 and math.isfinite(costs[j])
            else math.inf
        )
        rows.append(
            {
                "leg": j,
                "from": visit.body,
                "to": nxt.body,
                "departure": float(dep[j]),
                "arrival": float(arr[j + 1]),
                "tof_days": tof,
                "lambert_km_s": float(costs[j]),
                "reference_mass_kg": mass,
                "authority_ratio_estimate": ratio,
                "unchanged_authority_limit": limit,
                "authority_normalized_estimate": ratio / limit,
            }
        )
    projection = mass_projection(joint, visits, arr, dep, costs, quantities)
    worst = max(r["authority_normalized_estimate"] for r in rows)
    complete = bool(
        projection["complete"]
        and len(rows) == len(changed)
        and all(math.isfinite(r["authority_normalized_estimate"]) for r in rows)
    )
    return {
        "scope": "uncertified_ranking_estimates_only",
        "complete": complete,
        "incident_legs": rows,
        "worst_incident_authority_normalized_estimate": worst,
        "mass_projection": projection,
        "resource_bottleneck_estimate": max(
            worst, projection.get("propellant_utilization", math.inf)
        ),
        "near_authority_threshold_estimate": math.isfinite(worst) and 1.0 < worst <= 1.10,
        "mining_raw_kg_estimate": raw,
        "mining_weighted_kg_estimate": weighted,
        "mining_weighted_gain_kg_estimate": weighted - row["verified_weighted_kg"],
        "fleet_raw_shortfall_kg_estimate": max(
            0.0, raw_floor - (fleet_raw - row["verified_raw_kg"] + raw)
        ),
    }


def diagnostic_key(diagnostic, case_id="", mode=""):
    """No hard surrogate rejection: near-threshold windows can receive retiming."""
    return (
        not diagnostic["complete"],
        diagnostic["fleet_raw_shortfall_kg_estimate"] > 0,
        diagnostic["resource_bottleneck_estimate"],
        diagnostic["fleet_raw_shortfall_kg_estimate"],
        -diagnostic["mining_weighted_gain_kg_estimate"],
        case_id,
        mode,
    )


def choose_arms(cases, diagnostics, reference_ids):
    """Twelve historical control choices versus twelve complete-incidence choices.

    Both read the same grid. Identical per-slot diversity and retiming budgets
    isolate shortlist policy; expanded-grid gains are reported separately.
    """
    by_id = {c["case"]: c for c in cases}
    if len(by_id) != len(cases) or len(reference_ids) != PER_ARM_RETIMINGS:
        raise ValueError("control requires exactly twelve unique historical cases")
    control = [by_id[key] for key in reference_ids]
    slots = [(c["ship"], c["old"]) for c in control]
    if len(set(slots)) != PER_ARM_RETIMINGS:
        raise ValueError("control must cover twelve unique old-body slots")
    incident = []
    for ship, old in slots:
        available = [
            c
            for c in cases
            if (c["ship"], c["old"]) == (ship, old)
            and c["case"] in diagnostics
            and diagnostics[c["case"]]["complete"]
        ]
        if available:
            incident.append(
                min(available, key=lambda c: diagnostic_key(diagnostics[c["case"]], c["case"]))
            )
    # If no complete window exists, retain a historical fallback rather than
    # inventing a ratio from a truncated/incomplete geometry evaluation.
    filled = {(c["ship"], c["old"]) for c in incident}
    incident += [c for c in control if (c["ship"], c["old"]) not in filled]
    incident.sort(key=lambda c: slots.index((c["ship"], c["old"])))
    if Counter((c["ship"], c["old"]) for c in control) != Counter(
        (c["ship"], c["old"]) for c in incident
    ):
        raise ValueError("comparison arm slot inventories differ")
    return {"historical_priority": control, "complete_incidence": incident}


def screen_progress(records, cases, planned_rows):
    """Report shared grid work and never mistake an attempted/skipped case for coverage."""
    samples = [sample for record in records for sample in record["samples"]]
    ids = [record["case"] for record in records]
    return {
        "planned_rows": planned_rows,
        "completed_rows": len(samples),
        "screened_cases": sum(record["status"] == "screened" for record in records),
        "complete": (
            len(ids) == len(set(ids)) == len(cases)
            and set(ids) == {case["case"] for case in cases}
            and all(record["status"] == "screened" for record in records)
            and len(samples) == planned_rows
        ),
        "native_forward_feasible_rows": sum(s["feasible_surrogate"] for s in samples),
        "objective_eligible_native_forward_rows": sum(
            s.get("proxy", {}).get("eligible_for_refinement", False) for s in samples
        ),
        "complete_ranking_estimate_rows": sum(s["ranking_estimate"]["complete"] for s in samples),
        "near_threshold_ranking_estimate_rows": sum(
            s["ranking_estimate"]["near_authority_threshold_estimate"] for s in samples
        ),
        "native_forward_failure_counts": dict(Counter(s["failure"] for s in samples)),
    }
