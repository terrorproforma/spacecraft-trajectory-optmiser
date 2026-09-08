"""Frozen inputs, finite construction and honest proxy-only pool reporting."""

import dataclasses
import hashlib
import json
import math
from collections import Counter, defaultdict

import common

LIMITS = {
    "generation_calls": 1,
    "earth_seeds": 1,
    "family_members": 62,
    "maximum_depth": 10,
    "beam_width": 24,
    "expansions": 1 + 8 * 24,
    "completion_attempts": 1 + 9 * 24,
    "chain_tour_calls": 8 * 24,
    "collection_dp_passes": 8 * 24 + 9 * 24 * 4,
    "native_batch_capacity": 16384,
    "full_route_refinements": 0,
    "native_low_thrust_solves": 0,
    "fleet_promotions": 0,
    "wall_seconds": 120,
    "supervisor_allowance_seconds": 30,
    "termination_grace_seconds": 10,
}


def settings(wall_seconds=120):
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, cluster_search_settings

    if not 0 < wall_seconds <= LIMITS["wall_seconds"]:
        raise ValueError("Wall budget must be positive and no more than 120 seconds")
    pricing = ClusterPricingSettings(
        ships=1,
        beam_width=24,
        max_deploys=10,
        neighbours=61,
        chain_tour_scoring=True,
        chain_tour_candidates=24,
        harvest_substitution=False,
        collect_dp=True,
        collect_dp_inflation_fit="",
        refine_top=2,
        seed=0,
    )
    # This generation has just one Earth leg. Allow its entire finite 24-wide beam,
    # while retaining two variants per deployed set and every numerical mass gate.
    return dataclasses.replace(
        cluster_search_settings(pricing, 62),
        max_per_first=24,
        time_budget_seconds=wall_seconds,
    )


def load_inputs():
    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.search import EarthLeg
    from spacepdhcg.gtoc12.solution import Solution

    inputs = common.read(common.ROOT / "generation-input.json")
    catalogue, bonus = load_catalogue(), load_bonus_table()
    if catalogue.source_sha256 != inputs["catalogue_sha256"]:
        raise ValueError("Catalogue pin changed")
    if bonus.source_sha256 != inputs["bonus_sha256"]:
        raise ValueError("Bonus pin changed")
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    fleet = Solution.read(common.ROOT / "inputs/Result.txt")
    totals = []
    for ship in fleet.ships:
        collects = [
            (e.event_id, e.after.mass - e.before.mass)
            for e in ship.asteroid_visits()
            if e.after.mass > e.before.mass
        ]
        totals.append(
            {
                "ship": ship.ship_id,
                "raw_kg": sum(m for _, m in collects),
                "weighted_kg": sum(weights[a] * m for a, m in collects),
            }
        )
    original = next(row for row in totals if row["ship"] == inputs["ship"])
    raw = sum(row["raw_kg"] for row in totals)
    weighted = sum(row["weighted_kg"] for row in totals)
    if not math.isclose(raw, 14051.854893908598, abs_tol=1e-7, rel_tol=0):
        raise ValueError("Retained raw mass changed")
    if not math.isclose(weighted, 12810.135953048577, abs_tol=1e-7, rel_tol=0):
        raise ValueError("Retained weighted mass changed")
    excluded = {
        e.event_id
        for ship in fleet.ships
        if ship.ship_id != inputs["ship"]
        for e in ship.asteroid_visits()
    }
    allowed = set(inputs["allowed_ids"])
    if excluded != set(inputs["excluded_other_fleet_ids"]) or excluded & allowed:
        raise ValueError("Other-fleet footprint changed")
    if len(allowed) != 62 or allowed != set(inputs["family_ids"]):
        raise ValueError("Fixed 62-member family changed")
    seed = EarthLeg(**inputs["seed"])
    archive = common.read(common.ROOT / "inputs/ship-23.json")
    leg = archive["plan"]["legs"][0]
    measured = archive["legs"][0]
    if not archive["certified"] or not measured["certified"] or not seed.certified:
        raise ValueError("Archived Earth seed is not marked certified")
    if (seed.target, seed.launch_epoch, seed.tof_days, seed.propellant_kg) != (
        leg["to"],
        leg["t0"],
        leg["tf"] - leg["t0"],
        measured["propellant_kg"],
    ):
        raise ValueError("Seed differs from the saved measured Earth leg")
    ship = next(ship for ship in fleet.ships if ship.ship_id == inputs["ship"])
    first = ship.asteroid_visits()[0]
    if (ship.launch.epoch, first.epoch, first.event_id) != (
        seed.launch_epoch,
        seed.arrival_epoch,
        seed.target,
    ):
        raise ValueError("Earth seed differs from retained fleet events")
    baseline = {
        "raw_kg": raw,
        "weighted_kg": weighted,
        "ship_count": fleet.ship_count,
        "raw_floor_kg": fleet.ship_count * math.log(fleet.ship_count / 2) / 0.004,
        "replaced_ship": original,
        "score_scope": "Mass-event inventory of previously certified retained Result",
        "fullfleet_verification_recomputed_here": False,
    }
    return catalogue, weights, seed, sorted(allowed), excluded, baseline


def row_for(plan, index, weights, baseline, beam_score):
    summary = plan.summary()
    canonical = json.dumps(summary, sort_keys=True, allow_nan=False).encode()
    raw = float(plan.total_collected_kg)
    weighted = sum(weights[a] * mass for a, mass in plan.collected_mass.items())
    raw_fleet = baseline["raw_kg"] - baseline["replaced_ship"]["raw_kg"] + raw
    weighted_fleet = baseline["weighted_kg"] - baseline["replaced_ship"]["weighted_kg"] + weighted
    first = plan.legs[0]
    return {
        "index": index,
        "plan_sha256": hashlib.sha256(canonical).hexdigest(),
        "plan": summary,
        "earth_key": [first.to_id, first.departure_epoch, first.tof_days],
        "deploy_depth": len(plan.deploy_epochs),
        "collect_depth": len(plan.collect_epochs),
        "raw_proxy_kg": raw,
        "weighted_proxy_kg": weighted,
        "beam_score_including_propellant_penalty": beam_score,
        "candidate_fleet_raw_proxy_kg": raw_fleet,
        "candidate_fleet_weighted_proxy_kg": weighted_fleet,
        "proxy_raw_floor_eligible": math.isfinite(raw_fleet)
        and raw_fleet >= baseline["raw_floor_kg"],
        "proxy_weighted_improves": math.isfinite(weighted_fleet)
        and weighted_fleet > baseline["weighted_kg"] + 1e-9,
        "certified": False,
        "scored_fleet": False,
        "promoted": False,
        "scope": (
            "Generated impulsive/inflated proxy; requires fixed-cargo low-thrust refinement "
            "and both full-fleet checkers"
        ),
    }


def attrition(plans, certified_keys, rows):
    """Run the actual frozen selector; classify real pool rows without counterfactual gains."""
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, refine_candidates

    selected = refine_candidates(plans, certified_keys, ClusterPricingSettings(refine_top=2))
    selected_indices = [next(i for i, p in enumerate(plans) if p is value) for value in selected]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["earth_key"])].append(row["index"])
    firsts = {indices[0] for indices in groups.values()}
    return {
        "policy": "Frozen default one candidate per Earth leg, then refine_top=2",
        "actual_selected_indices": selected_indices,
        "generated_candidates": len(plans),
        "generated_depth_counts": dict(Counter(row["deploy_depth"] for row in rows)),
        "generated_closed_depth_counts": dict(
            Counter(row["deploy_depth"] for row in rows if row["plan"]["self_cleaning"])
        ),
        "earth_dedup_discarded_indices": [i for i in range(len(rows)) if i not in firsts],
        "refine_top_discarded_indices": sorted(firsts - set(selected_indices)),
        "same_earth_depth_groups": [
            {
                "earth_key": list(key),
                "indices": indices,
                "depths": sorted({rows[i]["deploy_depth"] for i in indices}),
            }
            for key, indices in groups.items()
            if len({rows[i]["deploy_depth"] for i in indices}) > 1
        ],
        "closed_raw_and_weighted_proxy_eligible_indices": [
            row["index"]
            for row in rows
            if row["plan"]["self_cleaning"]
            and row["proxy_raw_floor_eligible"]
            and row["proxy_weighted_improves"]
        ],
        "useful_discarded_certified_route_claim": False,
        "full_route_refinements": 0,
        "policy_comparison_ready": False,
        "next_gate": (
            "Review generated depth contrast and proxy fleet eligibility before preparing "
            "any matched-budget refinement arms"
        ),
    }
