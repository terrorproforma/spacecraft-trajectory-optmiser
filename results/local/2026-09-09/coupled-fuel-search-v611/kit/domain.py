"""Construct fixed-order coupled seeds and unchanged native joint evaluators."""

# Frozen source activation must precede production imports.
# ruff: noqa: E402

import math

import common
import numpy as np

common.activate()
from fixed_refine import validate_prescription

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.bundles import (
    ClusterPricingSettings,
    cluster_retime_settings,
    cluster_search_settings,
)
from spacepdhcg.gtoc12.jointopt import JointItinerary, JointSettings, route_from_summary
from spacepdhcg.gtoc12.retiming import Retimer, visits_of
from spacepdhcg.gtoc12.search import PlannedLeg, RoutePlan


def routes():
    control = route_from_summary(common.read(common.ROOT / "inputs/control-route.json"))
    prescription = common.read(common.ROOT / "inputs/probe-prescription.json")["plan"]
    legs = []
    for i in range(17):
        record = common.read(common.ROOT / "inputs" / f"probe-leg-{i:02d}.json")
        certificate = record["certificate"]
        legs.append(
            {
                "from": record["from"],
                "to": record["to"],
                "t0": record["departure"],
                "tf": record["arrival"],
                "mass_before": record["initial_mass_kg"],
                "mass_after": certificate["final_mass_kg"] if certificate else math.nan,
                "propellant_kg": record["solution"]["propellant_kg"],
                "certified": record["certified"],
            }
        )
    probe = route_from_summary(
        {
            "plan": prescription,
            "collected_mass_kg": prescription["collected_mass_kg"],
            "final_mass_kg": math.nan,
            "certified": False,
            "master_certified": False,
            "legs": legs,
        }
    )
    if sum(leg.certified for leg in probe.legs) != 16 or probe.legs[-1].certified:
        raise ValueError("Probe requires16 certified prefix legs and failed return")
    return control, probe


def joint_for(catalogue, weights, price):
    control, probe = routes()
    policy = ClusterPricingSettings()
    retimer = Retimer(
        catalogue,
        cluster_search_settings(policy, len(catalogue.ids)),
        cluster_retime_settings(policy, last=True),
        weights=weights,
    )
    retimer.protect_earth_leg(control.plan)
    settings = JointSettings(
        mesh_days=common.MESH,
        max_moves_per_mesh=common.MAX_MOVES,
        margin_price=price,
        insert=False,
        earth_leg=False,
        max_certifications=0,
        time_budget_seconds=30.0,
    )
    joint = JointItinerary(catalogue, retimer, weights=weights, settings=settings)
    joint.learn(control)
    joint.learn(probe)
    return joint, probe


def seeds(plan):
    visits, arr, dep = visits_of(plan)
    first_collect = next(i for i, visit in enumerate(visits) if visit.collect)
    if len(visits) != 18 or first_collect != 9:
        raise ValueError("Reviewed18-visit8-miner itinerary required")
    shifts = (
        ("as_flown_prefix", 0, 0),
        ("earlier_deploy_phase", -30, 0),
        ("earlier_collect_phase", 0, -30),
        ("longer_mining_gap", -30, 30),
    )
    result = []
    for name, deploy, collect in shifts:
        a, d = np.asarray(arr, dtype=float).copy(), np.asarray(dep, dtype=float).copy()
        a[:first_collect] += deploy
        d[:first_collect] += deploy
        a[first_collect:] += collect
        d[first_collect:] += collect
        result.append({"id": name, "visits": visits, "arrivals": a, "departures": d})
    return result


def physical_plan(visits, arrivals, departures):
    deploy, collect, legs = {}, {}, []
    for i, visit in enumerate(visits):
        if visit.deploy:
            deploy[visit.body] = float(arrivals[i])
        if visit.collect:
            collect[visit.body] = float(departures[i])
        if i == len(visits) - 1:
            continue
        if departures[i] > arrivals[i]:
            legs.append(
                PlannedLeg(
                    visit.body, visit.body, float(arrivals[i]), float(departures[i]), 0, 1, "camp"
                )
            )
        legs.append(
            PlannedLeg(
                visit.body,
                visits[i + 1].body,
                float(departures[i]),
                float(arrivals[i + 1]),
                0,
                1,
                visit.role_out,
            )
        )
    cargo = {
        body: C.maximum_collected_mass(epoch - deploy[body]) for body, epoch in collect.items()
    }
    plan = RoutePlan(tuple(legs), deploy, collect, cargo, 0, 3000)
    validate_prescription(plan, cargo)
    return plan


def search_order():
    # Every arm sees exactly the same seeds and caps; alternate ordering between seeds.
    return [
        (seed, price)
        for seed in range(4)
        for price in (common.PRICES if seed % 2 == 0 else tuple(reversed(common.PRICES)))
    ]


def bounds(visits=18):
    rows = 1 + len(common.MESH) * common.MAX_MOVES * (10 * visits - 14)
    return {
        "searches": 12,
        "visits": visits,
        "flight_legs_per_route": visits - 1,
        "max_rows_per_search": rows,
        "max_itinerary_rows": 12 * rows,
        "max_lambert_direction_requests": 12 * rows * (visits - 1) * 2,
        "max_full_refinements": 4,
        "max_native_leg_solves": 4 * (visits - 1),
    }


def incumbent_joint(catalogue, weights, ship, price):
    if ship not in common.SHIPS or price not in common.PRICES:
        raise ValueError("Unreviewed ship or objective arm")
    route = route_from_summary(common.read(common.ROOT / "inputs" / f"ship-{ship:02d}.json"))
    visits, arrivals, departures = visits_of(route.plan)
    if not route.certified or len(route.legs) != 17 or len(visits) != 18:
        raise ValueError("Certified17-leg incumbent required")
    if route.plan.foreign_deploy_epochs or route.plan.orphaned:
        raise ValueError("Independent own-miner closed itinerary required")
    policy = ClusterPricingSettings()
    retimer = Retimer(
        catalogue,
        cluster_search_settings(policy, len(catalogue.ids)),
        cluster_retime_settings(policy, last=True),
        weights=weights,
    )
    retimer.protect_earth_leg(route.plan)
    joint = JointItinerary(
        catalogue,
        retimer,
        weights=weights,
        settings=JointSettings(
            mesh_days=common.MESH,
            max_moves_per_mesh=common.MAX_MOVES,
            margin_price=price,
            insert=False,
            earth_leg=False,
            max_certifications=0,
            time_budget_seconds=30,
        ),
    )
    if joint.learn(route) != 17:
        raise ValueError("Every original leg must have measured certified cost")
    return joint, route, visits, np.asarray(arrivals), np.asarray(departures)


def shortlist(rows):
    selected, used = [], set()
    for price in common.PRICES:
        candidates = sorted(
            (r for r in rows if r["margin_price"] == price and r.get("objective_eligible")),
            key=lambda r: (-r["weighted_gain_kg"], -r["spare_kg"], r["ship"]),
        )
        for row in candidates:
            identity = row["plan_sha256"]
            if identity not in used:
                selected.append(row)
                used.add(identity)
                break
    if len(selected) > common.MAX_PROBES:
        raise AssertionError("At most one distinct candidate per objective arm")
    return selected
