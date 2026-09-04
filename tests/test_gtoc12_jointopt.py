"""Whole-itinerary joint re-optimisation: bookkeeping exactness, monotone certified acceptance,
determinism, no double deploy/collection, task selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.pipeline import RefinedLeg, RefinedRoute
from spacepdhcg.gtoc12.screening import propellant_for_delta_v
from spacepdhcg.gtoc12.search import EARTH_ID, RoutePlan

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")

ARCHIVE = Path(
    "results/gtoc12/runs/return_sweep_v2/ships/cluster_fleet_v7_clusters_family_0007/ship_01/"
    "route_summary.json"
)


@pytest.fixture(scope="module")
def catalogue():
    if not data_available():
        pytest.skip("pinned GTOC12 data not fetched")
    return load_catalogue()


@pytest.fixture(scope="module")
def archived():
    if not ARCHIVE.exists():
        pytest.skip("archived route not available")
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))


def _retimer(catalogue, weights=None):
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
    )
    from spacepdhcg.gtoc12.retiming import Retimer

    pricing = ClusterPricingSettings(collect_dp_inflation_fit=None)
    return Retimer(
        catalogue,
        cluster_search_settings(pricing, 40),
        cluster_retime_settings(pricing, last=True),
        weights,
    )


class _Solution:
    def __init__(self, delta_v: float, propellant: float) -> None:
        self.delta_v_km_s = delta_v
        self.propellant_kg = propellant


def _proxy_refine(plan: RoutePlan, *, factor: float = 1.0, fail_pair=None) -> RefinedRoute:
    """Stand-in for SCvx: flies every leg at ``factor`` x the plan's inflated proxy ΔV and
    reports the exact forward masses; ``fail_pair`` marks one body pair as not certifiable."""

    collected = dict(plan.collected_mass)
    legs: list[RefinedLeg] = []
    mass = C.MAX_INITIAL_MASS_KG
    certified = True
    flown = [leg for leg in plan.legs if leg.role != "camp"]
    for index, leg in enumerate(flown):
        if leg.from_id in plan.collect_epochs and (
            abs(plan.collect_epochs[leg.from_id] - leg.departure_epoch) < 1e-6
        ):
            mass += collected[leg.from_id]
        dv = factor * leg.delta_v_proxy_km_s * leg.inflation
        propellant = float(propellant_for_delta_v(mass, dv))
        ok = fail_pair != (leg.from_id, leg.to_id)
        legs.append(
            RefinedLeg(
                leg,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                _Solution(dv, propellant),  # type: ignore[arg-type]
                None,
                ok,
                mass,
                mass - propellant,
            )
        )
        if not ok:
            certified = False
            break
        mass -= propellant
        if leg.to_id != EARTH_ID and leg.to_id in plan.deploy_epochs:
            if abs(plan.deploy_epochs[leg.to_id] - leg.arrival_epoch) < 1e-6:
                mass -= C.MINER_MASS_KG
        if (
            leg.to_id in plan.collect_epochs
            and abs(plan.collect_epochs[leg.to_id] - leg.arrival_epoch) < 1e-6
            and not (index + 1 < len(flown) and flown[index + 1].from_id == leg.to_id)
        ):
            mass += collected[leg.to_id]
    final = mass - sum(collected.values())
    if final < C.DRY_MASS_KG - 1e-9:
        certified = False
    return RefinedRoute(
        plan=plan,
        legs=legs,
        collected_mass=collected if certified else {a: 0.0 for a in collected},
        final_mass_kg=final,
        certified=certified,
        master_certified=certified,
        passes=1,
        wall_seconds=0.0,
        scheduler_telemetry={},
        failures=[] if certified else [{"reason": "proxy"}],
    )


# -- bookkeeping exactness -----------------------------------------------------------------


@requires_data
def test_warm_start_reproduces_the_certified_bookkeeping_exactly(catalogue, archived) -> None:
    from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
    from spacepdhcg.gtoc12.retiming import visits_of

    route = route_from_summary(archived)
    assert route.certified and len(route.legs) == len(archived["legs"])
    retimer = _retimer(catalogue)
    joint = JointItinerary(catalogue, retimer)
    assert joint.learn(route) == len(route.legs)
    assert len(retimer.inflations) > 0  # every certified pair calibrated the surrogate
    visits, arr, dep = visits_of(route.plan)
    ev = joint.evaluate(visits, np.asarray(arr), np.asarray(dep))
    assert ev.feasible, ev.failure
    # every leg priced with its measured ΔV: the surrogate *is* the certified route here
    assert ev.measured_legs == len(route.legs)
    assert ev.collected_kg == pytest.approx(route.total_collected_kg, abs=1e-9)
    assert ev.spare_kg == pytest.approx(route.final_mass_kg - C.DRY_MASS_KG, abs=1e-6)
    plan = ev.plan
    # mining-rate bookkeeping: 10 kg/yr x (collect - deploy), deterministic
    for asteroid, mass in plan.collected_mass.items():
        stay = plan.collect_epochs[asteroid] - plan.deploy_epochs[asteroid]
        assert stay >= C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS - 1e-6
        assert mass == pytest.approx(C.MINING_RATE_KG_PER_YEAR * stay / C.YEAR_DAYS, abs=1e-12)
    # mass identity: initial - propellant - miners + collected = final (before unloading)
    flown = [leg for leg in plan.legs if leg.role != "camp"]
    assert plan.final_mass_proxy_kg == pytest.approx(
        C.MAX_INITIAL_MASS_KG
        - plan.propellant_proxy_kg
        - C.MINER_MASS_KG * len(plan.deploy_epochs)
        + sum(plan.collected_mass.values()),
        abs=1e-9,
    )
    assert len(ev.masses) == len(flown)
    assert ev.masses[0] == C.MAX_INITIAL_MASS_KG
    # a re-evaluation is bit-identical (deterministic, memoised Lambert)
    again = joint.evaluate(visits, np.asarray(arr), np.asarray(dep))
    assert again.objective == ev.objective and again.plan == ev.plan


@requires_data
def test_evaluate_rejects_rule_violations(catalogue, archived) -> None:
    from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
    from spacepdhcg.gtoc12.retiming import Visit, visits_of

    route = route_from_summary(archived)
    joint = JointItinerary(catalogue, _retimer(catalogue))
    joint.learn(route)
    visits, arr0, dep0 = visits_of(route.plan)
    arr0, dep0 = np.asarray(arr0), np.asarray(dep0)
    camp = next(j for j, v in enumerate(visits) if v.deploy and v.collect)

    def check(reason, visits=visits, arr=None, dep=None):
        ev = joint.evaluate(visits, arr0 if arr is None else arr, dep0 if dep is None else dep)
        assert not ev.feasible and ev.failure == reason, (ev.failure, reason)

    a = arr0.copy()
    a[-1] = C.MISSION_END_MJD + 1.0
    d = dep0.copy()
    d[-1] = a[-1]
    check("return_after_window", arr=a, dep=d)
    a = arr0.copy()
    d = dep0.copy()
    a[0] = d[0] = C.MISSION_START_MJD - 1.0
    check("launch_before_window", arr=a, dep=d)
    d = dep0.copy()
    d[camp] = arr0[camp] - 1.0  # leaves before it arrives
    check("negative_dwell", dep=d)
    # the camp collects at departure: a stay under a year breaks the mining rule
    a = arr0.copy()
    d = dep0.copy()
    a[camp] = d[camp] - 300.0
    check("stay_too_short", arr=a, dep=d)
    # the same asteroid collected twice, or deployed twice
    twice = list(visits)
    body = visits[camp + 1].body if camp + 1 < len(visits) - 1 else visits[camp].body
    twice[camp + 1] = Visit(body, False, True, twice[camp + 1].role_out)
    twice.insert(camp + 2, Visit(body, False, True, "collect_hop"))
    a = np.insert(arr0, camp + 2, arr0[camp + 1] + 1.0)
    d = np.insert(dep0, camp + 2, arr0[camp + 1] + 1.0)
    ev = joint.evaluate(twice, a, d)
    assert not ev.feasible and ev.failure in ("double_collect", "tof_outside_limits")
    dup = list(visits)
    dup[1] = Visit(visits[2].body, True, False, visits[1].role_out)
    ev = joint.evaluate(dup, arr0, dep0)
    assert not ev.feasible and ev.failure in (
        "double_deploy",
        "leg_infeasible",
        "tof_outside_limits",
    )
    # a hop shorter than the envelope
    a = arr0.copy()
    a[2] = dep0[1] + 10.0
    check("tof_outside_limits", arr=a)


# -- pattern search: determinism and monotonicity ---------------------------------------------


@requires_data
def test_pattern_search_is_deterministic_and_never_worsens_the_surrogate(
    catalogue, archived
) -> None:
    from spacepdhcg.gtoc12.jointopt import JointItinerary, JointSettings, route_from_summary
    from spacepdhcg.gtoc12.retiming import visits_of

    route = route_from_summary(archived)
    runs = []
    for _ in range(2):
        joint = JointItinerary(catalogue, _retimer(catalogue), settings=JointSettings())
        joint.learn(route)
        visits, arr, dep = visits_of(route.plan)
        base = joint.evaluate(visits, np.asarray(arr), np.asarray(dep))
        a2, d2, best, moves = joint.optimise_epochs(
            visits, np.asarray(arr), np.asarray(dep), mesh=(45.0, 20.0), max_moves=6
        )
        assert best.feasible and best.objective >= base.objective
        assert moves <= 12
        # epochs stay inside the window and the visit order's rules
        assert a2[0] >= C.MISSION_START_MJD and a2[-1] <= C.MISSION_END_MJD
        assert np.all(d2[1:-1] >= a2[1:-1] - 1e-9)
        runs.append((a2.tolist(), d2.tolist(), best.objective, moves))
    assert runs[0] == runs[1]


def test_move_set_is_deterministic_and_covers_single_epochs_visits_and_phases() -> None:
    from spacepdhcg.gtoc12.jointopt import JointItinerary

    moves = list(JointItinerary.moves(5, 10.0))
    assert moves == list(JointItinerary.moves(5, 10.0))
    # launch and return move as pairs (arrival == departure at Earth)
    assert {0: (10.0, 10.0)} in moves and {4: (-10.0, -10.0)} in moves
    # single epochs of an inner visit, the whole visit, and the phase shifts around it
    assert {2: (10.0, 0.0)} in moves and {2: (0.0, 10.0)} in moves and {2: (10.0, 10.0)} in moves
    assert {0: (10.0, 10.0), 1: (10.0, 10.0), 2: (10.0, 0.0)} in moves
    assert {2: (0.0, -10.0), 3: (-10.0, -10.0), 4: (-10.0, -10.0)} in moves
    assert {i: (10.0, 10.0) for i in range(5)} in moves


# -- certified acceptance -------------------------------------------------------------------


@requires_data
def test_optimise_ship_accepts_only_certified_gains_and_is_monotone(catalogue, archived) -> None:
    from spacepdhcg.gtoc12.jointopt import JointSettings, optimise_ship, route_from_summary

    route = route_from_summary(archived)
    settings = JointSettings(mesh_days=(45.0, 20.0), max_certifications=4, insert=False)
    # a refiner that trusts the surrogate: every certification is a gain and is accepted
    result = optimise_ship(
        route, catalogue, _retimer(catalogue), settings=settings, refine=_proxy_refine
    )
    assert result.baseline_error_kg == pytest.approx(0.0, abs=1e-9)
    accepted = [a for a in result.attempts if a.get("result") == "accepted"]
    assert result.route is not None and result.gain_kg > 0.0
    assert all(a["certified"] and a["certified_kg"] > route.total_collected_kg for a in accepted)
    # monotone: every accepted certification collects more than the previous one
    kgs = [a["certified_kg"] for a in accepted]
    assert kgs == sorted(kgs) and len(set(kgs)) == len(kgs)
    assert result.after_kg == pytest.approx(kgs[-1])
    assert result.route.certified and result.route.total_collected_kg == pytest.approx(kgs[-1])
    # no asteroid is deployed or collected twice in the accepted plan
    plan = result.route.plan
    deploys = [leg.to_id for leg in plan.legs if leg.role in ("earth_out", "deploy_hop")]
    assert len(deploys) == len(set(deploys)) == len(plan.deploy_epochs)
    assert set(plan.collect_epochs) <= set(plan.deploy_epochs)
    # determinism: the same inputs give the same accepted route
    again = optimise_ship(
        route, catalogue, _retimer(catalogue), settings=settings, refine=_proxy_refine
    )
    assert again.after_kg == result.after_kg
    assert again.route.plan.legs == result.route.plan.legs

    # a refiner that refuses one hop: nothing is accepted, the pair is banned, before unchanged
    pair = next((leg.from_id, leg.to_id) for leg in route.plan.legs if leg.role == "deploy_hop")
    retimer = _retimer(catalogue)
    refused = optimise_ship(
        route,
        catalogue,
        retimer,
        settings=settings,
        refine=lambda plan: _proxy_refine(plan, fail_pair=pair),
    )
    assert refused.route is None and refused.gain_kg == 0.0
    assert refused.before is route and refused.before_kg == route.total_collected_kg
    assert pair in retimer.bans or refused.certifications == 0
    # a refiner that flies every leg 30 % dearer than the surrogate cannot beat the archived
    # route: certified routes with less ore are never accepted
    dearer = optimise_ship(
        route,
        catalogue,
        _retimer(catalogue),
        settings=settings,
        refine=lambda plan: _proxy_refine(plan, factor=1.3),
    )
    assert dearer.route is None or dearer.route.total_collected_kg > route.total_collected_kg
    for attempt in dearer.attempts:
        if (
            attempt.get("certified")
            and attempt.get("certified_kg", 0.0) <= route.total_collected_kg
        ):
            assert attempt["result"] != "accepted"


@requires_data
def test_insertions_add_exactly_one_asteroid_without_double_visits(catalogue, archived) -> None:
    from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
    from spacepdhcg.gtoc12.retiming import visits_of
    from spacepdhcg.gtoc12.returnsweep import neighbourhood

    route = route_from_summary(archived)
    joint = JointItinerary(catalogue, _retimer(catalogue))
    joint.learn(route)
    visits, arr, dep = visits_of(route.plan)
    own = set(route.plan.asteroids)
    from types import SimpleNamespace

    from spacepdhcg.gtoc12.clusters import ClusterBands

    pool = neighbourhood(
        catalogue,
        route.plan,
        SimpleNamespace(cluster_bands=ClusterBands.collect_window(radius=3.0)),
        count=60,
    )
    candidates = [int(a) for a in pool.tolist()]
    options = joint.insertions(visits, np.asarray(arr), np.asarray(dep), candidates)
    # the ship's own asteroids are never inserted (candidates contains them)
    assert own <= set(candidates)
    for new_visits, a, d, ev, asteroid in options:
        assert asteroid not in own
        bodies = [v.body for v in new_visits[1:-1]]
        deploys = [v.body for v in new_visits if v.deploy]
        collects = [v.body for v in new_visits if v.collect]
        assert len(deploys) == len(set(deploys)) == len(own) + 1
        assert len(collects) == len(set(collects)) == len(own) + 1
        assert bodies.count(asteroid) == 2  # one deploy visit, one collect visit
        assert ev.feasible and ev.plan is not None
        assert set(ev.plan.deploy_epochs) == own | {asteroid}
        assert a.shape == d.shape == (len(new_visits),)
    # options are sorted best first
    objectives = [ev.objective for *_rest, ev, _a in options]
    assert objectives == sorted(objectives, reverse=True)


# -- task selection --------------------------------------------------------------------------


def test_select_tasks_puts_fleet_ships_first_and_dedups_by_asteroid_set(tmp_path) -> None:
    from spacepdhcg.gtoc12.archive import ArchivedGroup, ArchivedShip
    from spacepdhcg.gtoc12.jointcampaign import (
        JointCampaignSettings,
        fleet_asteroid_sets,
        select_tasks,
    )

    def summary(asteroids, kg, *, foreign=None, orphaned=(), legs=True):
        return {
            "total_collected_kg": kg,
            "plan": {
                "asteroids": list(asteroids),
                "foreign_deploy_epochs": dict(foreign or {}),
                "orphaned": list(orphaned),
                "legs": [{"role": "earth_return"}] if legs else [{"role": "deploy_hop"}],
            },
        }

    g1 = ArchivedGroup(
        "run/a",
        tmp_path,
        [
            ArchivedShip(
                1,
                [
                    (tmp_path / "a1", summary([1, 2, 3], 500.0)),
                    (tmp_path / "a1b", summary([1, 2, 9], 480.0)),
                ],
            ),
            ArchivedShip(2, [(tmp_path / "a2", summary([4, 5], 700.0))]),
        ],
    )
    g2 = ArchivedGroup(
        "run/b",
        tmp_path,
        [
            ArchivedShip(1, [(tmp_path / "b1", summary([1, 2, 3], 520.0))]),  # same set, heavier
            ArchivedShip(2, [(tmp_path / "b2", summary([6, 7], 600.0, foreign={7: 1.0}))]),
            ArchivedShip(3, [(tmp_path / "b3", summary([8], 460.0, orphaned=[8]))]),
            ArchivedShip(4, [(tmp_path / "b4", summary([10], 900.0, legs=False))]),
        ],
    )
    report = tmp_path / "run_report.json"
    report.write_text(
        json.dumps({"master": {"selected": [{"deploys": [9, 2, 1]}, {"deploys": [8]}]}}),
        encoding="utf-8",
    )
    assert fleet_asteroid_sets(report) == [(1, 2, 9), (8,)]
    tasks = select_tasks(
        [g1, g2], JointCampaignSettings(min_collected_kg=450.0), fleet_asteroid_sets(report)
    )
    names = [(t.group, t.slot, t.asteroids, t.in_fleet) for t in tasks]
    # fleet ships first in the report's order: the non-primary variant (1, 2, 9) is matched,
    # the orphaned stand-alone ship (8,) is allowed; then the rest best first, one per set,
    # the heaviest archive of a duplicated set kept; foreign collectors and ships without an
    # Earth return excluded
    assert names[:2] == [("run/a", 1, (1, 2, 9), True), ("run/b", 3, (8,), True)]
    assert names[2:] == [("run/a", 2, (4, 5), False), ("run/b", 1, (1, 2, 3), False)]
    top = select_tasks([g1, g2], JointCampaignSettings(top=1), fleet_asteroid_sets(report))
    assert [t.asteroids for t in top] == [(1, 2, 9), (8,), (4, 5)]
    assert all(math.isfinite(t.collected_kg) for t in tasks)
