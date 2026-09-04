"""Archived routes as master columns: plan reconstruction, discovery, re-certification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacepdhcg.gtoc12 import archive
from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.archive import discover_archives, group_plans, recertify_archives
from spacepdhcg.gtoc12.bundles import bundle_columns
from spacepdhcg.gtoc12.pipeline import RefinedRoute, plan_from_route_summary
from spacepdhcg.gtoc12.search import EARTH_ID, PlannedLeg, RoutePlan

T0 = C.MISSION_START_MJD
RATE = C.MINING_RATE_KG_PER_YEAR / C.YEAR_DAYS


def _leg(a, b, t0, tf, role):
    return PlannedLeg(a, b, t0, tf, 1.0, 1.0, role)


def _self_cleaning_plan() -> RoutePlan:
    """Earth -> 11 (deploy) -> 12 (deploy, camp) -> 11 (collect at arrival) -> Earth."""

    legs = (
        _leg(EARTH_ID, 11, T0, T0 + 400.0, "earth_out"),
        _leg(11, 12, T0 + 400.0, T0 + 600.0, "deploy_hop"),
        _leg(12, 12, T0 + 600.0, T0 + 1200.0, "camp"),
        _leg(12, 11, T0 + 1200.0, T0 + 1400.0, "collect_hop"),
        _leg(11, EARTH_ID, T0 + 1400.0, T0 + 1900.0, "earth_return"),
    )
    deploy = {11: T0 + 400.0, 12: T0 + 600.0}
    collect = {12: T0 + 1200.0, 11: T0 + 1400.0}
    collected = {a: RATE * (collect[a] - deploy[a]) for a in deploy}
    return RoutePlan(legs, deploy, collect, collected, 900.0, 3000.0 - 900.0 - 80.0)


def _cooperative_plans() -> tuple[RoutePlan, RoutePlan]:
    """A deployer that leaves 22 behind and a collector that picks it up (at its arrival)."""

    deployer = RoutePlan(
        (
            _leg(EARTH_ID, 21, T0, T0 + 400.0, "earth_out"),
            _leg(21, 22, T0 + 400.0, T0 + 600.0, "deploy_hop"),
            _leg(22, 21, T0 + 600.0, T0 + 1300.0, "collect_hop"),
            _leg(21, EARTH_ID, T0 + 1300.0, T0 + 1800.0, "earth_return"),
        ),
        {21: T0 + 400.0, 22: T0 + 600.0},
        {21: T0 + 1300.0},
        {21: RATE * 900.0},
        800.0,
        3000.0 - 800.0 - 80.0,
    )
    collector = RoutePlan(
        (
            _leg(EARTH_ID, 23, T0 + 100.0, T0 + 500.0, "earth_out"),
            _leg(23, 22, T0 + 500.0, T0 + 2000.0, "collect_hop"),
            _leg(22, 23, T0 + 2000.0, T0 + 2200.0, "collect_hop"),
            _leg(23, EARTH_ID, T0 + 2200.0, T0 + 2700.0, "earth_return"),
        ),
        {23: T0 + 500.0},
        {22: T0 + 2000.0, 23: T0 + 2200.0},
        {22: RATE * 1400.0, 23: RATE * 1700.0},
        700.0,
        3000.0 - 700.0 - 40.0,
        {22: T0 + 600.0},
    )
    return deployer, collector


def _legacy_summary(plan: RoutePlan, *, final_mass: float = 700.0) -> dict:
    """What older ``route_summary.json`` files carry: flown legs + collected masses only."""

    return {
        "certified": True,
        "asteroids": list(plan.asteroids),
        "collected_mass_kg": {str(a): m for a, m in plan.collected_mass.items()},
        "total_collected_kg": plan.total_collected_kg,
        "final_mass_kg": final_mass,
        "legs": [
            {
                "from": leg.from_id,
                "to": leg.to_id,
                "t0": leg.departure_epoch,
                "tf": leg.arrival_epoch,
            }
            for leg in plan.legs
            if leg.role != "camp"
        ],
    }


def _same_schedule(rebuilt: RoutePlan, truth: RoutePlan) -> None:
    assert rebuilt.deploy_epochs == truth.deploy_epochs
    assert rebuilt.collect_epochs == truth.collect_epochs
    assert rebuilt.foreign_deploy_epochs == pytest.approx(truth.foreign_deploy_epochs)
    assert rebuilt.collected_mass == pytest.approx(truth.collected_mass)
    flown = [(leg.from_id, leg.to_id, leg.role) for leg in truth.legs if leg.role != "camp"]
    assert [(leg.from_id, leg.to_id, leg.role) for leg in rebuilt.legs] == flown


def test_plan_reconstruction_from_legacy_summary_self_cleaning_with_camp() -> None:
    truth = _self_cleaning_plan()
    rebuilt = plan_from_route_summary(_legacy_summary(truth))
    _same_schedule(rebuilt, truth)
    assert rebuilt.self_cleaning and not rebuilt.orphaned
    assert rebuilt.final_mass_proxy_kg == pytest.approx(700.0 + truth.total_collected_kg)
    assert rebuilt.feasible


def test_plan_reconstruction_recovers_foreign_collects_and_snaps_to_the_deployer() -> None:
    deployer, collector = _cooperative_plans()
    d = plan_from_route_summary(_legacy_summary(deployer))
    _same_schedule(d, deployer)
    assert d.orphaned == (22,)
    # without the deployer the foreign epoch is backed out of the mass (collect at arrival)
    alone = plan_from_route_summary(_legacy_summary(collector))
    _same_schedule(alone, collector)
    assert alone.foreign_deploy_epochs[22] == pytest.approx(T0 + 600.0)
    # with the deployer known the epoch is snapped exactly (the pool requires 1e-6 days)
    snapped = plan_from_route_summary(_legacy_summary(collector), deployers=d.deploy_epochs)
    assert snapped.foreign_deploy_epochs == {22: T0 + 600.0}


def test_plan_reconstruction_prefers_the_embedded_plan_and_resnaps_foreign_epochs() -> None:
    _deployer, collector = _cooperative_plans()
    summary = {"certified": True, "plan": collector.summary(), "legs": [], "collected_mass_kg": {}}
    same = plan_from_route_summary(summary)
    assert same == collector
    moved = plan_from_route_summary(summary, deployers={22: T0 + 601.0})
    assert moved.foreign_deploy_epochs == {22: T0 + 601.0} and moved.legs == collector.legs


def _write(root: Path, rel: str, payload: dict) -> None:
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    (path / "route_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_discover_archives_groups_by_ship_parent_and_orders_variants(tmp_path: Path) -> None:
    deployer, collector = _cooperative_plans()
    single = _self_cleaning_plan()
    run = tmp_path / "runs" / "fleet_run"
    _write(run, "ship_01/retimed", _legacy_summary(single) | {"total_collected_kg": 50.0})
    _write(run, "ship_01/candidate_00", _legacy_summary(single) | {"total_collected_kg": 40.0})
    _write(run, "ship_02/candidate_00", _legacy_summary(single) | {"certified": False})
    _write(run, "fleets/fleet_000_02ships/ship_01", _legacy_summary(single))  # never a column
    family = tmp_path / "runs" / "cluster_run" / "clusters" / "family_0007"
    _write(family, "ship_01", _legacy_summary(deployer))
    _write(family, "ship_03", _legacy_summary(collector))
    groups = discover_archives([tmp_path / "runs" / "cluster_run", run])
    assert [g.name for g in groups] == ["cluster_run/clusters/family_0007", "fleet_run"]
    fam, fleet = groups
    assert [s.slot for s in fam.ships] == [1, 3]
    assert [s.slot for s in fleet.ships] == [1]  # ship_02 had no certified archive
    assert [p.parent.name for p, _ in fleet.ships[0].summaries] == ["retimed", "candidate_00"]
    plans = group_plans(fam)
    assert plans[3][0].foreign_deploy_epochs == {22: T0 + 600.0}
    assert plans[1][0].orphaned == (22,)
    # discovery is deterministic regardless of source order
    again = discover_archives([run, tmp_path / "runs" / "cluster_run"])
    assert [g.name for g in again] == [g.name for g in groups]
    # an external archive directory (another machine's run handed over as a bundle) is one
    # more source: same layout, its own group even when the family label coincides with ours
    external = tmp_path / "external" / "cluster_fleet_h100_v1"
    _write(external / "clusters" / "family_0007", "ship_01", _legacy_summary(deployer))
    _write(external / "clusters" / "family_0007", "ship_03", _legacy_summary(collector))
    merged = discover_archives([tmp_path / "runs" / "cluster_run", run, external])
    assert [g.name for g in merged] == [
        "cluster_fleet_h100_v1/clusters/family_0007",
        "cluster_run/clusters/family_0007",
        "fleet_run",
    ]
    assert len({g.directory for g in merged}) == 3
    assert [s.slot for s in merged[0].ships] == [1, 3]


def _fake_refine(plan: RoutePlan, catalogue, scvx=None) -> RefinedRoute:
    certified = 22 not in plan.deploy_epochs  # the deployer of 22 "fails" re-certification
    return RefinedRoute(
        plan=plan,
        legs=[],
        collected_mass=dict(plan.collected_mass),
        final_mass_kg=700.0,
        certified=certified,
        master_certified=certified,
        passes=1,
        wall_seconds=0.0,
        scheduler_telemetry={},
        failures=[] if certified else [{"reason": "fake"}],
    )


def test_recertify_archives_drops_failed_and_stranded_ships_and_builds_columns(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(archive, "refine_route", _fake_refine)
    deployer, collector = _cooperative_plans()
    single = _self_cleaning_plan()
    family = tmp_path / "clusters" / "family_0007"
    _write(family, "ship_01", _legacy_summary(deployer))
    _write(family, "ship_03", _legacy_summary(collector))
    other = tmp_path / "clusters" / "family_0008"
    _write(other, "ship_01", _legacy_summary(single))
    _write(other, "ship_02", _legacy_summary(collector) | {"total_collected_kg": 1.0})
    progress: list[dict] = []
    bundles = recertify_archives(
        None, discover_archives([tmp_path]), workers=1, on_progress=progress.append
    )
    assert len(progress) == 4 and progress[-1]["done"] == 4
    seven, eight = bundles
    # family 7: the deployer failed, so its collector is stranded and the bundle is empty
    assert seven.ships == [] and seven.label == 10_000
    reasons = [r["reason"] for r in seven.rejected]
    assert "archived primary route failed re-certification" in reasons
    assert [r["kind"] for r in seven.repairs] == ["removed_stranded"]
    # family 8: the self-cleaning ship stays; the collector of a miner nobody deploys is dropped
    assert [s.slot for s in eight.ships] == [1] and eight.consistent() == ""
    assert eight.members == (11, 12)
    columns = bundle_columns(eight, 5, prefix="a")
    assert [c.identifier for c in columns] == [5] and columns[0].label == "a10001_s1"
    assert columns[0].route is eight.ships[0].route
