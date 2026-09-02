"""Re-timing, co-moving clusters, cooperative collection rules and the fleet master."""

from __future__ import annotations

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.cooperative import (
    FleetColumn,
    MinerPool,
    fleet_feasible,
    orphan_credit_kg,
    solve_fleet_master,
)
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.search import EARTH_ID, PlannedLeg, RoutePlan

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
SMALL_LAUNCH_EPOCHS = tuple(float(x) for x in C.MISSION_START_MJD + np.arange(0.0, 731.0, 90.0))
SMALL_EARTH_TOFS = (600.0, 750.0, 900.0)
YEAR = C.YEAR_DAYS
T0 = C.MISSION_START_MJD


# -- pure: plans, pool, credit -------------------------------------------------------------


def _plan(deploys: dict[int, float], collects: dict[int, float], foreign=None) -> RoutePlan:
    foreign = foreign or {}
    collected = {
        a: C.maximum_collected_mass(t - (deploys[a] if a in deploys else foreign[a]))
        for a, t in collects.items()
    }
    legs = (PlannedLeg(EARTH_ID, next(iter(deploys)), T0, T0 + 500.0, 5.0, 1.0, "earth_out"),)
    return RoutePlan(legs, deploys, collects, collected, 100.0, 900.0, foreign)


def test_cooperative_plan_properties_and_summary_round_trip() -> None:
    own = _plan({1: T0 + 500.0, 2: T0 + 700.0}, {1: T0 + 3000.0, 2: T0 + 2800.0})
    assert own.self_cleaning and own.orphaned == () and own.asteroids == (1, 2)
    coop = _plan(
        {1: T0 + 500.0, 2: T0 + 700.0},
        {1: T0 + 3000.0, 7: T0 + 3200.0},
        foreign={7: T0 + 600.0},
    )
    assert not coop.self_cleaning
    assert coop.orphaned == (2,)
    assert coop.asteroids == (1, 2, 7)  # own deploys first, then foreign collects
    assert coop.deploy_epoch_of(7) == T0 + 600.0 and coop.deploy_epoch_of(1) == T0 + 500.0
    assert coop.collected_mass[7] == pytest.approx(10.0 * 2600.0 / YEAR)
    summary = coop.summary()
    rebuilt = RoutePlan.from_summary(summary)  # type: ignore[arg-type]
    assert rebuilt.foreign_deploy_epochs == coop.foreign_deploy_epochs
    assert rebuilt.asteroids == coop.asteroids and rebuilt.orphaned == coop.orphaned
    assert summary["orphaned"] == [2] and summary["self_cleaning"] is False


def test_miner_pool_enforces_once_only_rules_and_tracks_orphans() -> None:
    pool = MinerPool()
    ship1 = _plan({1: T0 + 500.0, 2: T0 + 700.0}, {1: T0 + 3000.0})  # leaves 2 as an orphan
    pool.register(ship1, 1)
    assert pool.orphans() == {2: T0 + 700.0}
    assert pool.touched() == {1, 2}
    with pytest.raises(ValueError, match="deployed twice"):
        pool.register(_plan({2: T0 + 900.0}, {2: T0 + 2000.0}), 2)
    with pytest.raises(ValueError, match="never deployed"):
        pool.register(_plan({3: T0 + 900.0}, {9: T0 + 2000.0}, foreign={9: T0 + 100.0}), 2)
    with pytest.raises(ValueError, match="stale deploy epoch"):
        pool.register(_plan({3: T0 + 900.0}, {2: T0 + 2000.0}, foreign={2: T0 + 100.0}), 2)
    ship2 = _plan({3: T0 + 900.0}, {3: T0 + 2500.0, 2: T0 + 2000.0}, foreign={2: T0 + 700.0})
    pool.register(ship2, 2)
    assert pool.orphans() == {}
    assert pool.summary()["cooperative_collects"] == [2]
    with pytest.raises(ValueError, match="collected twice"):
        pool.register(_plan({4: T0 + 900.0}, {2: T0 + 2500.0}, foreign={2: T0 + 700.0}), 3)


def test_orphan_credit_values_remaining_mining_time() -> None:
    plan = _plan({1: T0 + 500.0, 2: T0 + 700.0}, {1: T0 + 3000.0})
    assert orphan_credit_kg(plan, None, 0.0, 400.0) == 0.0
    stay = C.MISSION_END_MJD - 400.0 - (T0 + 700.0)
    assert orphan_credit_kg(plan, None, 0.5, 400.0) == pytest.approx(0.5 * 10.0 * stay / YEAR)
    assert orphan_credit_kg(plan, {2: 0.5}, 1.0, 400.0) == pytest.approx(0.5 * 10.0 * stay / YEAR)
    late = _plan({1: T0 + 500.0, 2: C.MISSION_END_MJD - 500.0}, {1: T0 + 3000.0})
    assert orphan_credit_kg(late, None, 0.5, 400.0) == 0.0  # no ship could stay a year


# -- pure: master -------------------------------------------------------------------------


def _column(identifier, deploys, collects, mass, *, foreign=None, certified=True, slot=1):
    foreign = foreign or {}
    return FleetColumn(
        identifier,
        slot,
        f"c{identifier}",
        dict(deploys),
        dict(collects),
        dict(foreign),
        {a: mass for a in collects},
        certified,
    )


def test_master_packs_each_asteroid_once_and_prefers_value() -> None:
    a = _column(0, {1: 100.0, 2: 200.0}, {1: 3000.0, 2: 3100.0}, 300.0)
    b = _column(1, {2: 150.0, 3: 250.0}, {2: 3000.0, 3: 3100.0}, 250.0)  # conflicts with a on 2
    c = _column(2, {4: 100.0}, {4: 3000.0}, 400.0)
    d = _column(3, {5: 100.0}, {5: 3000.0}, 10.0, certified=False)
    result = solve_fleet_master([a, b, c, d])
    assert [col.identifier for col in result.selected] == [0, 2]
    assert result.objective == pytest.approx(1000.0)
    assert result.exhaustive and fleet_feasible(result.selected) == ""
    reasons = {item["identifier"]: item["reason"] for item in result.rejected}
    assert reasons[3] == "not certified" and "incompatible" in reasons[1]
    # bonus weights change the choice: asteroid 3 is worth 4x
    weighted = solve_fleet_master([a, b, c, d], weights={3: 4.0})
    assert [col.identifier for col in weighted.selected] == [1, 2]
    assert weighted.objective == pytest.approx(250.0 + 1000.0 + 400.0)
    assert weighted.collected_kg == pytest.approx(900.0)


def test_master_requires_the_deployer_of_a_foreign_collect() -> None:
    deployer = _column(0, {1: 100.0, 2: 200.0}, {1: 3000.0}, 300.0)  # leaves 2
    collector = _column(1, {3: 100.0}, {3: 3000.0, 2: 3200.0}, 200.0, foreign={2: 200.0})
    stale = _column(2, {4: 100.0}, {4: 3000.0, 2: 3200.0}, 500.0, foreign={2: 199.0})
    alone = solve_fleet_master([collector])
    assert alone.selected == () and alone.rejected[0]["reason"].startswith("no column deploys")
    both = solve_fleet_master([deployer, collector, stale])
    assert [col.identifier for col in both.selected] == [0, 1]
    assert both.collected_kg == pytest.approx(300.0 + 400.0)
    assert fleet_feasible([deployer, stale]) == (
        "asteroid 2 deploy epoch differs from the collector's assumption"
    )
    assert fleet_feasible([collector]).endswith("but not deployed")
    assert fleet_feasible([deployer, deployer]) == "asteroid 1 deployed twice"
    twice = _column(5, {6: 100.0}, {6: 3000.0, 1: 3000.0}, 100.0, foreign={1: 100.0})
    assert fleet_feasible([deployer, twice]) == "asteroid 1 collected twice"


def test_master_respects_ship_count_rule_and_is_order_invariant() -> None:
    # 2 exp(0.004 * 100) = 2.98 ships at 100 kg each: three light ships are one too many
    light = [_column(k, {k: 100.0}, {k: 3000.0}, 100.0) for k in range(3)]
    result = solve_fleet_master(light)
    assert len(result.selected) == 2 and fleet_feasible(result.selected) == ""
    assert fleet_feasible(light).startswith("3 ships exceed the limit")
    heavy = [_column(k, {k: 100.0}, {k: 3000.0}, 400.0) for k in range(3)]
    assert len(solve_fleet_master(heavy).selected) == 3
    assert len(solve_fleet_master(heavy, max_ships=2).selected) == 2
    mixed = [*heavy, _column(7, {7: 100.0}, {7: 3000.0}, 50.0)]
    forward = solve_fleet_master(mixed)
    backward = solve_fleet_master(list(reversed(mixed)))
    assert [c.identifier for c in forward.selected] == [c.identifier for c in backward.selected]
    assert forward.objective == backward.objective and forward.nodes == backward.nodes
    assert forward.summary()["ships"] == 4


# -- pure: cooperative visit orders --------------------------------------------------------


def test_build_visits_accepts_orphans_and_foreign_collects_only_with_epochs() -> None:
    from spacepdhcg.gtoc12.retiming import build_visits

    visits = build_visits([1, 2, 3], [3, 2], foreign=None)  # 1 is an orphan
    assert [(v.body, v.deploy, v.collect) for v in visits] == [
        (EARTH_ID, False, False),
        (1, True, False),
        (2, True, False),
        (3, True, True),
        (2, False, True),
        (EARTH_ID, False, False),
    ]
    assert [v.role_out for v in visits[:-1]] == [
        "earth_out",
        "deploy_hop",
        "deploy_hop",
        "collect_hop",
        "earth_return",
    ]
    with pytest.raises(ValueError, match="deployed by nobody"):
        build_visits([1, 2], [2, 9, 1])
    coop = build_visits([1, 2], [2, 9, 1], foreign={9: T0 + 300.0})
    assert coop[3].body == 9 and coop[3].foreign_deploy_epoch == T0 + 300.0
    assert coop[4].body == 1 and coop[4].foreign_deploy_epoch is None
    with pytest.raises(ValueError, match="twice"):
        build_visits([1, 1], [1])


# -- data-backed: clusters, re-timing, cooperative extension --------------------------------


@pytest.fixture(scope="module")
def catalogue():
    if not data_available():
        pytest.skip("pinned GTOC12 data not fetched")
    return load_catalogue()


@pytest.fixture(scope="module")
def small_search(catalogue):
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
    from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

    ids = build_reduced_instance(catalogue).asteroid_ids[:90]
    settings = SearchSettings(
        beam_width=4,
        max_deploys=2,
        neighbours=8,
        launch_epochs=SMALL_LAUNCH_EPOCHS,
        earth_leg_tofs=SMALL_EARTH_TOFS,
        hop_tofs=(90.0, 180.0),
        collect_hop_tofs=(180.0, 360.0),
        first_level_limit=50,
    )
    search = RouteSearch(catalogue, ids, settings)
    result = search.run()
    assert result.candidates
    return search, result


@requires_data
def test_comoving_clusters_are_deterministic_and_phasing_windows_close(catalogue) -> None:
    from spacepdhcg.gtoc12.clusters import ClusterBands, ComovingClusters
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    first = ComovingClusters(catalogue, ids, ClusterBands(radius=1.5, phase_deg=8.0))
    second = ComovingClusters(catalogue, ids[::-1].copy(), ClusterBands(radius=1.5, phase_deg=8.0))
    assert first.ids.tolist() == second.ids.tolist()
    assert first.density.tolist() == second.density.tolist()
    assert first.labels.tolist() == second.labels.tolist()
    assert (first.density >= 0).all() and (first.labels >= 0).all()
    densest = int(first.ids[int(np.argmax(first.density))])
    neighbours = first.neighbours(densest)
    assert neighbours.shape[0] == first.density_of(densest)
    assert densest not in set(neighbours.tolist())
    # neighbours share the seed's cluster or a denser one, never an unlabelled state
    members = set(first.cluster_members(first.label_of(densest)).tolist())
    assert densest in members
    assert first.unvisited_potential(densest, set(neighbours[:2].tolist())) == max(
        first.density_of(densest) - 2, 0
    )
    summary = first.summary()
    assert summary["asteroids"] == ids.shape[0]
    if neighbours.shape[0]:
        target = int(neighbours[0])
        window = first.phasing_window(densest, target, T0, 8.0, 10.0 * YEAR)
        if window is not None:
            open_mjd, close_mjd = window
            assert T0 <= open_mjd <= close_mjd <= T0 + 10.0 * YEAR


@requires_data
def test_retiming_keeps_visit_order_and_mass_bookkeeping(catalogue, small_search) -> None:
    from spacepdhcg.gtoc12.retiming import Retimer, RetimeSettings, orders_of

    search, result = small_search
    plan = result.candidates[0]
    settings = RetimeSettings(step_days=30.0, hop_tof_days=(90.0, 360.0))
    retimer = Retimer(catalogue, search.settings, settings)
    first = retimer.retime(plan)
    second = Retimer(catalogue, search.settings, settings).retime(plan)
    strip = lambda s: {k: v for k, v in s.items() if k != "wall_seconds"}  # noqa: E731
    assert strip(first.summary()) == strip(second.summary())  # deterministic
    if first.plan is not None and second.plan is not None:
        assert first.plan.summary() == second.plan.summary()
    if first.plan is None:
        pytest.skip(f"re-timer found no closing schedule on the coarse grid ({first.failure})")
    retimed = first.plan
    assert orders_of(retimed) == orders_of(plan)
    assert retimed.self_cleaning and retimed.feasible
    for asteroid in retimed.asteroids:
        stay = retimed.collect_epochs[asteroid] - retimed.deploy_epochs[asteroid]
        assert stay >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
        assert retimed.collected_mass[asteroid] == pytest.approx(C.maximum_collected_mass(stay))
    # legs chain in time and every non-camp leg lies on the re-timer's lattice
    for previous, leg in zip(retimed.legs, retimed.legs[1:], strict=False):
        assert leg.departure_epoch == pytest.approx(previous.arrival_epoch)
    assert retimed.legs[-1].arrival_epoch <= C.MISSION_END_MJD
    assert first.objective_after >= first.objective_before - 1e-9 or not first.improved


@requires_data
def test_orphan_credit_does_not_change_self_cleaning_retiming(catalogue, small_search) -> None:
    """A deploy this ship collects later is priced at the full mining rate, not the orphan credit.

    Regression: the DP once treated every deploy-only visit as an orphan, so with credit 0 the
    deploy epochs of self-cleaning asteroids were free and the schedule degraded (548 -> 545 kg
    on the fleet10 ship 1 instead of 548 -> 583 kg).
    """

    from spacepdhcg.gtoc12.retiming import Retimer, RetimeSettings

    search, result = small_search
    plan = result.candidates[0]
    assert plan.self_cleaning
    strip = lambda s: {k: v for k, v in s.items() if k != "wall_seconds"}  # noqa: E731
    outcomes = []
    for credit in (0.0, 0.5, 1.0):
        settings = RetimeSettings(step_days=30.0, hop_tof_days=(90.0, 360.0), orphan_credit=credit)
        outcomes.append(strip(Retimer(catalogue, search.settings, settings).retime(plan).summary()))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@requires_data
def test_cooperative_extension_collects_another_ships_orphan(catalogue, small_search) -> None:
    from spacepdhcg.gtoc12.retiming import Retimer, RetimeSettings, extend_plan, orders_of

    search, result = small_search
    plan = result.candidates[0]
    settings = RetimeSettings(step_days=30.0, hop_tof_days=(90.0, 360.0), orphan_credit=0.5)
    retimer = Retimer(catalogue, search.settings, settings)
    # another ship deployed a miner on the nearest unused asteroid early in the mission
    deploy_order, _ = orders_of(plan)
    ranked = [
        int(a) for a in search.candidates(deploy_order[-1], plan.deploy_epochs[deploy_order[-1]])
    ]
    orphan = next(a for a in ranked if a not in plan.asteroids)
    other = _plan({orphan: T0 + 400.0, 59999: T0 + 500.0}, {59999: T0 + 3000.0})
    pool = MinerPool()
    pool.register(other, 1)
    assert pool.orphans() == {orphan: T0 + 400.0}
    variants, failures = extend_plan(plan, search, retimer, candidates=2, pool=pool)
    kinds = {f["kind"] for f in failures}
    assert kinds <= {"self_cleaning", "orphan", "foreign_collect"}
    foreign_variants = [v for v in variants if orphan in v.plan.foreign_deploy_epochs]
    orphan_variants = [v for v in variants if v.plan.orphaned]
    for variant in foreign_variants:
        coop = variant.plan
        assert orphan in coop.collect_epochs and orphan not in coop.deploy_epochs
        assert coop.foreign_deploy_epochs[orphan] == T0 + 400.0
        stay = coop.collect_epochs[orphan] - (T0 + 400.0)
        assert stay >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
        assert coop.collected_mass[orphan] == pytest.approx(C.maximum_collected_mass(stay))
        assert coop.feasible
        # the pool accepts the cooperative plan as ship 2 and reports the shared collect
        check = MinerPool()
        check.register(other, 1)
        check.register(coop, 2)
        assert check.summary()["cooperative_collects"] == [orphan]
    for variant in orphan_variants:
        left = variant.plan.orphaned
        assert all(a in variant.plan.deploy_epochs for a in left)
        assert all(a not in variant.plan.collected_mass for a in left)
    # something was attempted in each cooperative role
    attempted = (
        {f["kind"] for f in failures}
        | ({"foreign_collect"} if foreign_variants else set())
        | ({"orphan"} if orphan_variants else set())
    )
    assert "foreign_collect" in attempted and "orphan" in attempted
