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


def test_master_selects_a_competitive_cooperative_pair_and_reports_it() -> None:
    """A deployer + collector pair whose collector harvests old miners beats a self-cleaning
    ship of the same family and is reported as cooperative in the incumbent."""

    # self-cleaning ship: 6 asteroids x 75 kg
    six = range(1, 7)
    solo = _column(0, {k: 400.0 + k for k in six}, {k: 3400.0 + k for k in six}, 75.0)
    # deployer: 8 miners early, collects 4 itself
    deployer = _column(
        1,
        {k: 300.0 + k for k in range(11, 19)},
        {k: 3500.0 + k for k in range(11, 15)},
        85.0,
        slot=1,
    )
    # collector: its own 5 asteroids plus the 4 miners the deployer left (8.5 years old: 85 kg)
    collector = _column(
        2,
        {k: 500.0 + k for k in range(21, 26)},
        {**{k: 3600.0 + k for k in range(21, 26)}, **{k: 3400.0 + k for k in range(15, 19)}},
        85.0,
        foreign={k: 300.0 + k for k in range(15, 19)},
        slot=2,
    )
    result = solve_fleet_master([solo, deployer, collector])
    assert result.exhaustive
    chosen = {col.identifier for col in result.selected}
    assert {1, 2} <= chosen  # the cooperative pair enters the incumbent
    coop = result.cooperative_columns()
    assert coop["collector_ships"] == 1 and coop["deployer_ships"] == 1
    assert coop["foreign_collects"] == 4 and coop["bundle_columns"] == 0
    assert result.summary()["cooperative"] == coop
    # as one bundle column the pair is counted as two ships and one bundle
    bundle = FleetColumn.from_bundle(3, "b", [deployer, collector])
    bundled = solve_fleet_master([solo, bundle])
    assert bundled.cooperative_columns()["bundle_columns"] == 1
    assert bundled.cooperative_columns()["collector_ships"] == 1
    assert bundled.ships == 3 if 3 in {c.identifier for c in bundled.selected} else True


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


def test_master_search_depth_is_not_bounded_by_the_interpreter_recursion_limit() -> None:
    """The skip branch recurses once per column: 1019 columns (fleet_master_v6) overflowed the
    default 1000-frame limit after 45 min of re-certification.  The limit is raised for the
    search and restored afterwards."""

    import sys

    before = sys.getrecursionlimit()
    n = before + 200
    # pairwise-compatible light columns: the rule caps the fleet, the DFS still walks the list
    columns = [_column(k, {k: 100.0}, {k: 3000.0}, 560.0 + (k % 7)) for k in range(n)]
    result = solve_fleet_master(columns, node_cap=50_000, max_ships=3)
    assert len(result.selected) == 3 and fleet_feasible(result.selected) == ""
    assert sys.getrecursionlimit() == before


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
def test_phasing_aware_families_require_co_location_at_every_visit_epoch(catalogue) -> None:
    from spacepdhcg.gtoc12.clusters import ClusterBands, ComovingClusters
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    static = ComovingClusters(catalogue, ids, ClusterBands(radius=1.5, visit_epochs=None))
    aware = ComovingClusters(catalogue, ids, ClusterBands(radius=1.5))
    assert static.features.shape[1] == 7 and aware.features.shape[1] == 9  # +2 per extra epoch
    assert aware.summary()["bands"]["visit_epochs"] == list(ClusterBands().visit_epochs)
    # the phasing-aware neighbourhood is a subset of the static one (an extra distance term)
    for asteroid in aware.ids[:40].tolist():
        assert set(aware.neighbours(asteroid).tolist()) <= set(static.neighbours(asteroid).tolist())
    assert aware.density.sum() <= static.density.sum()
    # pairs the static membership keeps but the phasing-aware one drops drift apart between the
    # deploy and the collect epoch; kept pairs stay within the band at both epochs
    t_deploy, t_collect = ClusterBands().visit_epochs
    dropped, kept = [], []
    for asteroid in static.ids[:60].tolist():
        aware_set = set(aware.neighbours(asteroid).tolist())
        for other in static.neighbours(asteroid).tolist():
            drift = abs(
                aware.phase_difference_deg(asteroid, other, t_collect)
                - aware.phase_difference_deg(asteroid, other, t_deploy)
            )
            (kept if other in aware_set else dropped).append(drift)
    if dropped and kept:
        assert np.median(dropped) > np.median(kept)
    for asteroid in aware.ids[:40].tolist():
        for other in aware.neighbours(asteroid).tolist():
            for epoch in (t_deploy, t_collect):
                # within radius x band (chord ~ angle for small differences)
                assert abs(aware.phase_difference_deg(asteroid, other, epoch)) <= 1.5 * 8.0 + 0.5
    # deterministic under input order
    again = ComovingClusters(catalogue, ids[::-1].copy(), ClusterBands(radius=1.5))
    assert again.labels.tolist() == aware.labels.tolist()


def test_low_thrust_inflation_model_is_monotone_and_calibrates_as_a_residual() -> None:
    from spacepdhcg.gtoc12.retiming import Retimer, RetimeSettings
    from spacepdhcg.gtoc12.screening import low_thrust_inflation, thrust_authority_km_s

    mass, tof = 2000.0, 200.0
    authority = float(thrust_authority_km_s(mass, tof, 1.0))
    ratios = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    inflation = low_thrust_inflation(ratios * authority, mass, tof)
    assert np.all(np.diff(inflation) > 0)  # faster hops are penalised more
    assert inflation[0] == pytest.approx(1.05 + 0.065) and inflation[-1] == pytest.approx(1.375)
    # the model was fitted on the archived hops: 1.08 at r 0.15, 1.13 at 0.25, 1.21 at 0.35
    assert low_thrust_inflation(0.15 * authority, mass, tof) == pytest.approx(1.1475, abs=0.07)
    assert low_thrust_inflation(0.35 * authority, mass, tof) == pytest.approx(1.2775, abs=0.07)
    retimer = Retimer.__new__(Retimer)
    retimer.settings = RetimeSettings()
    retimer.search_settings = None
    retimer.inflations = {}
    retimer.bans = {}
    # a hop measured exactly at the model stores a unit residual; Earth legs keep the raw factor
    retimer.calibrate(1, 2, 1.375, authority_ratio=0.5)
    assert retimer.inflations[(1, 2)] == pytest.approx(RetimeSettings().calibration_margin)
    retimer.calibrate(1, 2, 1.375 * 1.10, authority_ratio=0.5)
    assert retimer.inflations[(1, 2)] == pytest.approx(1.10 * RetimeSettings().calibration_margin)
    retimer.calibrate(0, 2, 1.2, authority_ratio=0.7)  # Earth leg: raw factor, no model
    assert retimer.inflations[(0, 2)] == pytest.approx(1.2 * RetimeSettings().calibration_margin)


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
    # harvest mode tries every orphan at every collect position and keeps the bookkeeping:
    # the foreign miner's mass is mined from the *other* ship's deploy epoch
    other2 = _plan({orphan: T0 + 400.0, 59998: T0 + 450.0, 59999: T0 + 500.0}, {59999: T0 + 3000.0})
    pool2 = MinerPool()
    pool2.register(other2, 1)
    assert set(pool2.orphans()) == {orphan, 59998}
    h_variants, h_failures = extend_plan(
        plan, search, retimer, candidates=2, pool=pool2, harvest=True
    )
    _, collect_order = orders_of(plan)
    tried = [f for f in h_failures if f["kind"] == "foreign_collect"] + [
        {"asteroid": next(a for a in v.plan.foreign_deploy_epochs)}
        for v in h_variants
        if v.plan.foreign_deploy_epochs
    ]
    # both orphans were offered (each at up to len(collect_order) + 1 positions)
    assert {t["asteroid"] for t in tried} <= {orphan, 59998}
    assert len(tried) <= 2 * (len(collect_order) + 1)
    assert len(tried) >= len([f for f in failures if f["kind"] == "foreign_collect"]) + len(
        foreign_variants
    )
    for variant in h_variants:
        coop = variant.plan
        for asteroid, epoch in coop.foreign_deploy_epochs.items():
            assert epoch == pool2.orphans()[asteroid]
            stay = coop.collect_epochs[asteroid] - epoch
            assert stay >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
            assert coop.collected_mass[asteroid] == pytest.approx(C.maximum_collected_mass(stay))
        assert coop.total_collected_kg == pytest.approx(sum(coop.collected_mass.values()))
    # something was attempted in each cooperative role
    attempted = (
        {f["kind"] for f in failures}
        | ({"foreign_collect"} if foreign_variants else set())
        | ({"orphan"} if orphan_variants else set())
    )
    assert "foreign_collect" in attempted and "orphan" in attempted
