"""Cooperative cluster pricing: bundle columns, the bundle master, Earth-leg seeding, pricing."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.cooperative import (
    FleetColumn,
    MinerPool,
    fleet_feasible,
    greedy_fleet,
    ship_count,
    ship_rule_bound,
    solve_fleet_master,
)
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.pipeline import RefinedRoute
from spacepdhcg.gtoc12.screening import exhaust_velocity_km_s
from spacepdhcg.gtoc12.search import EARTH_ID, EarthLeg, PlannedLeg, RoutePlan, RouteSearch

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
T0 = C.MISSION_START_MJD
YEAR = C.YEAR_DAYS
SMALL_LAUNCH_EPOCHS = tuple(float(x) for x in T0 + np.arange(0.0, 731.0, 90.0))
SMALL_EARTH_TOFS = (500.0, 650.0, 800.0)


def _column(identifier, deploys, collects, mass, *, foreign=None, certified=True, slot=1):
    return FleetColumn(
        identifier,
        slot,
        f"c{identifier}",
        dict(deploys),
        dict(collects),
        dict(foreign or {}),
        {a: mass for a in collects},
        certified,
    )


# -- pure: bundle columns and the bundle master -----------------------------------------------


def test_bundle_column_aggregates_members_and_resolves_internal_foreign_collects() -> None:
    deployer = _column(0, {1: 100.0, 2: 200.0}, {1: 3000.0}, 300.0)  # leaves 2
    collector = _column(
        1, {3: 100.0}, {3: 3000.0, 2: 3200.0, 9: 3300.0}, 200.0, foreign={2: 200.0, 9: 50.0}
    )
    bundle = FleetColumn.from_bundle(7, "f0", [deployer, collector])
    assert bundle.ships == 2 and bundle.identifier == 7 and bundle.certified
    assert sorted(bundle.deploys) == [1, 2, 3] and sorted(bundle.collects) == [1, 2, 3, 9]
    assert bundle.foreign == {9: 50.0}  # 2 is deployed inside the bundle, 9 is not
    assert bundle.collected_kg == pytest.approx(300.0 + 3 * 200.0)
    assert bundle.summary()["members"] == ["c0", "c1"] and bundle.summary()["ships"] == 2
    assert bundle.routes() == [None, None] and ship_count([bundle, deployer]) == 3
    with pytest.raises(ValueError, match="deploys asteroid 1 twice"):
        FleetColumn.from_bundle(8, "bad", [deployer, _column(2, {1: 5.0}, {1: 3000.0}, 1.0)])
    with pytest.raises(ValueError, match="collects asteroid 3 twice"):
        FleetColumn.from_bundle(
            9,
            "bad",
            [collector, _column(2, {4: 5.0}, {4: 3000.0, 3: 3100.0}, 1.0, foreign={3: 1.0})],
        )


def test_master_counts_bundle_ships_in_the_fleet_rule_and_ship_cap() -> None:
    # 2 exp(0.004 * 100) = 2.98: a three-ship bundle of 100 kg ships breaks the rule alone
    light = FleetColumn.from_bundle(
        0, "light", [_column(k, {k: 100.0}, {k: 3000.0}, 100.0) for k in range(3)]
    )
    assert fleet_feasible([light]).startswith("3 ships exceed the limit")
    assert solve_fleet_master([light]).selected == ()
    heavy = FleetColumn.from_bundle(
        1, "heavy", [_column(10 + k, {10 + k: 100.0}, {10 + k: 3000.0}, 400.0) for k in range(3)]
    )
    single = _column(20, {20: 100.0}, {20: 3000.0}, 450.0)
    result = solve_fleet_master([heavy, single])
    assert [c.identifier for c in result.selected] == [1, 20]
    assert result.ships == 4 and result.summary()["columns"] == 2
    assert result.collected_kg == pytest.approx(1200.0 + 450.0)
    assert fleet_feasible(result.selected) == ""
    capped = solve_fleet_master([heavy, single], max_ships=3)
    assert [c.identifier for c in capped.selected] == [1]  # 3 ships x 400 beats 1 x 450
    assert solve_fleet_master([heavy, single], max_ships=2).selected == (single,)
    assert len(result.routes()) == 4


def test_greedy_fleet_iterates_on_the_mean_mass_and_matches_the_exact_master() -> None:
    # value order: the greedy takes every compatible column, then drops the lightest per ship
    # until 2 exp(0.004 mean) admits the count (the fixed-point iteration on the mean)
    columns = [
        _column(0, {1: 1.0}, {1: 3000.0}, 600.0),
        _column(1, {2: 1.0}, {2: 3000.0}, 500.0),
        FleetColumn.from_bundle(
            2,
            "b",
            [_column(10, {3: 1.0}, {3: 3000.0}, 90.0), _column(11, {4: 1.0}, {4: 3000.0}, 80.0)],
        ),
        _column(3, {5: 1.0}, {5: 3000.0}, 60.0),
    ]
    usable = sorted(columns, key=lambda c: (-c.collected_kg, c.identifier))
    greedy = greedy_fleet(usable, 100)
    assert fleet_feasible(greedy) == ""
    # 600 + 500 alone admit 2 exp(2.2) = 18 ships; adding the 85 kg bundle and the 60 kg ship
    # pulls the mean to 266 kg (limit 5.8 >= 5 ships), so the greedy keeps everything
    assert ship_count(greedy) == 5
    exact = solve_fleet_master(columns)
    assert exact.objective >= sum(c.collected_kg for c in greedy) - 1e-9
    assert exact.greedy_objective == pytest.approx(sum(c.collected_kg for c in greedy))
    assert fleet_feasible(exact.selected) == ""
    # a bundle whose foreign collect nobody supplies is never chosen
    stranded = FleetColumn.from_bundle(
        4,
        "s",
        [
            _column(12, {6: 1.0}, {6: 3000.0}, 700.0),
            _column(13, {7: 1.0}, {7: 3000.0, 8: 3000.0}, 700.0, foreign={8: 2.0}),
        ],
    )
    result = solve_fleet_master([*columns, stranded])
    assert stranded not in result.selected
    assert any(r["identifier"] == 4 and "foreign" in r["reason"] for r in result.rejected)
    # order invariance
    backward = solve_fleet_master(list(reversed(columns)))
    assert [c.identifier for c in backward.selected] == [c.identifier for c in exact.selected]


# -- data-backed: Earth-leg seeding, first level injection, pricing --------------------------


@pytest.fixture(scope="module")
def catalogue():
    if not data_available():
        pytest.skip("pinned GTOC12 data not fetched")
    return load_catalogue()


def _proxy_refine(plan: RoutePlan) -> RefinedRoute:
    """Stand-in for SCvx refinement that trusts the proxy plan (tests only)."""

    return RefinedRoute(
        plan=plan,
        legs=[],
        collected_mass=dict(plan.collected_mass),
        final_mass_kg=plan.final_mass_proxy_kg,
        certified=plan.feasible,
        master_certified=plan.feasible,
        passes=1,
        wall_seconds=0.0,
        scheduler_telemetry={},
        failures=[] if plan.feasible else [{"reason": "proxy infeasible"}],
    )


def _proxy_certify(catalogue, target, launch, tof, lambert_dv, scvx):
    """Stand-in for the single-leg SCvx check: odd targets fail, even ones cost 0.9x Lambert."""

    if target % 2:
        return None
    from spacepdhcg.gtoc12.screening import propellant_for_delta_v

    return EarthLeg(
        target, launch, tof, lambert_dv, float(propellant_for_delta_v(3000.0, 0.9 * lambert_dv))
    )


@pytest.fixture(scope="module")
def family(catalogue):
    from spacepdhcg.gtoc12.bundles import family_clusters
    from spacepdhcg.gtoc12.clusters import ClusterBands
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    families = family_clusters(
        catalogue, ids, bands=ClusterBands(radius=2.0, phase_deg=12.0), min_members=8
    )
    assert families and families[0][1].shape[0] >= 8
    assert [len(m) for _, m in families] == sorted((len(m) for _, m in families), reverse=True)
    return families[0]


@requires_data
def test_certify_earth_legs_ranks_dedups_caches_and_logs_rejects(catalogue, family) -> None:
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        certify_earth_legs,
        cluster_search_settings,
    )

    _label, members = family
    settings = cluster_search_settings(
        ClusterPricingSettings(launch_epochs=SMALL_LAUNCH_EPOCHS, earth_leg_tofs=SMALL_EARTH_TOFS),
        members.shape[0],
    )
    cache: dict = {}
    legs, rejected = certify_earth_legs(
        catalogue,
        members,
        settings,
        count=2,
        max_checks=6,
        cache=cache,
        certify=_proxy_certify,
        continuous=False,
    )
    assert 1 <= len(legs) <= 2 and len(cache) <= 6
    assert all(leg.target % 2 == 0 and leg.target in set(members.tolist()) for leg in legs)
    assert all(r["target"] % 2 == 1 for r in rejected)
    assert len({(leg.target, leg.launch_epoch) for leg in legs}) == len(legs)  # distinct
    for leg in legs:
        assert leg.launch_epoch in SMALL_LAUNCH_EPOCHS and leg.tof_days in SMALL_EARTH_TOFS
        assert 0.0 < leg.propellant_kg < 3000.0 - C.DRY_MASS_KG
    # the cache means a second call flies nothing new and returns the same legs
    again, _ = certify_earth_legs(
        catalogue,
        members,
        settings,
        count=2,
        max_checks=0,
        cache=cache,
        certify=_proxy_certify,
        continuous=False,
    )
    assert again == legs


def _tof_priced_certify(catalogue, target, launch, tof, lambert_dv, scvx):
    """Stand-in SCvx whose Earth leg gets cheaper with TOF (as the real ones do) and whose
    launch window is bounded: launches beyond +400 d of the start fail."""

    if target % 2 or launch > C.MISSION_START_MJD + 400.0:
        return None
    propellant = max(600.0 - 0.8 * (tof - 300.0), 150.0)
    lambert = lambert_dv if lambert_dv == lambert_dv else 5.0
    return EarthLeg(target, launch, tof, lambert, propellant)


@requires_data
def test_continuous_earth_leg_refinement_respects_bounds_and_is_deterministic(
    catalogue, family
) -> None:
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        certify_earth_legs,
        cluster_search_settings,
    )
    from spacepdhcg.gtoc12.earthleg import EarthLegBounds, refine_leg_scvx

    _label, members = family
    settings = cluster_search_settings(
        ClusterPricingSettings(launch_epochs=SMALL_LAUNCH_EPOCHS, earth_leg_tofs=SMALL_EARTH_TOFS),
        members.shape[0],
    )
    runs = []
    for _ in range(2):
        log: list = []
        legs, _rejected = certify_earth_legs(
            catalogue,
            members,
            settings,
            count=2,
            max_checks=6,
            cache={},
            certify=_tof_priced_certify,
            continuous=True,
            continuous_evaluations=10,
            optimiser_log=log,
        )
        runs.append(([(leg.target, leg.launch_epoch, leg.tof_days) for leg in legs], log))
    assert runs[0] == runs[1]  # deterministic
    legs_key, log = runs[0]
    assert legs_key and log
    radius = 0.5 * min(np.diff(np.asarray(SMALL_LAUNCH_EPOCHS)))
    for record in log:
        # the refined leg stays inside the grid window and the TOF band, and never costs more
        assert record["saved_kg"] >= -1e-9
        assert record["scvx_evaluations"] <= 10
        assert min(SMALL_EARTH_TOFS) <= record["tof_days"] <= max(SMALL_EARTH_TOFS)
        assert min(SMALL_LAUNCH_EPOCHS) <= record["launch_epoch"] <= max(SMALL_LAUNCH_EPOCHS)
        grid_launches = np.asarray(SMALL_LAUNCH_EPOCHS)
        assert np.min(np.abs(grid_launches - record["launch_epoch"])) <= radius + 1e-9
        # the stand-in rewards longer legs: the search lengthened the TOF unless already at the top
        assert record["tof_days"] >= record["trace"][0]["tof_days"] - 2 * 30.0 or (
            record["tof_days"] == max(SMALL_EARTH_TOFS)
        )
    # legs whose launch moved keep the official window: the stand-in refuses launches > +400 d
    for _target, launch, tof in legs_key:
        assert launch <= C.MISSION_START_MJD + 400.0
        assert tof == round(tof) and round(launch, 1) == launch
    # direct call: a refinement that cannot improve returns the start leg unchanged
    start = EarthLeg(legs_key[0][0], legs_key[0][1], max(SMALL_EARTH_TOFS), 5.0, 150.0)
    bounds = EarthLegBounds(
        launch_min=min(SMALL_LAUNCH_EPOCHS),
        launch_max=max(SMALL_LAUNCH_EPOCHS),
        tof_min=min(SMALL_EARTH_TOFS),
        tof_max=max(SMALL_EARTH_TOFS),
    )
    result = refine_leg_scvx(
        catalogue, start, _tof_priced_certify, None, bounds=bounds, max_evaluations=6
    )
    assert result.leg == start and result.saved_kg == 0.0 and result.evaluations <= 6


@requires_data
def test_injected_first_level_seeds_the_beam_at_the_certified_mass(catalogue, family) -> None:
    from spacepdhcg.gtoc12.search import SearchSettings

    _label, members = family
    settings = SearchSettings(
        beam_width=4,
        max_deploys=2,
        neighbours=8,
        launch_epochs=SMALL_LAUNCH_EPOCHS,
        earth_leg_tofs=SMALL_EARTH_TOFS,
        hop_tofs=(90.0, 180.0),
        collect_hop_tofs=(180.0, 360.0),
    )
    target = int(members[0])
    legs = [
        EarthLeg(target, T0 + 90.0, 650.0, 6.0, 450.0),
        EarthLeg(99_999, T0 + 90.0, 650.0, 6.0, 450.0),  # not in the pool: skipped
    ]
    # window 0: exactly the certified legs, at their measured propellant, no grid screening
    exact = RouteSearch(
        catalogue, members, replace(settings, first_level_window_days=0.0), first_level=legs
    )
    first = exact._first_level()
    assert [p.location for p in first] == [target]
    assert first[0].mass == pytest.approx(3000.0 - 450.0 - C.MINER_MASS_KG)
    assert first[0].legs[0].departure_epoch == T0 + 90.0
    assert first[0].legs[0].arrival_epoch == T0 + 740.0
    assert exact.lambert_evaluations == 0
    # a window unlocks the Lambert grid for the certified target only, priced at the measured
    # Delta-V / Lambert ratio, with the certified leg itself still first-class
    search = RouteSearch(catalogue, members, settings, first_level=legs)
    widened = search._first_level()
    assert search.lambert_evaluations > 0
    assert {p.location for p in widened} == {target}
    assert len(widened) >= len(first)
    keys = {(p.legs[0].departure_epoch, p.legs[0].arrival_epoch) for p in widened}
    assert (T0 + 90.0, T0 + 740.0) in keys and len(keys) == len(widened)  # no duplicates
    true_dv = exhaust_velocity_km_s() * math.log(3000.0 / (3000.0 - 450.0))
    for p in widened:
        if p.legs[0].departure_epoch == T0 + 90.0 and p.legs[0].tof_days == 650.0:
            continue
        assert p.legs[0].inflation == pytest.approx(true_dv / 6.0)
        assert abs(p.legs[0].departure_epoch - (T0 + 90.0)) <= settings.first_level_window_days
        assert abs(p.legs[0].tof_days - 650.0) <= settings.first_level_window_days


PRICING_SETTINGS = dict(
    ships=2,
    earth_legs_per_ship=2,
    earth_leg_checks=8,
    beam_width=4,
    max_deploys=3,
    neighbours=8,
    refine_top=2,
    retime_attempts=1,
    retime_rounds=1,
    retime_step_days=30.0,
    launch_epochs=SMALL_LAUNCH_EPOCHS,
    earth_leg_tofs=SMALL_EARTH_TOFS,
)


def _price(catalogue, family):
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, price_cluster

    label, members = family
    return price_cluster(
        catalogue,
        members,
        label=label,
        settings=ClusterPricingSettings(**PRICING_SETTINGS),
        certify_earth=_proxy_certify,
        refine=_proxy_refine,
    )


@pytest.fixture(scope="module")
def bundle(catalogue, family):
    return _price(catalogue, family)


@requires_data
def test_price_cluster_builds_a_consistent_orphan_free_bundle(catalogue, family, bundle) -> None:
    label, members = family
    summary = bundle.summary()
    assert summary["label"] == label and summary["members"] == members.shape[0]
    if not bundle.ships:
        pytest.skip(f"no chain closes in this family on the coarse grids: {summary['rejected']}")
    allowed = set(members.tolist())
    pool = MinerPool()  # raises on any double deploy / collect / stale epoch
    for ship in bundle.ships:
        plan = ship.route.plan
        assert ship.route.certified and plan.feasible
        assert set(plan.asteroids) <= allowed  # cluster co-motion: every visit is in the family
        pool.register(plan, ship.slot)
        for asteroid, epoch in plan.collect_epochs.items():
            stay = epoch - plan.deploy_epoch_of(asteroid)
            assert stay >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
            assert plan.collected_mass[asteroid] == pytest.approx(C.maximum_collected_mass(stay))
        assert plan.final_mass_proxy_kg >= C.DRY_MASS_KG + plan.total_collected_kg - 1e-6
        assert plan.legs[-1].arrival_epoch <= C.MISSION_END_MJD
    stats = summary["cooperative"]
    assert stats["orphans_left"] == [] and pool.orphans() == {}
    assert stats["asteroids"] == len(pool.deployed)
    assert stats["collected_kg"] == pytest.approx(sum(r.total_collected_kg for r in bundle.routes))
    # a foreign collect quotes exactly the deployer's epoch (mass/time bookkeeping across ships)
    for ship in bundle.ships:
        for asteroid, epoch in ship.route.plan.foreign_deploy_epochs.items():
            assert pool.deployed[asteroid][0] == pytest.approx(epoch)
            assert pool.deployed[asteroid][1] != ship.slot
    # determinism: the same family prices to the same bundle
    strip = lambda s: {  # noqa: E731
        k: v for k, v in s.items() if k not in ("wall_seconds", "peak_rss_mb")
    }
    second = _price(catalogue, family).summary()
    for ship_a, ship_b in zip(summary["ships"], second["ships"], strict=True):
        assert ship_a["asteroids"] == ship_b["asteroids"]
        assert ship_a["collected_kg"] == pytest.approx(ship_b["collected_kg"])
    assert strip(summary)["cooperative"] == strip(second)["cooperative"]
    # the bundle offers one multi-ship column whose internal foreign collects are resolved
    columns = [
        FleetColumn.from_plan(
            k, s.slot, f"s{s.slot}", s.route.plan, s.route.collected_mass, certified=True
        )
        for k, s in enumerate(bundle.ships)
    ]
    if len(columns) > 1:
        bundle_column = FleetColumn.from_bundle(99, "bundle", columns)
        assert bundle_column.foreign == {} and bundle_column.ships == len(columns)
        assert fleet_feasible([bundle_column]) == ""


@requires_data
def test_drop_asteroid_removes_one_visit_and_keeps_the_rest_in_order(
    catalogue, family, bundle
) -> None:
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
        drop_asteroid,
        profile_for_orders,
    )
    from spacepdhcg.gtoc12.retiming import Retimer, build_visits, orders_of

    plan = next((s.route.plan for s in bundle.ships if len(s.route.plan.deploy_epochs) >= 2), None)
    if plan is None:
        pytest.skip("no two-asteroid ship in the priced bundle")
    pricing = ClusterPricingSettings(**PRICING_SETTINGS)
    retimer = Retimer(
        catalogue,
        cluster_search_settings(pricing, family[1].shape[0]),
        cluster_retime_settings(pricing, last=True),
    )
    deploy_order, collect_order = orders_of(plan)
    victim = deploy_order[0]
    new_deploy = [a for a in deploy_order if a != victim]
    new_collect = [a for a in collect_order if a != victim]
    profile = profile_for_orders(plan, retimer, new_deploy, new_collect, None)
    assert len(profile) == len(build_visits(new_deploy, new_collect)) - 1
    assert all(m > C.DRY_MASS_KG for m in profile)
    dropped = drop_asteroid(plan, victim, retimer)
    if dropped is None:
        pytest.skip("the shortened order does not close on the coarse lattice")
    assert victim not in dropped.asteroids
    assert orders_of(dropped) == (new_deploy, new_collect)
    assert dropped.self_cleaning and dropped.feasible
    assert drop_asteroid(plan, 123456, retimer) is None


def _plan(first_target: int, launch: float, tof: float, hop_to: int) -> RoutePlan:
    """A two-asteroid self-cleaning plan skeleton (epochs only need to be ordered)."""

    arrive = launch + tof
    legs = [
        PlannedLeg(EARTH_ID, first_target, launch, arrive, 6.0, 1.0, "earth_out"),
        PlannedLeg(first_target, hop_to, arrive + 30.0, arrive + 200.0, 1.0, 1.2, "deploy_hop"),
        PlannedLeg(hop_to, first_target, arrive + 3000.0, arrive + 3200.0, 1.0, 1.2, "collect_hop"),
        PlannedLeg(
            first_target, EARTH_ID, arrive + 3300.0, arrive + 3800.0, 6.0, 1.0, "earth_return"
        ),
    ]
    return RoutePlan(
        tuple(legs),
        {first_target: arrive, hop_to: arrive + 200.0},
        {hop_to: arrive + 3000.0, first_target: arrive + 3300.0},
        {first_target: 300.0, hop_to: 250.0},
        550.0,
        1200.0,
    )


def test_refine_candidates_keeps_one_chain_per_earth_leg_with_certified_legs_first() -> None:
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, refine_candidates

    grid_a = _plan(10, T0, 500.0, 11)
    grid_a_dup = _plan(10, T0, 500.0, 12)  # same Earth leg as grid_a: skipped
    exact = _plan(20, T0 + 90.0, 550.0, 21)
    grid_b = _plan(30, T0, 600.0, 31)
    certified = {(20, T0 + 90.0, 550.0)}
    chosen = refine_candidates(
        [grid_a, grid_a_dup, exact, grid_b], certified, ClusterPricingSettings(refine_top=3)
    )
    assert chosen == [exact, grid_a, grid_b]
    assert refine_candidates([grid_a, grid_a_dup], certified, ClusterPricingSettings()) == [grid_a]


def test_ban_failed_legs_records_earth_legs_and_hop_pairs() -> None:
    from spacepdhcg.gtoc12.bundles import ban_failed_legs

    plan = _plan(10, T0, 500.0, 11)
    pairs: set = set()
    earth: set = set()
    ban_failed_legs(plan, [{"leg": 0}, {"leg": 2}, {"leg": 3}, {"leg": 99}, {}], pairs, earth)
    assert earth == {(10, T0, 500.0)}
    assert pairs == {(11, 10)}  # the Earth return (leg 3) is never banned as a pair


@requires_data
def test_search_bans_exclude_hops_and_seed_legs(catalogue, family) -> None:
    from spacepdhcg.gtoc12.search import SearchSettings

    _label, members = family
    settings = SearchSettings(
        beam_width=4,
        max_deploys=2,
        neighbours=8,
        launch_epochs=SMALL_LAUNCH_EPOCHS,
        earth_leg_tofs=SMALL_EARTH_TOFS,
        hop_tofs=(90.0, 180.0),
        collect_hop_tofs=(180.0, 360.0),
        first_level_window_days=0.0,
    )
    target = int(members[0])
    legs = [
        EarthLeg(target, T0 + 90.0, 650.0, 6.0, 450.0),
        EarthLeg(target, T0 + 180.0, 650.0, 6.0, 460.0),
    ]
    search = RouteSearch(catalogue, members, settings, first_level=legs)
    assert len(search._first_level()) == 2
    search.banned_earth.add((target, T0 + 90.0, 650.0))
    first = search._first_level()
    assert [p.legs[0].departure_epoch for p in first] == [T0 + 180.0]
    # a banned pair never appears as a deploy hop from its source
    children = search._expand(first[0])
    if children:
        victim = children[0].location
        search.banned_pairs.add((target, victim))
        assert all(c.location != victim for c in search._expand(first[0]))
        assert search._best_collect_hop(target, victim, T0 + 4000.0, 1500.0)[1] is None


@requires_data
def test_rank_families_is_deterministic_and_prefers_cheap_access(catalogue) -> None:
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_search_settings,
        family_clusters,
        rank_families,
    )
    from spacepdhcg.gtoc12.clusters import ClusterBands
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    families = family_clusters(
        catalogue, ids, bands=ClusterBands(radius=2.0, phase_deg=12.0), min_members=8
    )[:4]
    settings = cluster_search_settings(
        ClusterPricingSettings(launch_epochs=SMALL_LAUNCH_EPOCHS, earth_leg_tofs=SMALL_EARTH_TOFS),
        2,
    )
    ranked = rank_families(catalogue, families, settings)
    again = rank_families(catalogue, list(reversed(families)), settings)
    assert [label for label, _, _ in ranked] == [label for label, _, _ in again]
    scores = [stats["score"] for _, _, stats in ranked]
    assert scores == sorted(scores)
    for _label, members, stats in ranked:
        assert stats["members"] == members.shape[0]
        assert stats["score"] == pytest.approx(stats["earth_leg_kg"] + 4.0 * stats["hop_kg"])
        assert 0.0 < stats["earth_leg_kg"] < 3000.0 and 0.0 < stats["hop_kg"] < 2000.0


def test_orphan_repair_keeps_the_better_of_dropped_and_reverted(monkeypatch) -> None:
    """A re-timed variant that speculated on an orphan must not beat the plain chain after the
    orphan is dropped: the repair compares the dropped route with the clean variants."""

    from spacepdhcg.gtoc12 import bundles
    from spacepdhcg.gtoc12.bundles import BundleShip, ClusterBundle, _repair_orphans

    launch = T0 + 100.0
    arrive = launch + 500.0
    # deployer plan: deploys 5 and 6, collects only 6 (5 is the orphan)
    legs = [
        PlannedLeg(EARTH_ID, 5, launch, arrive, 6.0, 1.0, "earth_out"),
        PlannedLeg(5, 6, arrive + 30.0, arrive + 200.0, 1.0, 1.2, "deploy_hop"),
        PlannedLeg(6, EARTH_ID, arrive + 3300.0, arrive + 3800.0, 6.0, 1.0, "earth_return"),
    ]
    speculative = RoutePlan(
        tuple(legs), {5: arrive, 6: arrive + 200.0}, {6: arrive + 3300.0}, {6: 400.0}, 500.0, 1200.0
    )
    clean = _plan(6, launch, 500.0, 7)  # a certified variant without asteroid 5: 550 kg
    dropped_plan = RoutePlan(
        tuple(legs[1:]), {6: arrive + 200.0}, {6: arrive + 3300.0}, {6: 380.0}, 380.0, 1200.0
    )
    speculative_route = _proxy_refine(speculative)
    clean_route = _proxy_refine(clean)
    monkeypatch.setattr(bundles, "drop_asteroid", lambda plan, asteroid, retimer: dropped_plan)
    # no ship can take the orphan as a foreign collect
    monkeypatch.setattr(bundles, "extend_plan", lambda *args, **kwargs: ([], []))
    ship = BundleShip(1, speculative_route, [clean_route, speculative_route], {})
    bundle = ClusterBundle(0, (5, 6, 7), [ship])
    pool = MinerPool()
    pool.register(speculative, 1)
    assert pool.orphans() == {5: arrive}
    _repair_orphans(bundle, pool, {1: object()}, {1: object()}, _proxy_refine)
    assert ship.route is clean_route  # 550 kg beats the 380 kg dropped route
    assert bundle.repairs[-1]["kind"] == "reverted"
    assert 5 not in ship.route.plan.deploy_epochs and not ship.route.plan.orphaned


def test_pool_register_all_orders_deployers_before_collectors_and_bundle_consistency() -> None:
    from spacepdhcg.gtoc12.bundles import BundleShip, ClusterBundle

    launch = T0 + 100.0
    arrive = launch + 500.0
    # ship 1 collects 9, which ship 2 deploys (a repaired bundle can look like this)
    collector_legs = (
        PlannedLeg(EARTH_ID, 5, launch, arrive, 6.0, 1.0, "earth_out"),
        PlannedLeg(5, 9, arrive + 3000.0, arrive + 3200.0, 1.0, 1.2, "collect_hop"),
        PlannedLeg(9, EARTH_ID, arrive + 3300.0, arrive + 3800.0, 6.0, 1.0, "earth_return"),
    )
    collector = RoutePlan(
        collector_legs,
        {5: arrive},
        {5: arrive + 3000.0, 9: arrive + 3300.0},
        {5: 300.0, 9: 200.0},
        500.0,
        1200.0,
        foreign_deploy_epochs={9: arrive + 400.0},
    )
    deployer_legs = (
        PlannedLeg(EARTH_ID, 9, launch, arrive + 400.0, 6.0, 1.0, "earth_out"),
        PlannedLeg(9, 8, arrive + 430.0, arrive + 600.0, 1.0, 1.2, "deploy_hop"),
        PlannedLeg(8, EARTH_ID, arrive + 3300.0, arrive + 3800.0, 6.0, 1.0, "earth_return"),
    )
    deployer = RoutePlan(
        deployer_legs,
        {9: arrive + 400.0, 8: arrive + 600.0},
        {8: arrive + 3300.0},
        {8: 300.0},
        500.0,
        1200.0,
    )
    pool = MinerPool()
    with pytest.raises(ValueError, match="never deployed"):
        pool.register(collector, 1)
    pool = MinerPool()
    pool.register_all([(collector, 1), (deployer, 2)])
    assert pool.collected[9] == 1 and pool.deployed[9][1] == 2 and pool.orphans() == {}
    bundle = ClusterBundle(
        0,
        (5, 8, 9),
        [BundleShip(1, _proxy_refine(collector)), BundleShip(2, _proxy_refine(deployer))],
    )
    assert bundle.consistent() == ""
    assert bundle.cooperative_statistics()["cooperative_collects"] == 1
    # without the deployer the collector is stranded and the bundle says so
    stranded = ClusterBundle(0, (5, 9), [BundleShip(1, _proxy_refine(collector))])
    assert "never deployed" in stranded.consistent()

    # make_consistent gives up ships, not the bundle: a stale foreign epoch (the deployer was
    # re-timed by 15 days) strands the collector, which reverts to its clean variant when it
    # has one and leaves otherwise; two deployers of the same asteroid lose the lighter one
    from spacepdhcg.gtoc12.bundles import make_consistent

    moved = replace(deployer, deploy_epochs={9: arrive + 415.0, 8: arrive + 600.0})
    clean_plan = replace(
        collector,
        collect_epochs={5: arrive + 3000.0},
        collected_mass={5: 300.0},
        foreign_deploy_epochs={},
    )
    clean = _proxy_refine(clean_plan)
    ship1 = BundleShip(1, _proxy_refine(collector), [clean])
    bundle = ClusterBundle(0, (5, 8, 9), [ship1, BundleShip(2, _proxy_refine(moved))])
    assert "stale" in bundle.consistent()
    make_consistent(bundle)
    assert bundle.consistent() == "" and [s.slot for s in bundle.ships] == [1, 2]
    assert ship1.route is clean and bundle.repairs[-1]["kind"] == "reverted_stranded"
    no_variant = ClusterBundle(
        0, (5, 8, 9), [BundleShip(1, _proxy_refine(collector)), BundleShip(2, _proxy_refine(moved))]
    )
    make_consistent(no_variant)
    assert [s.slot for s in no_variant.ships] == [2]
    assert no_variant.repairs[-1]["kind"] == "removed_stranded"
    twice = ClusterBundle(
        0,
        (8, 9),
        [BundleShip(2, _proxy_refine(deployer)), BundleShip(3, _proxy_refine(moved))],
    )
    assert "deployed twice" in twice.consistent()
    make_consistent(twice)
    assert twice.consistent() == "" and [s.slot for s in twice.ships] == [
        3
    ]  # tie -> lower slot goes
    assert twice.repairs[-1]["kind"] == "removed_conflict" and not twice.rejected


def test_retimer_reports_an_invalid_visit_order_instead_of_raising() -> None:
    from spacepdhcg.gtoc12.retiming import Retimer

    class _Catalogue:  # the order check happens before any ephemeris is touched
        pass

    retimer = Retimer(_Catalogue())  # type: ignore[arg-type]
    result = retimer.retime_order([5, 5], [5], [3000.0, 2900.0, 2800.0])
    assert result.plan is None and result.failure.startswith("invalid_order")
    result = retimer.retime_order([5], [7], [3000.0, 2900.0])
    assert result.plan is None and "deployed by nobody" in result.failure


def test_earth_leg_record_arrival() -> None:
    leg = EarthLeg(5, T0, 500.0, 6.0, 450.0)
    assert leg.arrival_epoch == T0 + 500.0 and leg.certified
    planned = PlannedLeg(EARTH_ID, 5, T0, T0 + 500.0, 6.0, 1.0, "earth_out")
    assert planned.tof_days == leg.tof_days


# -- master: ship-rule bound, warm start, brute-force agreement --------------------------------


def test_ship_rule_bound_is_a_valid_and_tighter_relaxation() -> None:
    # 3 ships x 100 kg break the rule (limit 2.98) whatever is added from 100 kg units
    prefix = [0.0, 100.0, 200.0, 300.0]
    assert ship_rule_bound(0, 0.0, 0.0, prefix, prefix, 3) == pytest.approx(200.0)
    # a heavy start admits everything: bound = value + every unit
    assert ship_rule_bound(1, 600.0, 600.0, prefix, prefix, 3) == pytest.approx(900.0)
    # the value prefix (bonus-weighted) is what is added, the mass prefix decides feasibility
    assert ship_rule_bound(1, 600.0, 500.0, prefix, [0.0, 50.0, 100.0, 150.0], 3) == 650.0
    # a partial fleet that already breaks the rule and cannot be rescued is pruned
    assert ship_rule_bound(5, 100.0, 100.0, [0.0], [0.0], 3) == -math.inf
    assert ship_rule_bound(5, 100.0, 100.0, prefix, prefix, 0) == -math.inf


def _brute_force(columns, max_ships=100):
    from itertools import combinations

    best_value = 0.0
    for size in range(1, len(columns) + 1):
        for subset in combinations(columns, size):
            if ship_count(subset) <= max_ships and fleet_feasible(subset) == "":
                best_value = max(best_value, sum(c.collected_kg for c in subset))
    return best_value


def test_master_matches_brute_force_on_random_instances_and_uses_the_bound() -> None:
    rng = np.random.default_rng(12)
    for trial in range(25):
        columns = []
        asteroid = 1
        for identifier in range(int(rng.integers(4, 9))):
            deploys = {asteroid + k: 100.0 for k in range(int(rng.integers(1, 3)))}
            asteroid += len(deploys)
            if rng.random() < 0.3 and columns:  # share an asteroid with an earlier column
                shared = next(iter(columns[-1].deploys))
                deploys[shared] = columns[-1].deploys[shared]
            mass = float(rng.integers(60, 700))
            columns.append(_column(identifier, deploys, {a: 3000.0 for a in deploys}, mass))
        if trial % 5 == 0:  # a two-ship bundle
            members = [
                _column(90, {900: 1.0}, {900: 3000.0}, float(rng.integers(100, 500))),
                _column(91, {901: 1.0}, {901: 3000.0}, float(rng.integers(100, 500))),
            ]
            columns.append(FleetColumn.from_bundle(92, "b", members))
        result = solve_fleet_master(columns)
        assert result.exhaustive
        assert result.objective == pytest.approx(_brute_force(columns)), trial
        assert fleet_feasible(result.selected) == ""
    # determinism: identical calls give identical selections
    a = solve_fleet_master(columns).selected
    b = solve_fleet_master(columns).selected
    assert [c.identifier for c in a] == [c.identifier for c in b]


def test_master_warm_start_never_regresses_when_columns_are_added() -> None:
    # both greedy orders take the 600 kg column (2 x 300) and block the 350 + 350 pair
    big = _column(0, {1: 1.0, 2: 1.0}, {1: 3000.0, 2: 3000.0}, 300.0)
    left = _column(1, {1: 1.0}, {1: 3000.0}, 350.0)
    right = _column(2, {2: 1.0}, {2: 3000.0}, 350.0)
    columns = [big, left, right]
    exact = solve_fleet_master(columns)
    assert exact.exhaustive and exact.objective == pytest.approx(700.0)
    assert exact.greedy_objective == pytest.approx(600.0)
    # node cap 0: the search stops at the root, only the starting fleets can be returned
    cold = solve_fleet_master(columns, node_cap=0)
    assert not cold.exhaustive and cold.objective == pytest.approx(600.0)
    warm = solve_fleet_master(columns, node_cap=0, incumbent=exact.selected)
    assert warm.objective == pytest.approx(700.0) and fleet_feasible(warm.selected) == ""
    # adding columns keeps the incumbent feasible, so the answer never regresses
    more = [*columns, _column(3, {3: 1.0}, {3: 3000.0}, 20.0)]
    later = solve_fleet_master(more, node_cap=0, incumbent=warm.selected)
    assert later.objective >= 700.0 - 1e-9
    # an incumbent that lost a column is ignored (never trusted blindly)
    stale = solve_fleet_master([big, left], node_cap=0, incumbent=exact.selected)
    assert stale.objective == pytest.approx(600.0) and fleet_feasible(stale.selected) == ""
