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
def test_earth_leg_prescreen_flies_low_authority_ratio_legs_first(catalogue, family) -> None:
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        certify_earth_legs,
        cluster_search_settings,
    )
    from spacepdhcg.gtoc12.screening import thrust_authority_km_s

    _label, members = family
    settings = cluster_search_settings(
        ClusterPricingSettings(launch_epochs=SMALL_LAUNCH_EPOCHS, earth_leg_tofs=SMALL_EARTH_TOFS),
        members.shape[0],
    )
    flown: list[tuple[int, float, float, float]] = []

    def refuse_all(_catalogue, target, launch, tof, lambert_dv, _scvx):
        flown.append((target, launch, tof, lambert_dv))
        return None

    for prescreen in (0.7, float("inf")):
        flown.clear()
        legs, rejected = certify_earth_legs(
            catalogue,
            members,
            settings,
            count=1,
            max_checks=8,
            cache={},
            certify=refuse_all,
            continuous=False,
            prescreen_ratio=prescreen,
        )
        assert not legs and len(rejected) == len(flown) == 8
        ratios = [
            dv / float(thrust_authority_km_s(settings.initial_mass, tof, 1.0))
            for _t, _l, tof, dv in flown
        ]
        if prescreen == 0.7:
            low = ratios
            assert all(r <= 0.7 + 1e-9 for r in ratios)
            assert rejected[0]["authority_ratio"] == pytest.approx(ratios[0])
        else:
            # the plain score order tries at least one leg the prescreen deferred
            assert any(r > 0.7 for r in ratios) or ratios == low


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
            prescreen_ratio=float("inf"),  # score order only: the fake certifier is parity-based
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
def test_surrogate_earth_leg_optimiser_obeys_launch_constraints_and_is_deterministic(
    catalogue, family
) -> None:
    """The continuous (launch, TOF) optimiser stays inside the official launch constraints."""

    from spacepdhcg.gtoc12.earthleg import EarthLegBounds, EarthLegModel, optimise_earth_leg
    from spacepdhcg.gtoc12.screening import thrust_authority_km_s

    _label, members = family
    target = int(members[0])
    bounds = EarthLegBounds(
        launch_min=C.MISSION_START_MJD,
        launch_max=C.MISSION_START_MJD + 3.0 * C.YEAR_DAYS,  # later launches, as the references
        tof_min=300.0,
        tof_max=900.0,
    )
    model = EarthLegModel()
    runs = [
        optimise_earth_leg(
            catalogue,
            target,
            model=model,
            bounds=bounds,
            launch_grid_days=180.0,
            tof_grid_days=150.0,
            starts=3,
            max_evaluations=120,
        )
        for _ in range(2)
    ]
    assert [leg.summary() for leg in runs[0]] == [leg.summary() for leg in runs[1]]
    legs = runs[0]
    assert legs, "the grid must admit at least one Earth leg to this family"
    assert legs == sorted(legs, key=lambda leg: (-leg.score, leg.launch_epoch, leg.tof_days))
    for leg in legs:
        assert bounds.launch_min <= leg.launch_epoch <= bounds.launch_max
        assert bounds.tof_min <= leg.tof_days <= bounds.tof_max
        assert leg.arrival_epoch <= bounds.latest_arrival + 1e-9
        assert leg.evaluations <= 120
        # the flown ΔV (above the free 6 km/s launch v∞) fits the thrust authority of the leg
        full = thrust_authority_km_s(model.initial_mass, leg.tof_days, 1.0)
        assert leg.lambert_dv_km_s <= model.authority_ratio * full + 1e-9
        evaluated = model.evaluate(
            catalogue, target, np.asarray([leg.launch_epoch]), np.asarray([leg.tof_days])
        )
        assert bool(evaluated["feasible"][0])
        assert evaluated["departure_excess"][0] >= 0.0  # nothing charged below the allowance
        assert evaluated["score"][0] == pytest.approx(leg.score)
        # snapped so the certified plan is reproducible
        assert leg.tof_days == round(leg.tof_days)
        assert round(leg.launch_epoch, 1) == leg.launch_epoch
    # starts with distinct launch epochs give distinct legs (deduplicated on the snapped point)
    assert len({(leg.launch_epoch, leg.tof_days) for leg in legs}) == len(legs)


@requires_data
def test_forward_collection_tour_collects_in_deploy_order_after_one_repositioning_hop(
    catalogue, family
) -> None:
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_search_settings,
        family_clusters,
    )
    from spacepdhcg.gtoc12.clusters import ClusterBands
    from spacepdhcg.gtoc12.retiming import build_visits, orders_of

    # the campaign's own pool and phasing-aware bands: family 0 (99 co-moving members)
    a_au = catalogue.semi_major_axis_km / C.AU_KM
    mask = (a_au >= 2.2) & (a_au <= 3.0) & (catalogue.eccentricity <= 0.15)
    mask &= np.rad2deg(catalogue.inclination_rad) <= 8.0
    bands = ClusterBands(radius=2.0, phase_deg=8.0, visit_epochs=ClusterBands().visit_epochs)
    pool = catalogue.ids[mask]
    _label, members = family_clusters(catalogue, pool, bands=bands, min_members=12)[0]
    settings = cluster_search_settings(
        ClusterPricingSettings(**{**PRICING_SETTINGS, "beam_width": 6, "max_deploys": 3}),
        members.shape[0],
    )
    # the substitution pass is covered by its own test; here every plan comes from the beam's
    # own chains (the Lambert-count comparison below assumes no re-flown chains)
    settings = replace(
        settings,
        earth_leg_tofs=tuple(float(t) for t in range(300, 901, 50)),
        harvest_substitution=False,
    )
    search = RouteSearch(catalogue, members, settings)
    result = search.run()
    assert result.candidates

    def classify(plan) -> str:
        deploys, collects = orders_of(plan)
        if len(deploys) == 1:
            return "reverse"  # a single camp: every mode coincides
        if collects == deploys:
            return "forward_revisit"
        if collects == [deploys[-1], *deploys[:-1]]:
            return "forward"
        if collects == list(reversed(deploys)):
            return "reverse"
        return "greedy"

    modes = {classify(plan) for plan in result.candidates if len(plan.asteroids) >= 3}
    assert modes & {"forward", "forward_revisit"}, modes
    for plan in result.candidates:
        deploys, collects = orders_of(plan)
        camp = deploys[-1]
        mode = classify(plan)
        assert plan.feasible
        assert plan.legs[-1].role == "earth_return"
        assert plan.legs[-1].from_id == collects[-1]
        assert plan.collect_epochs[collects[-1]] == plan.legs[-1].departure_epoch
        # every stay honours the minimum and the masses are the official accumulation
        for asteroid in deploys:
            stay = plan.collect_epochs[asteroid] - plan.deploy_epochs[asteroid]
            assert stay >= C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS
            assert plan.collected_mass[asteroid] == pytest.approx(C.maximum_collected_mass(stay))
        # hops that leave an asteroid at an epoch other than its collection collect nothing:
        # only the forward_revisit tour has one (camp -> first deployed), all others none
        silent = [
            leg
            for leg in plan.legs
            if leg.role == "collect_hop"
            and abs(leg.departure_epoch - plan.collect_epochs[leg.from_id]) > 1e-6
        ]
        visits = build_visits(deploys, collects)
        if mode == "forward_revisit":
            assert len(silent) == 1
            assert silent[0].from_id == camp and silent[0].to_id == deploys[0]
            assert silent[0].departure_epoch >= plan.deploy_epochs[camp]
            assert collects[-1] == camp
            # the re-timer revisits the camp: deploy-only first, collect on the way back
            assert len(visits) == 2 + 2 * len(deploys)
            assert visits[len(deploys)].body == camp
            assert visits[len(deploys)].deploy and not visits[len(deploys)].collect
            assert visits[-2].body == camp and visits[-2].collect and not visits[-2].deploy
        elif silent:
            # the exact collect DP may also leave the camp uncollected and pick it up at any
            # later position of a free order (a repositioning hop, then a plain revisit)
            assert len(silent) == 1
            assert silent[0].from_id == camp and collects[0] != camp and camp in collects
            assert len(visits) == 2 + 2 * len(deploys)
            assert visits[len(deploys)].body == camp
            assert visits[len(deploys)].deploy and not visits[len(deploys)].collect
        else:
            assert collects[0] == camp
            assert len(visits) == 1 + 2 * len(deploys)  # camp visit deploys and collects
            if mode == "forward":
                assert collects[-1] == deploys[-2]
    # the candidate list is deterministic
    again = RouteSearch(catalogue, members, settings).run()
    assert [p.summary() for p in again.candidates] == [p.summary() for p in result.candidates]

    # collect look-ahead with the DP's pair table (the default): every deploy pair is priced at
    # its calibrated harvest-window cost at the collector's reference mass - independent of the
    # deploy-time mass, cached per unordered pair, an unreachable pair charged the fixed penalty
    hops = search.hops_from(int(members[0]), C.MISSION_START_MJD + 800.0)
    targets = np.asarray(hops["target_ids"], dtype=np.int64)
    cost = search.collect_lookahead(int(members[0]), targets, C.MISSION_START_MJD + 800.0, 2500.0)
    assert cost.shape == targets.shape and np.all(cost > 0.0) and np.all(np.isfinite(cost))
    assert np.all(cost <= settings.harvest_unreachable_kg)
    heavier = search.collect_lookahead(
        int(members[0]), targets, C.MISSION_START_MJD + 800.0, 2900.0
    )
    assert np.array_equal(heavier, cost)
    assert all(
        (min(int(members[0]), int(t)), max(int(members[0]), int(t))) in search._harvest_cache
        for t in targets
    )
    # the Lambert look-ahead (no DP table) prices the deploy-time mass: heavier costs more
    lambert = RouteSearch(catalogue, members, replace(settings, harvest_window_ranking=False))
    cost = lambert.collect_lookahead(int(members[0]), targets, C.MISSION_START_MJD + 800.0, 2500.0)
    assert cost.shape == targets.shape and np.all(cost > 0.0)
    assert np.isfinite(cost).any()
    heavier = lambert.collect_lookahead(
        int(members[0]), targets, C.MISSION_START_MJD + 800.0, 2900.0
    )
    finite = np.isfinite(cost)
    assert np.all(heavier[finite] > cost[finite])  # same ΔV costs more propellant at higher mass
    assert (int(members[0]), round(C.MISSION_START_MJD + 800.0, 6)) in lambert._lookahead_cache
    # a beam with the weight on is deterministic, for either look-ahead
    for ranking in (True, False):
        weighted = replace(settings, collect_lookahead_weight=0.5, harvest_window_ranking=ranking)
        first = RouteSearch(catalogue, members, weighted).run()
        second = RouteSearch(catalogue, members, weighted).run()
        assert [p.summary() for p in first.candidates] == [p.summary() for p in second.candidates]
    assert first.lambert_evaluations > result.lambert_evaluations  # the Lambert look-ahead screens


def _replay_plan(search: RouteSearch, plan: RoutePlan) -> tuple[float, float, dict[int, float]]:
    """Independent forward mass pass over a plan's legs with the beam's leg model: (final mass,
    propellant, collected kg per asteroid).  Collection happens at the departure of the leg
    leaving the collect epoch; one miner is dropped at every deploy arrival."""

    mass = search.settings.initial_mass
    propellant_total = 0.0
    collected: dict[int, float] = {}
    for leg in plan.legs:
        if leg.role == "camp":
            continue
        if leg.role in ("collect_hop", "earth_return"):
            asteroid = leg.from_id
            if abs(plan.collect_epochs[asteroid] - leg.departure_epoch) < 1e-6:
                gained = C.maximum_collected_mass(
                    plan.collect_epochs[asteroid] - plan.deploy_epoch_of(asteroid)
                )
                collected[asteroid] = gained
                mass += gained
        inflation = leg.inflation
        if leg.role == "collect_hop":
            inflation = search.hop_inflation_for(leg.delta_v_proxy_km_s, mass, leg.tof_days)
        propellant = search._propellant(mass, leg.delta_v_proxy_km_s, inflation)
        propellant_total += propellant
        mass -= propellant
        if leg.role in ("earth_out", "deploy_hop"):
            mass -= C.MINER_MASS_KG
    return mass, propellant_total, collected


def _assert_exact_bookkeeping(search: RouteSearch, plan: RoutePlan) -> None:
    mass, propellant, collected = _replay_plan(search, plan)
    assert mass == pytest.approx(plan.final_mass_proxy_kg, abs=1e-6)
    assert propellant == pytest.approx(plan.propellant_proxy_kg, abs=1e-6)
    assert collected == pytest.approx(plan.collected_mass)
    assert set(plan.deploy_epochs) == set(plan.collect_epochs) == set(collected)
    for asteroid, epoch in plan.deploy_epochs.items():
        assert plan.collect_epochs[asteroid] - epoch >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
    arrivals = {}
    for leg in plan.legs:
        if leg.role == "deploy_hop" or leg.role == "earth_out":
            arrivals[leg.to_id] = leg.arrival_epoch
    assert arrivals == pytest.approx(plan.deploy_epochs)  # every deploy is one arrival
    assert plan.feasible


@requires_data
def test_harvest_substitution_reflies_the_chain_exactly_and_only_adds_better_plans(
    catalogue,
) -> None:
    """The substitution pass (a) re-flies the deploy chain through the substitute with the
    beam's own leg model and an exact mass chain, (b) never drops a beam plan, (c) adds only
    feasible self-cleaning plans that swap at most one deploy per accepted round against a beam
    chain with the same Earth leg and score above the seed they came from, (d) keeps the exact
    forward-pass bookkeeping on every emitted plan, and (e) is deterministic."""

    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_search_settings,
        family_clusters,
    )
    from spacepdhcg.gtoc12.clusters import ClusterBands

    a_au = catalogue.semi_major_axis_km / C.AU_KM
    mask = (a_au >= 2.2) & (a_au <= 3.0) & (catalogue.eccentricity <= 0.15)
    mask &= np.rad2deg(catalogue.inclination_rad) <= 8.0
    bands = ClusterBands(radius=2.0, phase_deg=8.0, visit_epochs=ClusterBands().visit_epochs)
    _label, members = family_clusters(catalogue, catalogue.ids[mask], bands=bands, min_members=12)[
        0
    ]
    settings = cluster_search_settings(
        ClusterPricingSettings(
            **{**PRICING_SETTINGS, "beam_width": 4, "max_deploys": 4, "neighbours": 16}
        ),
        members.shape[0],
    )
    settings = replace(
        settings,
        earth_leg_tofs=tuple(float(t) for t in range(300, 901, 50)),
        harvest_substitution=False,
    )
    base = RouteSearch(catalogue, members, settings)
    base_result = base.run()
    assert base_result.best is not None

    # -- (a) the chain re-flight, on a beam chain of three deploys
    level = base._select(base._first_level())
    partial = base._select(base._expand(level[0]))[0]
    partial = base._select(base._expand(partial))[0]
    assert len(partial.deployed) == 3
    chain = [a for a, _ in partial.deployed]
    mass_after_earth = partial.mass + partial.hop_propellant + C.MINER_MASS_KG * 2
    same = base._rebuild_chain(partial, 1, chain[1])
    assert same is not None and [a for a, _ in same.deployed] == chain
    assert same.mass + same.hop_propellant + C.MINER_MASS_KG * 2 == pytest.approx(mass_after_earth)
    departure = base._deploy_departure(partial, chain[1])
    pool = [
        int(b) for b in base.hops_from(chain[0], departure)["target_ids"] if int(b) not in chain
    ]
    rebuilt = next(
        (r for r in (base._rebuild_chain(partial, 1, b) for b in pool) if r is not None), None
    )
    assert rebuilt is not None
    ids = [a for a, _ in rebuilt.deployed]
    assert ids[0] == chain[0] and ids[2] == chain[2] and ids[1] not in chain
    assert len(set(ids)) == 3 and rebuilt.location == ids[2]
    assert rebuilt.legs[0] == partial.legs[0]  # the Earth leg is kept verbatim
    assert rebuilt.mass + rebuilt.hop_propellant + C.MINER_MASS_KG * 2 == pytest.approx(
        mass_after_earth
    )
    # the chain's own legs reproduce its mass: exact leg-by-leg replay
    mass, burnt = mass_after_earth, 0.0
    for leg in rebuilt.legs[1:]:
        if leg.role != "deploy_hop":
            continue
        propellant = base._propellant(mass, leg.delta_v_proxy_km_s, leg.inflation)
        assert leg.inflation == pytest.approx(
            base.hop_inflation_for(leg.delta_v_proxy_km_s, mass, leg.tof_days)
        )
        burnt += propellant
        mass -= propellant + C.MINER_MASS_KG
    assert mass == pytest.approx(rebuilt.mass) and burnt == pytest.approx(rebuilt.hop_propellant)
    assert [e for _a, e in rebuilt.deployed] == [
        leg.arrival_epoch for leg in rebuilt.legs if leg.role != "camp"
    ]
    # a substitute outside the screened neighbours, a banned pair or a repeat is refused
    assert base._rebuild_chain(partial, 1, chain[2]) is None
    assert base._rebuild_chain(partial, 0, ids[1]) is None
    base.banned_pairs.add((chain[0], ids[1]))
    assert base._rebuild_chain(partial, 1, ids[1]) is None
    base.banned_pairs.clear()

    # -- (b)-(e) the pass itself
    on = RouteSearch(
        catalogue,
        members,
        replace(settings, harvest_substitution=True, substitution_budget_seconds=150.0),
    )
    on_result = on.run()
    base_summaries = [p.summary() for p in base_result.candidates]
    on_summaries = [p.summary() for p in on_result.candidates]
    for summary in base_summaries:
        assert summary in on_summaries
    extra = [p for p in on_result.candidates if p.summary() not in base_summaries]
    assert on.substitution_stats["tried"] > 0
    assert len(extra) == on.substitution_stats["improved"]
    assert on.plan_score(on_result.best) >= base.plan_score(base_result.best) - 1e-9
    seeds = [p for p in base_result.candidates if p.feasible][: settings.substitution_top]
    floor = min(base.plan_score(p) for p in seeds)
    allowed = set(int(a) for a in members)
    for plan in extra:
        assert plan.feasible and plan.self_cleaning
        assert len(set(plan.asteroids)) == len(plan.asteroids)
        assert set(plan.asteroids) <= allowed
        kin = [b for b in base_result.candidates if b.legs[0] == plan.legs[0]]
        assert kin
        swapped = min(len(set(b.deploy_epochs) ^ set(plan.deploy_epochs)) for b in kin)
        assert 2 <= swapped <= 2 * settings.substitution_rounds
        assert on.plan_score(plan) > floor
        _assert_exact_bookkeeping(on, plan)
    for plan in on_result.candidates:
        _assert_exact_bookkeeping(on, plan)
    again = RouteSearch(catalogue, members, on.settings).run()
    assert [p.summary() for p in again.candidates] == on_summaries
    if extra:
        assert on.substitution_stats["gain_kg"] > 0.0 or on.plan_score(extra[0]) > floor


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
    return_sweep=False,  # the SCvx return sweep is exercised with a stub in its own test
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
def test_price_cluster_hands_the_swept_return_cells_to_the_retimer_and_the_dp(
    catalogue, family, monkeypatch
) -> None:
    """With ``return_sweep`` on, every certified ship's return is swept (stubbed SCvx here) at
    its return-departure mass around its flown cell, nearest cells first, and the certified
    cells are set on the re-timer and the slot's DP pair table before the joint re-timing."""

    from spacepdhcg.gtoc12 import bundles
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, price_cluster
    from spacepdhcg.gtoc12.retiming import Retimer
    from spacepdhcg.gtoc12.returnsweep import ReturnSweep

    calls: list[dict] = []
    handed: list[tuple[str, int]] = []

    def fake_sweep_return(_catalogue, asteroid, mass, departures, tofs, **kwargs):
        calls.append(
            {
                "asteroid": int(asteroid),
                "mass": float(mass),
                "departures": np.asarray(departures),
                "tofs": np.asarray(tofs),
                **kwargs,
            }
        )
        shape = (departures.shape[0], tofs.shape[0])
        certified = np.zeros(shape, dtype=bool)
        certified[shape[0] // 2, 0] = True  # one certified cell near the flown return
        return ReturnSweep(
            int(asteroid),
            float(mass),
            np.asarray(departures),
            np.asarray(tofs),
            np.ones(shape, dtype=bool),
            certified,
            np.where(certified, 6.0, np.inf),
            np.where(certified, 250.0, np.inf),
            solves=int(np.prod(shape)),
            wall_seconds=0.5,
        )

    real_set = Retimer.set_return_sweep

    def spy_set(self, sweep):
        handed.append(("retimer", int(sweep.asteroid)))
        real_set(self, sweep)

    monkeypatch.setattr(bundles, "sweep_return", fake_sweep_return)
    monkeypatch.setattr(Retimer, "set_return_sweep", spy_set)
    label, members = family
    settings = ClusterPricingSettings(
        **{**PRICING_SETTINGS, "return_sweep": True, "return_sweep_budget_seconds": 30.0}
    )
    bundle = price_cluster(
        catalogue,
        members,
        label=label,
        settings=settings,
        certify_earth=_proxy_certify,
        refine=_proxy_refine,
    )
    if not bundle.ships:
        pytest.skip("no chain closes in this family on the coarse grids")
    summary = bundle.summary()
    reports = [s.get("return_sweep") for s in summary["ships"]]
    assert calls and all(r is not None for r in reports)
    assert [r["asteroid"] for r in reports] == [c["asteroid"] for c in calls]
    assert [("retimer", c["asteroid"]) for c in calls] == handed
    for call, report in zip(calls, reports, strict=True):
        # the grid is the compact lattice-aligned box around the flown return; the TOFs are
        # snapped onto the re-timer's (here 30-day) lattice
        step = settings.retime_step_days
        assert tuple(call["tofs"]) == tuple(
            sorted({step * round(t / step) for t in settings.return_sweep_tofs})
        )
        assert call["departures"].shape[0] <= (
            settings.return_sweep_back_steps + settings.return_sweep_forward_steps + 1
        )
        assert call["time_budget_seconds"] <= settings.return_sweep_budget_seconds
        assert call["nearest_to"] is not None
        assert np.all(call["departures"] + call["tofs"].min() <= C.MISSION_END_MJD)
        # the proxy route has no SCvx legs, so the mass is the plan's return-departure proxy
        assert call["mass"] > C.DRY_MASS_KG
        assert report["certified"] == 1
        assert report["cells"] == call["departures"].shape[0] * call["tofs"].shape[0]
        assert report["cheapest_certified_kg"] == pytest.approx(250.0)
        assert report["cheapest_cell"][1] == pytest.approx(call["tofs"][0])


@requires_data
def test_single_slot_pricing_stays_inside_the_declared_memory_budget(bundle) -> None:
    """The regression guard for the v6 memory transient (3.04 GB PSS over 4 workers).

    ``price_cluster`` marks every phase; the marks must show (a) the high-water mark the
    pricing added stays under the declared per-slot budget and (b) freed heap is handed back
    at each mark (the resident size after a phase is bounded, not ratcheting).  The declared
    budget itself must fit three workers under the operator's 2 GB process-tree limit.
    """

    from spacepdhcg.gtoc12.memory import MemoryBudget

    budget = MemoryBudget()
    assert budget.workers == 3 and budget.fits(), budget.projected_tree_mb
    records = bundle.memory_phases
    assert records and records[0]["phase"] == "start"
    phases = [r["phase"] for r in records]
    assert any(p.endswith("earth legs") for p in phases) and "orphan repair" in phases
    if any(np.isnan(r["peak_mb"]) for r in records):
        pytest.skip("no resource usage on this platform")
    baseline = records[0]["peak_mb"]  # whatever the test process held before the pricing
    grown = sum(r["peak_growth_mb"] for r in records[1:])
    assert budget.check_slot(baseline + grown, baseline_mb=baseline), (grown, records)
    assert bundle.peak_rss_mb - baseline <= budget.slot_peak_mb
    # the heap is trimmed at every mark: what stays resident is live data, not freed pages
    trimmed = [r["rss_after_trim_mb"] for r in records if r["rss_after_trim_mb"] is not None]
    if trimmed:
        assert max(trimmed) <= records[0]["rss_mb"] + budget.slot_peak_mb / 2.0
    assert all(r["elapsed_seconds"] >= 0.0 for r in records)
    # a priced slot parks its search for the orphan repair with its memo tables released (the
    # per-slot live growth v7 measured): the report says what was dropped, and it was something
    for ship in bundle.ships:
        released = ship.report["released_caches"]
        assert set(released) == {
            "hops",
            "returns",
            "collects",
            "lookahead",
            "harvest",
            "chain_tours",
            "pairs",
        }
        assert released["hops"] > 0 and all(v >= 0 for v in released.values())


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


@requires_data
def test_pinned_deploys_keep_their_epoch_through_drop_and_retiming(
    catalogue, family, bundle
) -> None:
    """A deploy another ship collects must survive the deployer's re-timing at the same epoch.

    Without the pin, dropping an orphan re-times the whole chain, shifts the remaining deploy
    epochs and strands every collector of those miners (the probe_v4 bundle lost 240 kg that way).
    """

    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
        drop_asteroid,
    )
    from spacepdhcg.gtoc12.retiming import Retimer, Visit, build_visits, orders_of

    visits = build_visits([1, 2, 3], [3, 2, 1], pinned={2: 64600.0, 3: 64700.0})
    assert [v.pinned_arrival for v in visits] == [None, None, 64600.0, 64700.0, None, None, None]
    assert Visit(5, True, False, "deploy_hop").pinned_arrival is None

    pricing = ClusterPricingSettings(**PRICING_SETTINGS)
    retimer = Retimer(
        catalogue,
        cluster_search_settings(pricing, family[1].shape[0]),
        cluster_retime_settings(pricing, last=True),
    )
    # the DP realises a TOF as whole lattice steps: every grid TOF must be one (400 d bounds on a
    # 30-day lattice used to make the DP accept legs the forward pass then refused)
    step = retimer.settings.step_days
    for role in ("earth_out", "earth_return", "deploy_hop", "collect_hop"):
        tofs = retimer._tofs(role)
        assert np.allclose(tofs / step, np.round(tofs / step))
    # a plan this re-timer produced, re-timed again with every deploy pinned, must reproduce the
    # deploy epochs exactly (the unpinned optimum is one admissible pinned solution)
    plan = None
    for ship in bundle.ships:
        for variant in [ship.route, *ship.variants]:
            if len(variant.plan.deploy_epochs) < 2:
                continue
            retimed = retimer.retime(variant.plan)
            if retimed.plan is not None:
                plan = retimed.plan
                break
        if plan is not None:
            break
    if plan is None:
        pytest.skip("no fixture plan closes on the coarse re-timing lattice")
    deploy_order, collect_order = orders_of(plan)
    result = retimer.retime_order(
        deploy_order,
        collect_order,
        retimer._plan_masses(plan),
        original=plan,
        pinned=dict(plan.deploy_epochs),
    )
    assert result.plan is not None
    for asteroid, epoch in plan.deploy_epochs.items():
        assert result.plan.deploy_epochs[asteroid] == pytest.approx(epoch, abs=1e-9)
    # dropping the first deploy while pinning the second keeps the second where it was
    victim, kept = deploy_order[0], deploy_order[1]
    dropped = drop_asteroid(plan, victim, retimer, pinned={kept: plan.deploy_epochs[kept]})
    if dropped is not None:
        assert dropped.deploy_epochs[kept] == pytest.approx(plan.deploy_epochs[kept], abs=1e-9)
    # an off-lattice pin is infeasible rather than silently rounded
    off = retimer.retime_order(
        deploy_order,
        collect_order,
        retimer._plan_masses(plan),
        original=plan,
        pinned={kept: plan.deploy_epochs[kept] + 0.37},
    )
    assert off.plan is None


@requires_data
def test_joint_harvest_orders_share_the_pool_once_and_retime_with_foreign_epochs(
    catalogue, family, bundle
) -> None:
    """Collect tours over the pooled miners: each miner to one ship, camps first, deploys kept."""

    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
    )
    from spacepdhcg.gtoc12.cooperative import MinerPool
    from spacepdhcg.gtoc12.harvest import (
        HarvestSettings,
        harvest_report,
        joint_collect_orders,
        retime_harvest,
    )
    from spacepdhcg.gtoc12.retiming import Retimer, orders_of

    if len(bundle.ships) < 2:
        pytest.skip("the joint harvest needs two ships in the priced bundle")
    pricing = ClusterPricingSettings(**PRICING_SETTINGS)
    search_settings = cluster_search_settings(pricing, family[1].shape[0])
    plans = {ship.slot: ship.route.plan for ship in bundle.ships}
    pool_miners = {a for plan in plans.values() for a in plan.deploy_epochs}
    states, uncollected = joint_collect_orders(catalogue, plans, search_settings)
    again, again_uncollected = joint_collect_orders(catalogue, plans, search_settings)
    assert [s.collect_order for s in states.values()] == [s.collect_order for s in again.values()]
    assert uncollected == again_uncollected  # deterministic
    collected = [a for state in states.values() for a in state.collect_order]
    assert len(collected) == len(set(collected))  # each miner to exactly one ship
    assert set(collected) | set(uncollected) == pool_miners
    for slot, state in states.items():
        deploy_order, _ = orders_of(plans[slot])
        assert state.deploy_order == deploy_order  # the deploy chain is kept
        assert state.collect_order[0] == deploy_order[-1]  # the camp is collected first
        own = set(deploy_order)
        assert set(state.foreign) == set(state.collect_order) - own
        for asteroid, epoch in state.foreign.items():
            deployer = next(s for s, p in plans.items() if asteroid in p.deploy_epochs)
            assert deployer != slot and plans[deployer].deploy_epochs[asteroid] == epoch
        assert state.stop_reason
    report = harvest_report(states, uncollected)
    assert report["foreign_collects"] == sum(len(s.foreign) for s in states.values())
    # a max_collects cap is honoured and is the stop reason when it binds
    capped, _ = joint_collect_orders(
        catalogue, plans, search_settings, HarvestSettings(max_collects=1)
    )
    assert all(len(s.collect_order) == 1 for s in capped.values())
    assert all(s.stop_reason == "max_collects" for s in capped.values())
    # the DP re-times a new order with the foreign deploy epochs, or drops the tail to close
    retimer = Retimer(catalogue, search_settings, cluster_retime_settings(pricing, last=False))
    plan = None
    failures = []
    for slot in sorted(states):
        plan, dropped, failure = retime_harvest(plans[slot], states[slot], retimer, drop_tail=3)
        if plan is not None:
            break
        failures.append(failure)
    if plan is None:
        pytest.skip(f"no harvest order closes on the coarse lattice: {failures}")
    new_deploy, new_collect = orders_of(plan)
    assert new_deploy == states[slot].deploy_order
    assert new_collect == [a for a in states[slot].collect_order if a not in dropped]
    assert set(plan.foreign_deploy_epochs) == set(new_collect) - set(new_deploy)
    for asteroid in new_collect:
        stay = plan.collect_epochs[asteroid] - plan.deploy_epoch_of(asteroid)
        assert stay >= C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS - 1e-6
        assert plan.collected_mass[asteroid] == pytest.approx(C.maximum_collected_mass(stay))
    assert plan.feasible
    # a pool of the deployers alone accepts the new plan (every foreign collect has a deployer)
    deployers = MinerPool()
    for other, other_plan in plans.items():
        if other != slot:
            deployers.deployed.update({a: (e, other) for a, e in other_plan.deploy_epochs.items()})
    deployers.register(plan, slot)
    assert set(deployers.collected) == set(new_collect)


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
    monkeypatch.setattr(
        bundles, "drop_asteroid", lambda plan, asteroid, retimer, pinned=None: dropped_plan
    )
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
    # mutual collection (ship 2 also collects ship 1's miner 5): no slot order registers this
    # one ship at a time, the two-phase registration does (the joint harvest produces such pairs)
    mutual_deployer = replace(
        deployer,
        collect_epochs={8: arrive + 3300.0, 5: arrive + 3100.0},
        collected_mass={8: 300.0, 5: 250.0},
        foreign_deploy_epochs={5: arrive},
    )
    mutual_collector = replace(
        collector, collect_epochs={9: arrive + 3300.0}, collected_mass={9: 200.0}
    )
    pool = MinerPool()
    pool.register_all([(mutual_collector, 1), (mutual_deployer, 2)])
    assert pool.collected == {9: 1, 8: 2, 5: 2} and pool.orphans() == {}
    # a stale epoch inside the cycle still fails, and leaves the pool untouched
    stale = replace(mutual_deployer, foreign_deploy_epochs={5: arrive + 15.0})
    pool = MinerPool()
    with pytest.raises(ValueError, match="stale deploy epoch"):
        pool.register_all([(mutual_collector, 1), (stale, 2)])
    assert pool.deployed == {} and pool.collected == {}
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


@requires_data
def test_leg_stats_decode_references_with_shared_roles(catalogue) -> None:
    from spacepdhcg.gtoc12.data import data_directory
    from spacepdhcg.gtoc12.legstats import ROLES, compare, format_table, leg_costs

    path = data_directory() / "37_mass_optimal_self_cleaning.txt"
    if not path.exists():
        pytest.skip("reference solution not fetched")
    report = leg_costs("antipodes37", path, catalogue)
    assert report.ships == 37
    summary = report.summary(cheap_hop_kg=75.0)
    for role in ROLES:
        assert summary["roles"][role]["propellant_kg"]["n"] > 0
    hist = summary["roles"]["collect_hop"]["histogram_kg"]
    assert sum(b["count"] for b in hist) == summary["roles"]["collect_hop"]["propellant_kg"]["n"]
    assert 0.3 < summary["hops_at_or_under_cheap_kg"] < 0.7  # the references: ~46%
    assert 400.0 < summary["roles"]["earth_out"]["propellant_kg"]["median"] < 520.0
    comparison = compare({"a": path, "b": path}, catalogue)
    table = format_table(comparison)
    assert "collect_hop" in table and "<= cheap fraction" in table
    a, b = comparison["solutions"]["a"], comparison["solutions"]["b"]
    assert {k: v for k, v in a.items() if k != "name"} == {
        k: v for k, v in b.items() if k != "name"
    }


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
    # node cap 0 without the LP: the search stops at the root, only the starting fleets can be
    # returned; with the LP branch and bound the same cap is rescued
    cold = solve_fleet_master(columns, node_cap=0, lp_bound=False)
    assert not cold.exhaustive and cold.objective == pytest.approx(600.0)
    rescued = solve_fleet_master(columns, node_cap=0)
    assert rescued.lp_proven and rescued.objective == pytest.approx(700.0)
    warm = solve_fleet_master(columns, node_cap=0, incumbent=exact.selected, lp_bound=False)
    assert warm.objective == pytest.approx(700.0) and fleet_feasible(warm.selected) == ""
    # adding columns keeps the incumbent feasible, so the answer never regresses
    more = [*columns, _column(3, {3: 1.0}, {3: 3000.0}, 20.0)]
    later = solve_fleet_master(more, node_cap=0, incumbent=warm.selected, lp_bound=False)
    assert later.objective >= 700.0 - 1e-9
    # an incumbent that lost a column is ignored (never trusted blindly)
    stale = solve_fleet_master([big, left], node_cap=0, incumbent=exact.selected)
    assert stale.objective == pytest.approx(600.0) and fleet_feasible(stale.selected) == ""


@requires_data
def test_family_partitions_unions_radii_and_bands_without_duplicates(catalogue, monkeypatch):
    """Several partitions are priced as one cheapest-first list: labels offset per partition,
    duplicate member sets dropped, every partition ranked at its own visit epochs."""

    from spacepdhcg.gtoc12 import bundles
    from spacepdhcg.gtoc12.bundles import (
        FAMILY_LABEL_STRIDE,
        family_clusters,
        family_partitions,
    )
    from spacepdhcg.gtoc12.clusters import ClusterBands
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    seen_epochs: list[tuple[float, ...]] = []

    def cheap_rank(_catalogue, families, _settings=None, *, visit_epochs=None, **_kw):
        seen_epochs.append(tuple(visit_epochs))
        # cheaper the larger the family; ties on the label like the real ranker
        ranked = [
            (
                int(label),
                members,
                {"members": float(members.shape[0]), "score": 1000.0 - members.shape[0]},
            )
            for label, members in families
        ]
        ranked.sort(key=lambda item: (item[2]["score"], item[0]))
        return ranked

    monkeypatch.setattr(bundles, "rank_families", cheap_rank)
    collect = ClusterBands.collect_window(radius=2.0, phase_deg=12.0)
    phasing = ClusterBands(radius=2.0, phase_deg=12.0)
    partitions = [("collect_r2", collect), ("phasing_r2", phasing), ("collect_r2_again", collect)]
    ranked = family_partitions(catalogue, ids, bands=partitions, min_members=8)
    assert ranked, "the reduced instance has co-moving families at radius 2.0"
    # every partition was ranked at its own visit epochs, in order
    assert seen_epochs == [collect.phase_epochs, phasing.phase_epochs, collect.phase_epochs]
    # the first partition is present verbatim (labels unchanged) and the repeat adds nothing
    first = sorted(
        (label, tuple(int(a) for a in m))
        for label, m, s in ranked
        if s["partition"] == "collect_r2"
    )
    single = sorted(
        (int(label), tuple(int(a) for a in m))
        for label, m in family_clusters(catalogue, ids, bands=collect, min_members=8)
    )
    assert first == single
    assert not any(s["partition"] == "collect_r2_again" for _l, _m, s in ranked)
    # member sets and labels are unique; later partitions carry the stride offset
    keys = [frozenset(int(a) for a in m) for _l, m, _s in ranked]
    assert len(keys) == len(set(keys))
    labels = [label for label, _m, _s in ranked]
    assert len(labels) == len(set(labels))
    for label, _m, stats in ranked:
        assert label == stats["label_in_partition"] + stats["partition_index"] * FAMILY_LABEL_STRIDE
        assert stats["radius"] == 2.0
    assert any(s["partition_index"] == 1 for _l, _m, s in ranked)
    # cheapest first across partitions
    scores = [s["score"] for _l, _m, s in ranked]
    assert scores == sorted(scores)


def test_cluster_band_partitions_parses_radius_lists_and_band_sets():
    import argparse

    from spacepdhcg.gtoc12.cli import cluster_band_partitions

    args = argparse.Namespace(
        cluster_radius="1.75, 1.6",
        cluster_phase_deg=8.0,
        collect_epoch_families=True,
        static_families=False,
        all_family_bands=False,
    )
    only_collect = cluster_band_partitions(args)
    assert [name for name, _b in only_collect] == ["collect_r1.75", "collect_r1.6"]
    assert [b.radius for _n, b in only_collect] == [1.75, 1.6]
    assert all(len(b.phase_epochs) == 4 for _n, b in only_collect)
    args.all_family_bands = True
    both = cluster_band_partitions(args)
    assert [name for name, _b in both] == [
        "collect_r1.75",
        "phasing_r1.75",
        "collect_r1.6",
        "phasing_r1.6",
    ]
    assert [len(b.phase_epochs) for _n, b in both] == [4, 2, 4, 2]
    # the legacy single float still works
    args = argparse.Namespace(
        cluster_radius=1.5,
        cluster_phase_deg=8.0,
        collect_epoch_families=False,
        static_families=True,
        all_family_bands=False,
    )
    assert [name for name, _b in cluster_band_partitions(args)] == ["static_r1.5"]
