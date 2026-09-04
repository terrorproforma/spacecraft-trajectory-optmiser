"""Itinerary decoding, proxies, clustering bands, fleet rules, determinism and bounded memory."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.fleet import FleetPlan, assemble_fleet
from spacepdhcg.gtoc12.proxies import low_thrust_feasible, phasing_edelbaum_proxy
from spacepdhcg.gtoc12.solution import Event, ShipTrajectory, Solution, StateLine

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
# Earth legs cost >= ~4.8 km/s after the launch credit, so only long flights at good epochs pass
# the inflated thrust-authority check; small tests need a real (if coarse) launch grid.
SMALL_LAUNCH_EPOCHS = tuple(float(x) for x in C.MISSION_START_MJD + np.arange(0.0, 731.0, 90.0))
SMALL_EARTH_TOFS = (600.0, 750.0, 900.0)


# -- fleet rules (pure) --


def _fake_route(asteroids: tuple[int, ...], collected: float):
    return SimpleNamespace(
        plan=SimpleNamespace(asteroids=asteroids), total_collected_kg=collected, certified=True
    )


def test_fleet_rule_follows_bonus_formula() -> None:
    plan = FleetPlan([_fake_route((1, 2), 250.0), _fake_route((3,), 150.0)])
    assert plan.average_collected_kg == 200.0
    assert plan.ship_limit == pytest.approx(2.0 * np.exp(C.SHIP_COUNT_RHO_PER_KG * 200.0))
    assert plan.rule_satisfied
    assert plan.used_asteroids() == {1, 2, 3}
    crowded = FleetPlan([_fake_route((k,), 10.0) for k in range(5)])
    assert not crowded.rule_satisfied
    with pytest.raises(ValueError, match="exceeds the limit"):
        assemble_fleet(crowded, catalogue=None)


def test_low_thrust_feasibility_is_mass_consistent() -> None:
    # 2 km/s proxy in 120 days: fine for a light ship, impossible for a heavy one
    assert low_thrust_feasible(1200.0, 2.0, 120.0)
    assert not low_thrust_feasible(3000.0, 2.0, 60.0)
    light = low_thrust_feasible(np.array([1000.0, 3000.0]), np.array([1.5, 1.5]), 90.0)
    assert light.tolist() == [True, False]


# -- data-backed checks --


@pytest.fixture(scope="module")
def catalogue():
    if not data_available():
        pytest.skip("pinned GTOC12 data not fetched")
    return load_catalogue()


@requires_data
def test_phasing_proxy_is_zero_for_self_and_grows_with_separation(catalogue) -> None:
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    source = int(ids[0])
    proxy = phasing_edelbaum_proxy(
        catalogue, source, np.array([source]), C.MISSION_START_MJD, np.array([120.0, 240.0])
    )
    assert proxy["delta_v"].shape == (1, 2)
    assert np.allclose(proxy["delta_v"], 0.0, atol=1e-9)
    pool = ids[1:200]
    ranked = phasing_edelbaum_proxy(catalogue, source, pool, C.MISSION_START_MJD, np.array([180.0]))
    a_au = catalogue.semi_major_axis_km[catalogue.index_of(pool)] / C.AU_KM
    a_src = catalogue.semi_major_axis_km[catalogue.index_of(source)] / C.AU_KM
    far = np.abs(a_au - a_src) > 0.3
    near = np.abs(a_au - a_src) < 0.03
    assert far.any() and near.any()
    assert np.median(ranked["best_delta_v"][far]) > np.median(ranked["best_delta_v"][near])


@requires_data
def test_element_bands_and_band_pool_fallback(catalogue) -> None:
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
    from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings, element_deviations

    ids = build_reduced_instance(catalogue).asteroid_ids
    source = int(ids[3])
    da, de, di = element_deviations(catalogue, source, ids[:50])
    assert da.shape == de.shape == di.shape == (50,)
    self_index = int(np.where(ids[:50] == source)[0][0])
    assert da[self_index] == 0.0 and de[self_index] == 0.0 and di[self_index] == 0.0
    settings = SearchSettings(neighbours=12)
    search = RouteSearch(catalogue, ids[:200], settings)
    pool = search.band_pool(source)
    assert source not in set(pool.tolist())
    da, de, di = element_deviations(catalogue, source, pool)
    inside = (
        (da <= settings.filter_scale * settings.band_a_au)
        & (de <= settings.filter_scale * settings.band_e)
        & (di <= settings.filter_scale * settings.band_i_deg)
    )
    # either every member is inside the bands, or the sparse fallback returned exactly
    # ``neighbours`` nearest asteroids
    assert inside.all() or pool.shape[0] == settings.neighbours
    excluded = RouteSearch(catalogue, ids[:200], settings, excluded={source})
    assert source not in set(excluded.ids.tolist())


@requires_data
def test_first_level_is_block_invariant_and_search_deterministic(catalogue) -> None:
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
    from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

    ids = build_reduced_instance(catalogue).asteroid_ids[:90]
    common = dict(
        beam_width=4,
        max_deploys=2,
        neighbours=8,
        launch_epochs=SMALL_LAUNCH_EPOCHS,
        earth_leg_tofs=SMALL_EARTH_TOFS,
        hop_tofs=(90.0, 180.0),
        collect_hop_tofs=(180.0, 360.0),
        first_level_limit=50,
    )
    big = RouteSearch(catalogue, ids, SearchSettings(earth_block=1000, **common))
    small = RouteSearch(catalogue, ids, SearchSettings(earth_block=7, **common))
    first_big = [(p.location, p.epoch, p.mass) for p in big._first_level()]
    first_small = [(p.location, p.epoch, p.mass) for p in small._first_level()]
    assert first_big == first_small  # chunked screening changes memory, not results
    assert first_big  # the grids must actually admit Earth legs or the test is vacuous
    first = RouteSearch(catalogue, ids, SearchSettings(**common)).run()
    second = RouteSearch(catalogue, ids, SearchSettings(**common)).run()
    assert first.candidates
    assert [c.summary() for c in first.candidates] == [c.summary() for c in second.candidates]
    assert first.best_by_depth == second.best_by_depth
    for plan in first.candidates:
        # the collection tour starts at the camp asteroid (last deployed) and every stay >= 1 yr
        assert plan.feasible
        for asteroid in plan.asteroids:
            stay = plan.collect_epochs[asteroid] - plan.deploy_epochs[asteroid]
            assert stay >= C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS


@requires_data
def test_search_time_budget_returns_partial_results(catalogue) -> None:
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
    from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

    ids = build_reduced_instance(catalogue).asteroid_ids[:90]
    settings = SearchSettings(
        beam_width=4,
        max_deploys=6,
        neighbours=8,
        launch_epochs=SMALL_LAUNCH_EPOCHS,
        earth_leg_tofs=SMALL_EARTH_TOFS,
        hop_tofs=(180.0,),
        collect_hop_tofs=(180.0, 360.0),
        time_budget_seconds=0.0,
    )
    result = RouteSearch(catalogue, ids, settings).run()
    assert result.depth_reached == 1
    assert any(f.get("reason") == "time budget exhausted" for f in result.failures)


# -- itinerary decoding on a synthetic two-asteroid, self-cleaning ship --


def _state(epoch: float, catalogue, body: int, mass: float) -> StateLine:
    from spacepdhcg.gtoc12.ephemeris import asteroid_state, earth_state

    r, v = earth_state(epoch) if body == 0 else asteroid_state(catalogue, body, epoch)
    return StateLine(epoch, r.copy(), v.copy(), mass)


@requires_data
def test_itinerary_decoding_roles_and_stats(catalogue) -> None:
    from spacepdhcg.gtoc12.references import decode_itineraries, summarise

    a, b = int(catalogue.ids[10]), int(catalogue.ids[20])
    t0 = C.MISSION_START_MJD + 10.0
    ship = ShipTrajectory(1)
    ship.items.append(
        Event(C.EVENT_LAUNCH, _state(t0, catalogue, 0, 3000.0), _state(t0, catalogue, 0, 3000.0))
    )
    ship.items.append(
        Event(a, _state(t0 + 500, catalogue, a, 2600.0), _state(t0 + 500, catalogue, a, 2560.0))
    )
    ship.items.append(
        Event(b, _state(t0 + 700, catalogue, b, 2500.0), _state(t0 + 700, catalogue, b, 2460.0))
    )
    ship.items.append(
        Event(b, _state(t0 + 3000, catalogue, b, 2400.0), _state(t0 + 3000, catalogue, b, 2463.0))
    )
    ship.items.append(
        Event(a, _state(t0 + 3300, catalogue, a, 2300.0), _state(t0 + 3300, catalogue, a, 2376.7))
    )
    t_end = t0 + 3900
    ship.items.append(
        Event(
            C.EVENT_EARTH_FLYBY,
            _state(t_end, catalogue, 0, 2100.0),
            _state(t_end, catalogue, 0, 2100.0 - 139.7),
        )
    )
    itineraries = decode_itineraries(Solution([ship]), catalogue)
    assert len(itineraries) == 1
    it = itineraries[0]
    assert [x for x, _ in it.deploys] == [a, b]
    assert [x for x, _, _ in it.collects] == [b, a]
    assert it.collected_mass_kg == pytest.approx(63.0 + 76.7)
    assert it.unloaded_mass_kg == pytest.approx(139.7)
    assert [leg.role for leg in it.legs] == [
        "earth_out",
        "deploy_hop",
        "collect_hop",
        "collect_hop",
        "earth_return",
    ]
    assert it.legs[1].tof_days == pytest.approx(200.0)
    assert it.legs[1].propellant_kg == pytest.approx(60.0)
    summary = summarise(itineraries, catalogue)
    assert summary["collections_self_cleaning"] == 2
    assert summary["collections_cooperative"] == 0
    assert summary["per_ship_asteroids"]["max"] == 2
    assert summary["stay_days"]["min"] == pytest.approx(2300.0)
