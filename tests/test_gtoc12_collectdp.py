"""Exact collect-tour DP, its pair-cost table, collect-epoch families and the master LP bound."""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.clusters import ClusterBands, ComovingClusters
from spacepdhcg.gtoc12.collectdp import (
    CollectDPSettings,
    CollectPairTable,
    plan_collect_tour,
)
from spacepdhcg.gtoc12.cooperative import (
    FleetColumn,
    _LpModel,
    fleet_feasible,
    lp_branch_and_bound,
    lp_fleet_bound,
    ship_rule_mass_floor,
    solve_fleet_master,
)
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
YEAR = C.YEAR_DAYS
T0 = C.MISSION_START_MJD
COARSE = CollectDPSettings(step_days=60.0, tofs=(120.0, 180.0, 240.0, 360.0, 480.0))


# -- master LP bound -----------------------------------------------------------------------


def _column(identifier, deploys, collects, mass, *, foreign=None, ships=1):
    column = FleetColumn(
        identifier,
        1,
        f"c{identifier}",
        dict(deploys),
        dict(collects),
        dict(foreign or {}),
        {a: mass for a in collects},
        True,
    )
    if ships > 1:
        members = tuple(
            FleetColumn(
                identifier * 100 + k,
                1,
                f"c{identifier}_{k}",
                {a: e for a, e in deploys.items() if a % ships == k},
                {a: e for a, e in collects.items() if a % ships == k},
                {},
                {a: mass for a in collects if a % ships == k},
                True,
            )
            for k in range(ships)
        )
        column = FleetColumn.from_bundle(identifier, f"b{identifier}", members)
    return column


def _brute_force(columns: list[FleetColumn], weights=None) -> float:
    best = 0.0
    for r in range(1, len(columns) + 1):
        for subset in itertools.combinations(columns, r):
            if fleet_feasible(subset) == "":
                best = max(best, sum(c.value(weights) for c in subset))
    return best


def _random_columns(seed: int, count: int, asteroids: int) -> list[FleetColumn]:
    rng = np.random.default_rng(seed)
    columns = []
    for k in range(count):
        size = int(rng.integers(2, 6))
        ids = [int(a) for a in rng.choice(asteroids, size=size, replace=False) + 1]
        deploys = {a: T0 + 400.0 + 30.0 * i for i, a in enumerate(ids)}
        # some columns leave their last miner as an orphan; some collect a foreign one
        collected_ids = ids[:-1] if k % 3 == 0 else ids
        collects = {a: T0 + 3500.0 + 30.0 * i for i, a in enumerate(collected_ids)}
        mass = float(rng.uniform(40.0, 90.0))
        foreign = {}
        if k % 4 == 1 and k > 0:
            other = columns[k - 1]
            orphan = [a for a in other.deploys if a not in other.collects]
            if orphan and orphan[0] not in deploys:
                foreign = {orphan[0]: other.deploys[orphan[0]]}
                collects[orphan[0]] = T0 + 3900.0
        columns.append(_column(k, deploys, collects, mass, foreign=foreign))
    return columns


def test_ship_rule_mass_floor_matches_the_rule() -> None:
    assert ship_rule_mass_floor(1) == 0.0 and ship_rule_mass_floor(2) == 0.0
    for ships in (3, 5, 16, 40):
        floor = ship_rule_mass_floor(ships)
        mean = floor / ships
        # exactly on the rule boundary: N == 2 exp(rho * mean)
        assert C.maximum_ship_count(mean) == pytest.approx(ships, rel=1e-9)
        assert C.maximum_ship_count(mean * 0.999) < ships


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lp_bound_is_valid_and_lp_branch_and_bound_is_exact(seed: int) -> None:
    columns = _random_columns(seed, 9, 14)
    weights = {a: 1.0 + 0.1 * (a % 3) for a in range(1, 15)}
    optimum = _brute_force(columns, weights)
    usable = sorted(columns, key=lambda c: (-c.value(weights), c.identifier))
    values = [c.value(weights) for c in usable]
    relaxation = lp_fleet_bound(usable, values)
    assert relaxation.bound >= optimum - 1e-6
    # every per-N LP bounds the best N-ship fleet, and the dual node bound at the root equals
    # (weak duality, any dual vector) at least the LP value it came from
    zero = np.zeros(relaxation.sizes.shape[0])
    root = relaxation.node_bound(0, 0.0, 0, 0.0, zero, C.MAX_SHIPS, np.ones(len(usable), bool))
    assert root >= relaxation.bound - 1e-6
    # the LP branch and bound (from a zero incumbent) finds and proves the optimum
    model = _LpModel(usable, values)
    result = lp_branch_and_bound(model, relaxation.relaxations, 0.0, node_limit=5000)
    assert result.proven
    assert result.value == pytest.approx(optimum, abs=1e-6)
    if result.selection is not None:
        chosen = tuple(usable[i] for i in result.selection)
        assert fleet_feasible(chosen) == ""
        assert sum(c.value(weights) for c in chosen) == pytest.approx(optimum, abs=1e-6)
    # the full master agrees and reports a closed gap
    master = solve_fleet_master(columns, weights=weights, node_cap=100_000)
    assert master.objective == pytest.approx(optimum, abs=1e-6)
    assert master.proven and master.upper_bound == pytest.approx(optimum, abs=1e-6)
    assert master.lp_bound >= optimum - 1e-6
    summary = master.summary()
    assert summary["proven_optimal"] and summary["lp_gap_kg"] >= -1e-6


def test_master_with_a_tiny_node_cap_is_rescued_by_the_lp_branch_and_bound() -> None:
    columns = _random_columns(7, 12, 18)
    optimum = _brute_force(columns)
    capped = solve_fleet_master(columns, node_cap=3, lp_node_limit=5000)
    assert not capped.exhaustive
    assert capped.objective == pytest.approx(optimum, abs=1e-6)
    assert capped.proven and capped.lp_proven
    # without the LP the same cap leaves a gap; the bound stays valid either way
    plain = solve_fleet_master(columns, node_cap=3, lp_bound=False)
    assert plain.upper_bound >= optimum - 1e-6 and not plain.proven
    # an LP node limit too small to close leaves a valid, finite bound
    stuck = solve_fleet_master(columns, node_cap=3, lp_node_limit=1)
    assert stuck.upper_bound >= optimum - 1e-6 and math.isfinite(stuck.upper_bound)
    assert stuck.lp_bound >= stuck.objective - 1e-6


def test_bundle_columns_count_their_ships_in_the_lp() -> None:
    six = range(1, 7)
    bundle = _column(0, {k: T0 + 400.0 for k in six}, {k: T0 + 3500.0 for k in six}, 30.0, ships=2)
    solo = _column(1, {9: T0 + 400.0}, {9: T0 + 3500.0}, 30.0)
    assert bundle.ships == 2
    usable = [bundle, solo]
    relaxation = lp_fleet_bound(usable, [c.value(None) for c in usable])
    # three light ships (30 kg per asteroid) break the rule: 2 exp(0.004 x 70) = 2.65 < 3
    assert 3 not in relaxation.relaxations
    assert relaxation.relaxations[2] == pytest.approx(180.0)
    assert relaxation.bound == pytest.approx(180.0)


# -- collect DP (synthetic table) ----------------------------------------------------------


class _FakeTable:
    """Pair table with hand-made costs: a (n_t, n_tof) ΔV per ordered pair, one return table."""

    def __init__(self, ids, costs, step=60.0, n_t=40, tofs=(120.0, 180.0), return_dv=1.0):
        self.settings = CollectDPSettings(
            step_days=step, tofs=tofs, return_tofs=(300.0,), propellant_weight=1.0
        )
        self.epochs = T0 + 6.0 * YEAR + step * np.arange(n_t)
        self.tofs = np.asarray(tofs)
        self.tof_steps = np.rint(self.tofs / step).astype(np.int64)
        self.return_tofs = np.asarray([300.0])
        self.ids = ids
        self.costs = costs
        self.return_dv = return_dv
        self.hops_requested: list[tuple[int, int]] = []

    def index_at_or_after(self, epoch):
        return int(np.searchsorted(self.epochs, epoch - 1e-9))

    def hop(self, source, target):
        self.hops_requested.append((source, target))
        cost = np.asarray(self.costs[(source, target)], dtype=np.float64)
        table = np.broadcast_to(
            cost[:, None] if cost.ndim else cost, (self.epochs.shape[0], self.tofs.shape[0])
        )
        return np.array(table, dtype=np.float32)

    def earth_return(self, source):
        return np.full((self.epochs.shape[0], 1), self.return_dv, dtype=np.float32)

    hop_propellant = CollectPairTable.hop_propellant
    return_propellant = CollectPairTable.return_propellant


def test_collect_dp_bookkeeping_min_stay_single_collect_and_free_order() -> None:
    ids = [11, 12, 13]
    cheap, dear = 0.3, 3.0
    # camp = 13; the cheap cycle is 13 -> 11 -> 12 (reverse of the deploy order is dear)
    costs = {
        (13, 11): cheap,
        (11, 12): cheap,
        (12, 13): cheap,
        (13, 12): dear,
        (12, 11): dear,
        (11, 13): dear,
    }
    table = _FakeTable(ids, costs)
    deployed = [(11, T0 + 5.0 * YEAR), (12, T0 + 5.5 * YEAR), (13, T0 + 6.0 * YEAR)]
    tour = plan_collect_tour(table, deployed, 13, T0 + 6.0 * YEAR, 1500.0)
    assert tour is not None
    assert tour.order == (13, 11, 12)  # order chosen by cost, not by deployment
    assert not tour.reposition
    # each asteroid collected exactly once, after at least one year of mining, on departure
    assert sorted(tour.collect_epochs) == ids
    for asteroid, epoch in deployed:
        assert tour.collect_epochs[asteroid] - epoch >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-9
    departures = {source: departure for source, _target, departure, _tof, _dv in tour.hops}
    for asteroid in tour.order[:-1]:
        assert tour.collect_epochs[asteroid] == pytest.approx(departures[asteroid])
    assert tour.collect_epochs[tour.order[-1]] == pytest.approx(tour.return_departure)
    # the last collect is as late as the lattice allows (rate x stay is the whole objective
    # once the cheap cycle is fixed): the return leaves at the last lattice epoch
    assert tour.return_departure == pytest.approx(table.epochs[-1])
    # mass bookkeeping: the collected proxy is rate x stay for every miner
    expected = sum(C.maximum_collected_mass(tour.collect_epochs[a] - e) for a, e in deployed)
    assert tour.collected_proxy_kg == pytest.approx(expected)
    # objective = collected - weight x propellant, so the propellant recovered is consistent
    assert tour.objective_kg == pytest.approx(expected - tour.propellant_proxy_kg)
    # arrivals never precede departures along the tour
    epoch = T0 + 6.0 * YEAR
    for _source, _target, departure, tof, _dv in tour.hops:
        assert departure >= epoch - 1e-9
        epoch = departure + tof
    assert tour.return_departure >= epoch - 1e-9


def test_collect_dp_leaves_the_camp_uncollected_when_a_revisit_is_cheaper() -> None:
    ids = [1, 2, 3]
    # camp 3 deployed at the camp epoch: it cannot be collected for a year, but the hop 3 -> 1
    # is cheap only in the first few lattice epochs (a phasing window that closes); leaving
    # the camp uncollected at once and coming back (3 -> 1 -> 2 -> 3) is the cheap tour
    n_t = 40
    window = np.full(n_t, 3.0)
    window[:4] = 0.2
    costs = {(a, b): 3.0 for a in ids for b in ids if a != b}
    costs.update({(3, 1): window, (1, 2): 0.2, (2, 3): 0.2})
    table = _FakeTable(ids, costs, n_t=n_t)
    deployed = [(1, T0 + 4.0 * YEAR), (2, T0 + 4.5 * YEAR), (3, T0 + 6.0 * YEAR)]
    tour = plan_collect_tour(table, deployed, 3, T0 + 6.0 * YEAR, 1500.0)
    assert tour is not None
    assert tour.reposition and tour.order == (1, 2, 3)
    # the repositioning hop leaves the camp without collecting it
    first = tour.hops[0]
    assert first[0] == 3 and first[1] == 1
    assert abs(tour.collect_epochs[3] - first[2]) > 1.0
    assert tour.collect_epochs[3] == pytest.approx(tour.return_departure)
    assert len(tour.hops) == 3  # reposition + two collect hops


def test_collect_dp_returns_none_when_no_tour_fits() -> None:
    ids = [1, 2]
    table = _FakeTable(ids, {(1, 2): 50.0, (2, 1): 50.0})  # unflyable hops
    deployed = [(1, T0 + 4.0 * YEAR), (2, T0 + 6.0 * YEAR)]
    assert plan_collect_tour(table, deployed, 2, T0 + 6.0 * YEAR, 1500.0) is None
    # too many asteroids for the settings
    small = CollectDPSettings(max_asteroids=1)
    table.settings = small
    assert plan_collect_tour(table, deployed, 2, T0 + 6.0 * YEAR, 1500.0) is None


def test_collect_dp_settings_reject_off_lattice_tofs() -> None:
    with pytest.raises(ValueError, match="multiple"):
        CollectDPSettings(step_days=30.0, tofs=(100.0,))


# -- pair table on the real catalogue ------------------------------------------------------


@requires_data
def test_pair_table_is_deterministic_cached_and_bounded() -> None:
    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids[:4].tolist()
    settings = CollectDPSettings(step_days=60.0, tofs=(120.0, 240.0), cache_pairs=3)
    table = CollectPairTable(catalogue, settings)
    first = table.hop(ids[0], ids[1]).copy()
    evaluations = table.lambert_evaluations
    again = table.hop(ids[0], ids[1])
    assert table.lambert_evaluations == evaluations  # cached
    np.testing.assert_array_equal(first, again)
    fresh = CollectPairTable(catalogue, settings).hop(ids[0], ids[1])
    np.testing.assert_array_equal(first, fresh)  # deterministic across instances
    assert first.dtype == np.float32 and first.shape == (table.epochs.shape[0], 2)
    finite = first[np.isfinite(first)]
    assert finite.size > 0 and np.all(finite >= 0.0)
    # bounded cache: the oldest pair is evicted once more than ``cache_pairs`` are held
    table.hop(ids[0], ids[2])
    table.hop(ids[0], ids[3])
    assert table.cached_pairs == 3
    table.hop(ids[1], ids[2])
    assert table.cached_pairs == 3
    assert (ids[0], ids[1]) not in table._hops
    # Earth returns respect the end of the window (inf beyond it)
    ret = table.earth_return(ids[0])
    assert ret.shape == (table.epochs.shape[0], len(settings.return_tofs))
    late = table.epochs[:, None] + table.return_tofs[None, :] > C.MISSION_END_MJD
    assert np.all(~np.isfinite(ret[late]))


@requires_data
def test_beam_completion_keeps_the_dp_tour_when_it_scores_better() -> None:
    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids
    settings = SearchSettings(
        beam_width=6,
        max_deploys=4,
        launch_epochs=tuple(float(x) for x in T0 + np.arange(0.0, 731.0, 180.0)),
        earth_leg_tofs=(600.0, 750.0, 900.0),
        collect_dp_step_days=60.0,
        first_level_limit=200,
    )
    search = RouteSearch(catalogue, ids, settings)
    result = search.run()
    stats = search.collect_dp_stats
    assert stats["priced"] > 0 and stats["won"] + stats["failed"] <= stats["priced"]
    assert stats["seconds"] > 0.0
    # the shared table has been used and stays bounded
    assert search.collect_table.cached_pairs <= settings.collect_dp_cache_pairs
    for plan in result.candidates:
        assert plan.feasible
        assert set(plan.deploy_epochs) == set(plan.collect_epochs)
        for asteroid in plan.deploy_epochs:
            stay = plan.collect_epochs[asteroid] - plan.deploy_epochs[asteroid]
            assert stay >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
    # switching the DP off is the previous behaviour and never scores a chain higher
    off = RouteSearch(catalogue, ids, dataclasses.replace(settings, collect_dp=False))
    baseline = off.run()
    assert off.collect_dp_stats["priced"] == 0
    if result.best is not None and baseline.best is not None:
        assert search.plan_score(result.best) >= off.plan_score(baseline.best) - 1e-6


# -- collect-epoch families ----------------------------------------------------------------


@requires_data
def test_collect_window_families_weight_harvest_epoch_co_motion() -> None:
    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids
    bands = ClusterBands.collect_window(radius=2.0)
    assert len(bands.phase_epochs) == 4 and bands.epoch_weights == (0.5, 1.0, 1.0, 1.0)
    assert bands.phase_epochs[1] > T0 + 8.0 * YEAR and bands.phase_epochs[-1] < T0 + 14.0 * YEAR
    clusters = ComovingClusters(catalogue, ids, bands)
    default = ComovingClusters(catalogue, ids, ClusterBands(radius=2.0))
    # the feature vector has one (cos, sin) pair per epoch, weighted
    assert clusters.features.shape[1] == 5 + 2 * 4 and default.features.shape[1] == 5 + 2 * 2
    summary = clusters.summary()
    assert summary["bands"]["phase_weights"] == [0.5, 1.0, 1.0, 1.0]
    # neighbours under the collect-window bands are co-located at the harvest epochs: their
    # phase difference at the three collect epochs stays inside the band
    checked = 0
    for source in clusters.ids[:40]:
        for target in clusters.neighbours(int(source))[:3]:
            for epoch in bands.phase_epochs[1:]:
                delta = abs(clusters.phase_difference_deg(int(source), int(target), epoch))
                assert delta <= bands.radius * bands.phase_deg * 1.05 + 1e-9
            checked += 1
    assert checked > 0
    with pytest.raises(ValueError, match="one entry per visit epoch"):
        ClusterBands(visit_epochs=(T0, T0 + YEAR), phase_weights=(1.0,)).epoch_weights
