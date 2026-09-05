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
        self.return_sweeps: dict = {}
        self._return_overrides: dict = {}

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
    return_inflation = CollectPairTable.return_inflation
    return_inflation_at = CollectPairTable.return_inflation_at

    def return_override(self, asteroid):
        return self._return_overrides.get(int(asteroid))


def test_harvest_window_cost_is_the_window_minimum_over_both_directions() -> None:
    ids = [11, 12]
    n_t = 40
    # 11 -> 12 is dear except for a cheap window late in the lattice; 12 -> 11 cheap early only
    forward = np.full(n_t, 3.0)
    forward[30:34] = 0.4
    backward = np.full(n_t, 3.0)
    backward[2:6] = 0.3
    table = _FakeTable(ids, {(11, 12): forward, (12, 11): backward}, n_t=n_t)
    # every ΔV flyable: the test is about the window minimum, not the authority limit
    table.settings = dataclasses.replace(table.settings, hop_authority_ratio=100.0)
    table._geometry = {}
    _FakeTable.harvest_window_cost = CollectPairTable.harvest_window_cost
    early = (float(table.epochs[0]), float(table.epochs[8]))
    late = (float(table.epochs[28]), float(table.epochs[36]))
    mass = 1400.0
    cost = table.harvest_window_cost
    cheap_early = cost(11, 12, mass, window=early, max_tof_days=200.0)
    cheap_late = cost(12, 11, mass, window=late, max_tof_days=200.0)
    dear = cost(11, 12, mass, window=(early[1], late[0]), max_tof_days=200.0)
    # the direction is the table's choice: early -> the 12 -> 11 cell, late -> 11 -> 12
    one = np.asarray([120.0])
    expected_early = float(table.hop_propellant(np.asarray([[0.3]]), mass, one)[0, 0])
    expected_late = float(table.hop_propellant(np.asarray([[0.4]]), mass, one)[0, 0])
    assert cheap_early == pytest.approx(expected_early)
    assert cheap_late == pytest.approx(expected_late)
    assert dear > 5.0 * cheap_late
    # too short a TOF cap -> nothing flyable -> inf (the beam charges its deterrent, not a prune)
    assert cost(11, 12, mass, window=late, max_tof_days=60.0) == math.inf
    # an empty window is inf as well
    assert cost(11, 12, mass, window=(late[1], late[0]), max_tof_days=200.0) == math.inf


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


def test_collect_dp_prices_the_return_strictly_from_certified_sweep_cells() -> None:
    """With a return sweep set for the camp the DP may only end the tour on a certified cell:
    the return departs where the override is ``ok`` (even though the model would rather leave
    at the last lattice epoch), it is priced at the cell's measured inflation, and a sweep with
    no certified cell closes no tour at all."""

    ids = [11, 12, 13]
    cheap, dear = 0.3, 3.0
    costs = {
        (13, 11): cheap,
        (11, 12): cheap,
        (12, 13): cheap,
        (13, 12): dear,
        (12, 11): dear,
        (11, 13): dear,
    }
    deployed = [(11, T0 + 5.0 * YEAR), (12, T0 + 5.5 * YEAR), (13, T0 + 6.0 * YEAR)]
    camp_epoch = T0 + 6.0 * YEAR
    table = _FakeTable(ids, costs)
    free = plan_collect_tour(table, deployed, 13, camp_epoch, 1500.0)
    assert free is not None and free.return_departure == pytest.approx(table.epochs[-1])
    # certified cells: a single departure epoch, well before the lattice end, out of the last
    # collected asteroid 12 (the cheap cycle ends there); every other cell of every camp is
    # infeasible (a sweep in hand for every asteroid, all refused but that one cell)
    n_t, n_k = table.epochs.shape[0], table.return_tofs.shape[0]
    certified_index = 30
    ok = np.zeros((n_t, n_k), dtype=bool)
    ok[certified_index, 0] = True
    inflation = np.where(ok, 1.37, np.nan)
    refused = (np.full((n_t, n_k), np.nan), np.zeros((n_t, n_k), dtype=bool))
    for asteroid in ids:
        table._return_overrides[asteroid] = refused
        table.return_sweeps[asteroid] = object()
    table._return_overrides[12] = (inflation, ok)
    swept = plan_collect_tour(table, deployed, 13, camp_epoch, 1500.0)
    assert swept is not None
    assert swept.order[-1] == 12
    assert swept.return_departure == pytest.approx(table.epochs[certified_index])
    # the last collect moved with the return (collected on departure), the others are free
    assert swept.collect_epochs[12] == pytest.approx(swept.return_departure)
    assert swept.collected_proxy_kg < free.collected_proxy_kg
    # priced at the cell's measured inflation, not the model's, and no authority limit applied;
    # the DP's move mass is the camp mass + the miners mined to the window end - the burn
    # schedule of the two hops flown before the return
    mass = (
        1500.0
        + sum(C.maximum_collected_mass(table.epochs[-1] - e) for _a, e in deployed)
        - swept.diagnostics.get("burn_per_hop_kg", 0.0) * 2
    )
    from spacepdhcg.gtoc12.screening import propellant_for_delta_v

    measured = float(propellant_for_delta_v(mass, table.return_dv * 1.37))
    hop_propellant = sum(swept.hop_propellant_kg)
    assert swept.propellant_proxy_kg == pytest.approx(hop_propellant + measured, rel=1e-6)
    assert table.return_inflation_at(12, swept.return_departure, 300.0, 1.0, mass) == 1.37
    assert table.return_inflation_at(11, swept.return_departure, 300.0, 1.0, mass) != 1.37
    # a camp whose sweep refused every cell cannot end the tour (strict: the model is not a
    # fallback); a camp without a sweep is priced by the model and still closes
    table._return_overrides[12] = refused
    assert plan_collect_tour(table, deployed, 13, camp_epoch, 1500.0) is None
    del table.return_sweeps[13], table._return_overrides[13]
    modelled = plan_collect_tour(table, deployed, 13, camp_epoch, 1500.0)
    assert modelled is not None and modelled.order[-1] == 13


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


@requires_data
def test_tighter_collect_window_radius_gives_smaller_denser_families() -> None:
    from spacepdhcg.gtoc12.bundles import family_clusters

    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids
    tight = family_clusters(
        catalogue, ids, bands=ClusterBands.collect_window(radius=1.5), min_members=4
    )
    loose = family_clusters(
        catalogue, ids, bands=ClusterBands.collect_window(radius=2.5), min_members=4
    )
    assert tight and loose
    assert max(len(m) for _l, m in tight) <= max(len(m) for _l, m in loose)
    # every member of a tight family is within the tight radius of some other member
    # (the families are connected components of the radius graph), so the harvest-window phase
    # spread inside a tight family is bounded by the band x radius x members
    clusters = ComovingClusters(catalogue, ids, ClusterBands.collect_window(radius=1.5))
    for _label, members in tight[:5]:
        member_set = {int(a) for a in members}
        for a in members:
            assert any(int(n) in member_set for n in clusters.neighbours(int(a)))


# -- mass schedule, phasing lattice and return grid of the DP --------------------------------


def test_collect_dp_second_pass_credits_the_burnt_propellant() -> None:
    ids = [1, 2, 3]
    costs = {(a, b): 1.5 for a in ids for b in ids if a != b}
    table = _FakeTable(ids, costs, return_dv=3.0)
    deployed = [(1, T0 + 4.0 * YEAR), (2, T0 + 4.5 * YEAR), (3, T0 + 6.0 * YEAR)]
    tour = plan_collect_tour(table, deployed, 3, T0 + 6.0 * YEAR, 1500.0)
    assert tour is not None
    assert len(tour.hop_propellant_kg) == len(tour.hops)
    burn = tour.diagnostics["burn_per_hop_kg"]
    assert burn > 0.0 and burn == pytest.approx(np.mean(tour.hop_propellant_kg), rel=0.5)
    # pass 1 priced the same hops on a heavier ship: its objective is worse
    assert tour.diagnostics["pass1_objective_kg"] < tour.objective_kg
    # the fixed-schedule call reproduces the second pass exactly
    fixed = plan_collect_tour(table, deployed, 3, T0 + 6.0 * YEAR, 1500.0, burn_per_hop=burn)
    assert fixed is not None
    assert fixed.order == tour.order and fixed.objective_kg == pytest.approx(tour.objective_kg)
    # a heavier schedule (no credit) prices the same hops dearer than the credited one
    heavy = plan_collect_tour(table, deployed, 3, T0 + 6.0 * YEAR, 1500.0, burn_per_hop=0.0)
    assert heavy is not None and heavy.propellant_proxy_kg > tour.propellant_proxy_kg


def test_collect_dp_heavy_return_only_closes_with_the_burn_credit() -> None:
    """A return at the authority limit for the heavy pass-1 ship is flyable for the ship that
    has burnt its hop propellant: the two-pass DP finds the tour the one-pass DP refused."""

    from spacepdhcg.gtoc12.screening import thrust_authority_km_s

    ids = [1, 2, 3, 4]
    costs = {(a, b): 2.0 for a in ids for b in ids if a != b}  # ~100 kg per hop
    deployed = [(a, T0 + (3.0 + 0.5 * i) * YEAR) for i, a in enumerate(ids)]
    probe = _FakeTable(ids, costs, n_t=30)
    heavy = 1500.0 + sum(C.maximum_collected_mass(probe.epochs[-1] - e) for _a, e in deployed)
    # return ΔV at 1.02 x the 0.5 authority limit of the heavy (no burn credit) ship over 300 d:
    # unflyable for pass 1, flyable once ~300 kg of hop propellant are credited
    return_dv = 0.5 * float(thrust_authority_km_s(heavy, 300.0, 1.0)) * 1.02
    table = _FakeTable(ids, costs, return_dv=return_dv, n_t=30)
    single = plan_collect_tour(table, deployed, 4, T0 + 4.5 * YEAR, 1500.0, burn_per_hop=0.0)
    assert single is None
    two_pass = plan_collect_tour(table, deployed, 4, T0 + 4.5 * YEAR, 1500.0, burn_per_hop=100.0)
    assert two_pass is not None and sorted(two_pass.order) == ids


@requires_data
def test_default_dp_lattice_resolves_fifteen_day_phasing_and_thirty_day_returns() -> None:
    settings = CollectDPSettings()
    assert settings.step_days == 15.0
    assert 60.0 in settings.tofs and 180.0 in settings.tofs
    assert settings.return_tofs[0] == 240.0 and settings.return_tofs[-1] == 720.0
    assert np.all(np.diff(settings.return_tofs) == 30.0)
    catalogue = load_catalogue()
    table = CollectPairTable(catalogue, settings)
    assert np.all(np.diff(table.epochs) == 15.0)
    ids = build_reduced_instance(catalogue).asteroid_ids[:2].tolist()
    hop = table.hop(ids[0], ids[1])
    # a 15-day lattice sees the relative-phase window a 30-day lattice straddles: the cheapest
    # departure of the fine lattice is never dearer than the coarse lattice's
    coarse = CollectPairTable(
        catalogue, CollectDPSettings(step_days=30.0, tofs=(60.0, 90.0, 180.0, 240.0))
    ).hop(ids[0], ids[1])
    assert (
        np.nanmin(np.where(np.isfinite(hop), hop, np.nan))
        <= np.nanmin(np.where(np.isfinite(coarse), coarse, np.nan)) + 1e-6
    )


# -- calibrated hop inflation --------------------------------------------------------------


def _synthetic_hops(n: int, seed: int):
    from spacepdhcg.gtoc12.hopcalib import HopSamples

    rng = np.random.default_rng(seed)
    mass = rng.uniform(1200.0, 2400.0, n)
    tof = rng.choice([90.0, 120.0, 180.0, 240.0, 300.0, 360.0], n)
    from spacepdhcg.gtoc12.screening import thrust_authority_km_s

    ratio = rng.uniform(0.05, 0.6, n)
    lambert = ratio * thrust_authority_km_s(mass, tof, 1.0)
    delta_a = rng.normal(0.0, 0.02, n)
    delta_l = rng.normal(0.0, 0.15, n)
    truth = 1.0 + 0.7 * ratio + 0.05 * tof / YEAR + 0.3 * np.abs(delta_l) / np.pi
    scvx = lambert * (truth + rng.normal(0.0, 0.02, n))
    return HopSamples(
        np.arange(1, n + 1, dtype=np.int64),
        np.arange(2, n + 2, dtype=np.int64),
        np.full(n, T0 + 8.0 * YEAR),
        tof,
        mass,
        lambert,
        scvx,
        delta_a,
        delta_l,
        ["synthetic"] * n,
    )


def test_inflation_fit_recovers_the_model_and_reports_holdout_residuals() -> None:
    from spacepdhcg.gtoc12.hopcalib import InflationFit, fit_inflation

    train, holdout = _synthetic_hops(600, 0), _synthetic_hops(300, 1)
    fit = fit_inflation(train, holdout, quantile=0.5)
    c = fit.coefficients
    assert c[1] == pytest.approx(0.7, abs=0.05)  # authority-ratio slope
    assert c[4] == pytest.approx(0.3, abs=0.1)  # phase-difference slope
    assert abs(c[3]) < 0.1  # Δa carries no signal here
    stats = fit.residuals
    assert stats["holdout"]["rms"] < 0.03 and abs(stats["holdout"]["median"]) < 0.01
    assert stats["holdout"]["n"] == 300 and "holdout_propellant_error_kg" in stats
    # a higher quantile only shifts the constant, towards heavier (conservative) pricing
    heavier = fit_inflation(train, holdout, quantile=0.9)
    assert heavier.coefficients[0] > c[0]
    assert heavier.coefficients[1:] == pytest.approx(c[1:])
    assert heavier.residuals["train"]["share_under_priced"] <= 0.12
    # round trip through the JSON summary and vectorised evaluation on a (n_t, n_tof) table
    again = InflationFit.from_summary(fit.summary())
    assert again.coefficients == fit.coefficients
    dv = np.full((7, 3), 1.0)
    tofs = np.array([90.0, 180.0, 360.0])
    table = again.inflation(dv, 1500.0, np.broadcast_to(tofs, dv.shape), 0.01, np.zeros((7, 1)))
    assert table.shape == (7, 3) and np.all(table >= 1.0)
    # more TOF at the same ΔV lowers the ratio term: slower hops are cheaper per km/s
    assert table[0, 2] < table[0, 0]


def test_fake_table_hop_propellant_uses_the_fit_per_pair_and_epoch() -> None:
    from spacepdhcg.gtoc12.hopcalib import InflationFit

    ids = [1, 2, 3]
    costs = {(a, b): 1.0 for a in ids for b in ids if a != b}
    flat = _FakeTable(ids, costs)
    fitted = _FakeTable(ids, costs)
    fitted.settings = dataclasses.replace(
        fitted.settings, inflation_fit=InflationFit((0.9, 0.5, 0.0, 0.0, 0.0), 0.5)
    )
    fitted.pair_geometry = lambda s, t, epochs: (0.0, np.zeros(len(epochs)))
    dv = np.full((5, 2), 1.0)
    without = CollectPairTable.hop_propellant(flat, dv, 1500.0, flat.tofs)
    with_fit = CollectPairTable.hop_propellant(
        fitted, dv, 1500.0, fitted.tofs, pair=(1, 2), epochs=fitted.epochs[:5]
    )
    assert without.shape == with_fit.shape == (5, 2)
    # flat 1.2 vs 0.9 + 0.5 r with r << 0.6: the fit prices these slow hops cheaper
    assert np.all(with_fit < without)
    # without pair/epochs the fit is not applied (falls back to the flat factor)
    fallback = CollectPairTable.hop_propellant(fitted, dv, 1500.0, fitted.tofs)
    np.testing.assert_allclose(fallback, without)
