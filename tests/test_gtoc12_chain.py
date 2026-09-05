"""Ninth iteration: chain-level tour scoring in the beam, LP-dual feedback, reference prior."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math

import numpy as np
import pytest
from test_gtoc12_collectdp import _FakeTable, _random_columns

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.chainprior import (
    REFERENCE_FILES,
    ChainPrior,
    extract_chain_prior,
    load_chain_prior,
    reference_solution_files,
)
from spacepdhcg.gtoc12.collectdp import plan_collect_tour
from spacepdhcg.gtoc12.cooperative import lp_asteroid_prices, solve_fleet_master
from spacepdhcg.gtoc12.data import data_available, data_directory, load_catalogue
from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings, _Partial

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
YEAR = C.YEAR_DAYS
T0 = C.MISSION_START_MJD


# -- tour-cost scoring: exactness against brute force ---------------------------------------


def _brute_force_tour(table: _FakeTable, deployed, camp, camp_epoch, mass, w=1.0):
    """Every collect order (camp first, or camp left uncollected and collected later), every
    lattice departure, every hop TOF and the single return TOF, priced with the DP's own leg
    model at burn 0 (mass on a move = camp mass + the collected set mined to the window end)."""

    ids = [a for a, _ in deployed]
    deploy = dict(deployed)
    t0 = table.index_at_or_after(camp_epoch)
    epochs = table.epochs[t0:]
    n_t = epochs.shape[0]
    min_stay = C.MIN_MINING_STAY_YEARS * YEAR
    mined_end = {a: C.maximum_collected_mass(max(epochs[-1] - deploy[a], 0.0)) for a in ids}

    def mined(a: int, t: int) -> float:
        stay = epochs[t] - deploy[a]
        if stay < min_stay - 1e-9:
            return -math.inf
        return C.MINING_RATE_KG_PER_YEAR * stay / YEAR

    def hop_cost(a: int, b: int, t: int, k: int, move_mass: float) -> float:
        dv = float(table.hop(a, b)[t0 + t, k])
        cost = table.hop_propellant(np.asarray([[dv]]), move_mass, np.asarray([table.tofs[k]]))
        return float(cost[0, 0])

    def departures(i: int, earliest: int, steps: list[int]):
        """Departure lattice indices for stops ``i..``: each at or after the previous arrival
        (camping allowed); the last stop's departure is the Earth-return departure."""

        if i == len(steps) + 1:
            yield ()
            return
        for t in range(earliest, n_t):
            if i < len(steps):
                arrival = t + steps[i]
                if arrival >= n_t:
                    break
            else:
                arrival = t
            for rest in departures(i + 1, arrival, steps):
                yield (t, *rest)

    best = -math.inf
    for order in itertools.permutations(ids):
        skip = order[0] != camp
        # locations visited in flight order: the camp, then the collect order
        stops = list(order) if not skip else [camp, *order]
        n_hops = len(stops) - 1
        for tofs in itertools.product(range(table.tofs.shape[0]), repeat=n_hops):
            steps = [int(table.tof_steps[k]) for k in tofs]
            for deps in departures(0, 0, steps):
                value = 0.0
                collected: set[int] = set()
                for i, stop in enumerate(stops):
                    t = deps[i]
                    if i == 0 and skip:
                        move_mass = mass
                    else:
                        collected.add(stop)
                        gain = mined(stop, t)
                        if not math.isfinite(gain):
                            value = -math.inf
                            break
                        value += gain
                        move_mass = max(
                            mass + sum(mined_end[a] for a in collected), C.DRY_MASS_KG + 1.0
                        )
                    if i < n_hops:
                        cost = hop_cost(stop, stops[i + 1], t, tofs[i], move_mass)
                    else:
                        ret = float(table.earth_return(stop)[t0 + t, 0])
                        cost = float(
                            table.return_propellant(
                                np.asarray([[ret]]), move_mass, table.return_tofs
                            )[0, 0]
                        )
                    if not math.isfinite(cost):
                        value = -math.inf
                        break
                    value -= w * cost
                best = max(best, value)
    return best


def test_collect_tour_dp_matches_brute_force_on_a_small_lattice() -> None:
    ids = [11, 12, 13]
    n_t = 12
    rng = np.random.default_rng(7)
    # epoch-dependent pair costs with phasing windows, so order *and* timing matter
    costs = {}
    for a in ids:
        for b in ids:
            if a != b:
                profile = 1.0 + 1.5 * rng.random(n_t)
                profile[rng.integers(0, n_t - 3) : rng.integers(n_t - 3, n_t)] *= 0.3
                costs[(a, b)] = profile
    table = _FakeTable(ids, costs, n_t=n_t, tofs=(120.0, 180.0), return_dv=1.2)
    deployed = [(11, T0 + 4.5 * YEAR), (12, T0 + 5.0 * YEAR), (13, T0 + 5.5 * YEAR)]
    camp, camp_epoch, mass = 13, T0 + 6.0 * YEAR, 1500.0
    for w in (1.0, 0.15):
        tour = plan_collect_tour(
            table, deployed, camp, camp_epoch, mass, propellant_weight=w, burn_per_hop=0.0
        )
        brute = _brute_force_tour(table, deployed, camp, camp_epoch, mass, w=w)
        assert tour is not None and math.isfinite(brute)
        assert tour.objective_kg == pytest.approx(brute, abs=1e-6)
        # the tour's components re-assemble its objective
        assert tour.objective_kg == pytest.approx(
            tour.collected_proxy_kg - w * tour.propellant_proxy_kg, abs=1e-6
        )
    # deterministic: the same table gives the same tour bit for bit
    again = plan_collect_tour(table, deployed, camp, camp_epoch, mass, burn_per_hop=0.0)
    first = plan_collect_tour(table, deployed, camp, camp_epoch, mass, burn_per_hop=0.0)
    assert again is not None and first is not None
    assert again.order == first.order and again.hops == first.hops


# -- chain score arithmetic and prices --------------------------------------------------------


class _StubSearch(RouteSearch):
    """RouteSearch whose collect table is a hand-made one (no catalogue needed)."""

    def __init__(self, table, settings, prices=None, prior=None):
        # bypass the catalogue-backed constructor: set what the scoring path reads
        self.catalogue = None
        self.settings = settings
        self.weights = {}
        self.asteroid_prices = dict(prices or {})
        self.chain_prior = prior
        self.banned_pairs = set()
        self._collect_table = table
        self._clusters = None
        self.chain_tour_stats = {
            "scored": 0,
            "no_tour": 0,
            "not_closing": 0,
            "cache_hits": 0,
            "seconds": 0.0,
            "reranked": 0,
            "levels": 0,
        }
        self._chain_tour_cache = {}

    @property
    def clusters(self):
        return None


def _partial(deployed, camp_epoch, mass, hop_propellant):
    legs = [dataclasses.make_dataclass("L", ["departure_epoch"])(T0)]
    return _Partial(legs, deployed[-1][0], camp_epoch, mass, list(deployed), hop_propellant)


def test_chain_score_is_the_tour_objective_at_the_beam_exchange_rate_minus_prices() -> None:
    ids = [1, 2, 3]
    costs = {(a, b): 1.0 for a in ids for b in ids if a != b}
    table = _FakeTable(ids, costs, n_t=20, return_dv=1.0)
    settings = SearchSettings(
        chain_tour_scoring=True, propellant_weight=0.15, collect_dp_propellant_weight=1.0
    )
    deployed = [(1, T0 + 4.5 * YEAR), (2, T0 + 5.0 * YEAR), (3, T0 + 5.5 * YEAR)]
    search = _StubSearch(table, settings)
    partial = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=200.0)
    partial.score = 123.0  # heuristic score (the fallback base)
    # the first scored level prices the tour with the nominal burn schedule
    tour = search._chain_tour(partial)
    assert tour is not None and search.chain_tour_stats["scored"] == 1
    # the tour goes through the exact forward mass pass; the chain is ranked by the plan score
    # its completion would get (weighted collected - propellant weight x total propellant)
    plan = search._plan_from_tour(partial, tour)
    assert plan is not None and plan.feasible
    score = search._chain_score(partial)
    assert search.chain_tour_stats["cache_hits"] == 1  # same chain, same burn: cached
    assert score == pytest.approx(search.plan_score(plan))
    assert score == pytest.approx(
        sum(plan.collected_mass.values()) - 0.15 * plan.propellant_proxy_kg
    )
    # the plan's propellant is everything spent from launch: the deploy phase plus the tour
    # re-priced by the exact pass (its mass schedule differs from the DP's by the burn credit and
    # the collected mass, a few kg here)
    deploy_phase = settings.initial_mass - 1500.0 - 3 * C.MINER_MASS_KG
    tour_exact = plan.propellant_proxy_kg - deploy_phase
    assert tour_exact > 0.0
    assert tour_exact == pytest.approx(tour.propellant_proxy_kg, rel=0.1)
    # components recorded for the report and the children's burn schedule
    assert partial.chain_collect_kg == pytest.approx(sum(tour.hop_propellant_kg))
    assert partial.chain_return_kg == pytest.approx(
        tour.propellant_proxy_kg - sum(tour.hop_propellant_kg)
    )
    assert partial.chain_collected_kg == pytest.approx(sum(plan.collected_mass.values()))
    assert partial.chain_burn == pytest.approx(
        sum(tour.hop_propellant_kg) / len(tour.hop_propellant_kg)
    )
    # the measured burn is the children's mass schedule: a child inherits it and prices anew
    child = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=200.0)
    child.chain_burn = partial.chain_burn
    child.score = 0.0
    search._chain_score(child)
    assert search.chain_tour_stats["scored"] == 2
    # prices subtract exactly once per claimed asteroid (same nominal burn: same tour)
    priced = _StubSearch(table, settings, prices={2: 40.0, 3: 10.0, 99: 1000.0})
    priced_partial = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=200.0)
    priced_partial.score = 123.0
    assert priced._chain_score(priced_partial) == pytest.approx(score - 50.0)
    # a chain that cannot close on the mass budget falls back below every closing chain
    light = _partial(deployed, T0 + 6.0 * YEAR, C.DRY_MASS_KG + 5.0, hop_propellant=200.0)
    light.score = 123.0
    assert search._chain_score(light) == pytest.approx(123.0 - RouteSearch.CHAIN_FALLBACK_KG)
    assert search.chain_tour_stats["not_closing"] == 1


def test_chain_prior_penalty_is_monotone_and_zero_on_the_reference_manifold() -> None:
    prior = ChainPrior(
        collect_hop_kg_p75=83.0,
        collect_hop_kg_median=66.0,
        deploy_hop_kg_p25=76.0,
        deploy_hop_kg_median=98.0,
        collect_share_p90=0.5,
    )
    # reference-like chain: median deploy hops, collect hops at the median -> no penalty
    assert (
        prior.penalty(
            deploy_hops_kg=8 * 98.0, deploy_hops=8, collect_hops_kg=8 * 66.0, collect_hops=8
        )
        == 0.0
    )
    # dear harvest: kg above p75 per hop, summed
    dear = prior.penalty(
        deploy_hops_kg=8 * 98.0, deploy_hops=8, collect_hops_kg=8 * 90.0, collect_hops=8
    )
    assert dear == pytest.approx(8 * (90.0 - 83.0))
    # cheap deploy + dear harvest: the deploy shortfall below p25 is charged too
    signature = prior.penalty(
        deploy_hops_kg=8 * 60.0, deploy_hops=8, collect_hops_kg=8 * 90.0, collect_hops=8
    )
    assert signature == pytest.approx(dear + 8 * (76.0 - 60.0))
    # cheap deploy with a cheap harvest is fine (the references' own cheapest ships)
    assert (
        prior.penalty(
            deploy_hops_kg=8 * 60.0, deploy_hops=8, collect_hops_kg=8 * 60.0, collect_hops=8
        )
        == 0.0
    )
    # monotone in both arguments
    grid = np.linspace(300.0, 1000.0, 15)
    in_collect = [
        prior.penalty(deploy_hops_kg=700.0, deploy_hops=8, collect_hops_kg=c, collect_hops=8)
        for c in grid
    ]
    assert all(b >= a for a, b in itertools.pairwise(in_collect))
    in_deploy = [
        prior.penalty(deploy_hops_kg=d, deploy_hops=8, collect_hops_kg=720.0, collect_hops=8)
        for d in grid
    ]
    assert all(b <= a for a, b in itertools.pairwise(in_deploy))
    assert (
        prior.penalty(deploy_hops_kg=0.0, deploy_hops=0, collect_hops_kg=0.0, collect_hops=0) == 0.0
    )


def test_chain_prior_penalty_enters_the_chain_score_at_its_weight() -> None:
    ids = [1, 2, 3]
    costs = {(a, b): 2.5 for a in ids for b in ids if a != b}  # dear hops
    table = _FakeTable(ids, costs, n_t=20, return_dv=1.0)
    deployed = [(1, T0 + 4.5 * YEAR), (2, T0 + 5.0 * YEAR), (3, T0 + 5.5 * YEAR)]
    prior = ChainPrior(20.0, 15.0, 200.0, 250.0, 0.5)  # everything is dear against this prior
    plain = _StubSearch(table, SearchSettings(chain_tour_scoring=True))
    weighted = _StubSearch(
        table, SearchSettings(chain_tour_scoring=True, chain_prior_weight=0.5), prior=prior
    )
    a = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=100.0)
    b = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=100.0)
    a.score = b.score = 0.0
    tour = plain._chain_tour(a)  # the nominal-burn tour both scores price (cached)
    assert tour is not None
    base = plain._chain_score(a)
    priced = weighted._chain_score(b)
    expected = prior.penalty(
        deploy_hops_kg=100.0,
        deploy_hops=2,
        collect_hops_kg=sum(tour.hop_propellant_kg),
        collect_hops=len(tour.hop_propellant_kg),
    )
    assert expected > 0.0
    assert priced == pytest.approx(base - 0.5 * expected)


# -- LP duals as asteroid prices -----------------------------------------------------------------


def test_lp_asteroid_prices_are_nonnegative_duals_of_the_binding_rows() -> None:
    columns = _random_columns(3, 9, 14)
    weights = {a: 1.0 + 0.1 * (a % 3) for a in range(1, 15)}
    master = solve_fleet_master(columns, weights=weights)
    priced = lp_asteroid_prices(columns, weights=weights, target_size=master.ships + 1)
    assert priced is not None
    assert priced.size >= 1 and priced.size <= master.ships + 1
    used = {a for c in columns for a in (*c.deploys, *c.collects)}
    assert set(priced.prices) <= used
    assert all(p > 0.0 for p in priced.prices.values())
    # the LP value at the priced size bounds every fleet of that size (weak duality) and the
    # summary is JSON-serialisable and sorted best first
    assert priced.lp_value >= 0.0
    summary = priced.summary()
    json.dumps(summary)
    tops = [item["kg"] for item in summary["top"]]
    assert tops == sorted(tops, reverse=True)
    # pinning the size reproduces the same prices (deterministic HiGHS solve)
    again = lp_asteroid_prices(columns, weights=weights, size=priced.size)
    assert again is not None and again.prices == priced.prices
    # an infeasible size gives no prices
    assert lp_asteroid_prices(columns, weights=weights, size=40) is None
    # a column whose asteroids are all priced has a non-positive reduced cost at optimality
    # unless it sits at its upper bound: at least one selected column carries the prices
    selected_priced = sum(
        1 for c in master.selected for a in (*c.deploys, *c.collects) if a in priced.prices
    )
    assert selected_priced > 0 or not priced.prices


def test_bound_share_prices_dominate_the_row_duals_and_conserve_the_dual_value() -> None:
    columns = _random_columns(1, 12, 16)
    rows_only = lp_asteroid_prices(columns, target_size=6)
    shared = lp_asteroid_prices(columns, target_size=6, bound_share=True)
    assert rows_only is not None and shared is not None and shared.size == rows_only.size
    # sharing only adds: every row-priced asteroid keeps at least its row dual, and the
    # selected columns' asteroids (where the rent sat on the bound) now carry a price too
    for asteroid, price in rows_only.prices.items():
        assert shared.prices.get(asteroid, 0.0) >= price - 1e-9
    assert set(rows_only.prices) <= set(shared.prices)
    # the dual value is conserved: sum of shared prices = sum of row duals + sum of bound duals,
    # which the LP's own dual objective fixes (weak duality with the same mu and nu)
    assert shared.mu == pytest.approx(rows_only.mu) and shared.nu == pytest.approx(rows_only.nu)
    assert sum(shared.prices.values()) >= sum(rows_only.prices.values()) - 1e-9
    # deterministic
    again = lp_asteroid_prices(columns, target_size=6, bound_share=True)
    assert again is not None and again.prices == shared.prices


def test_collect_tour_ignores_a_non_finite_burn_schedule() -> None:
    ids = [1, 2, 3]
    costs = {(a, b): 1.0 for a in ids for b in ids if a != b}
    table = _FakeTable(ids, costs, n_t=20, return_dv=1.0)
    deployed = [(1, T0 + 4.5 * YEAR), (2, T0 + 5.0 * YEAR), (3, T0 + 5.5 * YEAR)]
    two_pass = plan_collect_tour(table, deployed, 3, T0 + 6.0 * YEAR, 1500.0)
    nan_burn = plan_collect_tour(
        table, deployed, 3, T0 + 6.0 * YEAR, 1500.0, burn_per_hop=float("nan")
    )
    assert two_pass is not None and nan_burn is not None
    assert nan_burn.order == two_pass.order and nan_burn.hops == two_pass.hops


def test_price_clusters_hands_the_dispatch_time_prices_to_each_family(monkeypatch) -> None:
    from spacepdhcg.gtoc12 import bundles

    seen: list[tuple[int, dict]] = []

    def fake_price_cluster(catalogue, members, *, label, asteroid_prices=None, **kwargs):
        seen.append((label, dict(asteroid_prices or {})))
        return bundles.ClusterBundle(label, tuple(int(a) for a in members), [])

    monkeypatch.setattr(bundles, "price_cluster", fake_price_cluster)
    prices: dict[int, float] = {}

    def on_result(bundle):
        # the master runs after every family: raise the price of asteroid 7 each time
        prices[7] = prices.get(7, 0.0) + 10.0

    clusters = [(1, np.asarray([1, 2, 7])), (2, np.asarray([3, 7])), (3, np.asarray([4, 5]))]
    bundles.price_clusters(
        None, clusters, workers=1, on_result=on_result, prices=lambda: dict(prices)
    )
    assert [label for label, _ in seen] == [1, 2, 3]
    # family 1 was dispatched before any master (no prices), family 2 after one, 3 after two
    assert seen[0][1] == {}
    assert seen[1][1] == {7: 10.0}
    assert seen[2][1] == {7: 20.0}


@requires_data
def test_archive_pricing_columns_price_the_archived_fleet_without_recertification() -> None:
    from pathlib import Path

    from spacepdhcg.gtoc12.archive import pricing_columns

    source = Path("results/gtoc12/runs/probe_v6_family")
    if not source.is_dir():
        pytest.skip("archived probe run not present")
    columns = pricing_columns([source])
    assert columns, "the archive has certified routes"
    for column in columns:
        assert column.route is None and column.certified  # pricing only, never assembled
        assert column.label.startswith("archive:")
        assert column.collected_kg > 0.0
        # a ship collects its own miners or another archived ship's (foreign) ones
        assert set(column.collected_mass) <= set(column.deploys) | set(column.foreign)
    assert len({c.identifier for c in columns}) == len(columns)
    # the same directory prices the same way twice (deterministic discovery and LP)
    again = pricing_columns([source])
    assert [c.label for c in again] == [c.label for c in columns]
    # one family's ships share no asteroid: the LP takes them all and no packing row binds, so
    # nothing is priced (a price is a *conflict* price, not a value)
    priced = lp_asteroid_prices(columns, target_size=len(columns) + 5)
    assert priced is not None and priced.prices == {}
    assert priced.size <= sum(c.ships for c in columns)
    # a re-timed twin of every ship (the return-sweep / joint-itinerary archives do exactly this)
    # conflicts with its original on every asteroid: now the rows bind and the ships' asteroids
    # carry prices, and only theirs
    twins = [
        dataclasses.replace(c, identifier=c.identifier + 500, label=c.label + "/twin")
        for c in columns
    ]
    priced = lp_asteroid_prices(columns + twins, target_size=len(columns))
    assert priced is not None and priced.prices
    used = {a for c in columns for a in c.deploys}
    assert set(priced.prices) <= used
    assert all(p > 0.0 for p in priced.prices.values())


# -- prior extraction (data-backed) -----------------------------------------------------------


@requires_data
def test_chain_prior_extraction_is_reproducible_and_targets_are_quantiles() -> None:
    paths = reference_solution_files(data_directory())
    if not paths:
        pytest.skip("archived reference solutions not present")
    catalogue = load_catalogue()
    first = extract_chain_prior(catalogue, paths[:1])
    second = extract_chain_prior(catalogue, paths[:1])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["ships_decoded"] > 0
    assert first["sources"][0]["file"] in REFERENCE_FILES
    targets, dist = first["targets"], first["distributions"]
    assert targets["collect_hop_kg_p75"] == dist["collect_hop_kg"]["p75"]
    assert targets["deploy_hop_kg_p25"] == dist["deploy_hop_kg"]["p25"]
    assert targets["collect_share_p90"] == dist["collect_share"]["p90"]
    # the reference structure the ninth iteration targets: cheap collect hops, dearer deploys
    assert 50.0 < targets["collect_hop_kg_median"] < 80.0
    assert targets["deploy_hop_kg_median"] > targets["collect_hop_kg_median"]
    prior = ChainPrior.from_document(first)
    assert prior.collect_hop_kg_p75 == targets["collect_hop_kg_p75"]
    # the committed document loads and agrees with a fresh full extraction
    committed = load_chain_prior("benchmarks/gtoc12/chain_prior_v1.json")
    full = ChainPrior.from_document(extract_chain_prior(catalogue, paths))
    assert committed.collect_hop_kg_p75 == pytest.approx(full.collect_hop_kg_p75)
    assert committed.deploy_hop_kg_p25 == pytest.approx(full.deploy_hop_kg_p25)


# -- the beam with chain scoring and prices (data-backed) -------------------------------------


@requires_data
def test_beam_chain_scoring_is_deterministic_and_prices_steer_it_off_claimed_asteroids() -> None:
    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids
    settings = SearchSettings(
        beam_width=6,
        max_deploys=4,
        launch_epochs=tuple(float(x) for x in T0 + np.arange(0.0, 731.0, 180.0)),
        earth_leg_tofs=(600.0, 750.0, 900.0),
        collect_dp_step_days=60.0,
        first_level_limit=200,
        harvest_substitution=False,
        chain_tour_scoring=True,
        chain_tour_candidates=12,
        chain_tour_min_deploys=3,
    )
    first = RouteSearch(catalogue, ids, settings)
    result = first.run()
    stats = first.chain_tour_stats
    assert result.best is not None
    assert stats["levels"] >= 1 and stats["scored"] > 0
    second = RouteSearch(catalogue, ids, settings).run()
    assert [c.summary() for c in result.candidates] == [c.summary() for c in second.candidates]
    # every emitted plan is still an exact, feasible schedule
    for plan in result.candidates:
        assert plan.feasible
        for asteroid in plan.deploy_epochs:
            stay = plan.collect_epochs[asteroid] - plan.deploy_epochs[asteroid]
            assert stay >= C.MIN_MINING_STAY_YEARS * YEAR - 1e-6
    # dual feedback: price the best plan's asteroids prohibitively -> the beam's best avoids
    # them, and the count of emitted plans using them is non-increasing in the price
    claimed = set(result.best.asteroids)
    counts = []
    for price in (0.0, 50.0, 5000.0):
        priced = RouteSearch(
            catalogue, ids, settings, asteroid_prices={a: price for a in claimed}
        ).run()
        counts.append(sum(1 for p in priced.candidates if claimed & set(p.asteroids)))
        if price >= 5000.0 and priced.best is not None:
            assert not (claimed & set(priced.best.asteroids))
    assert counts[0] >= counts[1] >= counts[2]
