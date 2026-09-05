"""Tenth iteration: harvest-phase prior - reference |Δλ| extraction, penalty arithmetic, and the
collect DP / chain score preferring phase-aligned harvest hops at comparable propellant."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from test_gtoc12_chain import _partial, _StubSearch
from test_gtoc12_collectdp import _FakeTable

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.chainprior import REFERENCE_FILES, reference_solution_files
from spacepdhcg.gtoc12.collectdp import CollectPairTable, plan_collect_tour
from spacepdhcg.gtoc12.data import data_available, data_directory, load_catalogue
from spacepdhcg.gtoc12.harvestphase import (
    HarvestPhasePrior,
    extract_harvest_phase,
    load_harvest_phase,
    phase_deg_at,
)
from spacepdhcg.gtoc12.search import SearchSettings

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
YEAR = C.YEAR_DAYS
T0 = C.MISSION_START_MJD
COMMITTED = "benchmarks/gtoc12/harvest_phase_v1.json"


def _prior(
    p75: float = 4.8, kg_per_deg: float = 3.0, days_per_deg: float = 0.0
) -> HarvestPhasePrior:
    return HarvestPhasePrior(p75, 2.7, kg_per_deg, days_per_deg, 0.25, "synthetic")


def _phase_table(ids, costs, phases_deg: dict[tuple[int, int], float], prior, **kwargs):
    """A fake pair table whose pairs sit at fixed ``|Δλ|`` (deg) at every departure epoch."""

    table = _FakeTable(ids, costs, **kwargs)
    table.settings = dataclasses.replace(table.settings, harvest_phase=prior, phase_weight=1.0)
    table.pair_geometry = lambda s, t, epochs: (
        0.01,
        np.full(len(epochs), math.radians(phases_deg.get((s, t), 0.0))),
    )
    return table


_FakeTable.phase_deg = CollectPairTable.phase_deg
_FakeTable.phase_penalty = CollectPairTable.phase_penalty


# -- penalty arithmetic -------------------------------------------------------------------------


def test_penalty_is_zero_on_the_reference_manifold_and_linear_above_it() -> None:
    prior = _prior(p75=4.8, kg_per_deg=3.0, days_per_deg=0.4)
    assert prior.kg_per_deg_total == pytest.approx(3.0 + 0.4 * 0.25)
    phases = np.asarray([0.0, 2.7, 4.8, 4.8 + 1e-12, 10.0, -10.0, 60.0])
    penalty = prior.penalty_kg(phases)
    assert penalty.shape == phases.shape
    assert np.all(penalty[:4] <= 1e-9)  # at or below p75: nothing charged
    assert penalty[4] == pytest.approx(prior.kg_per_deg_total * (10.0 - 4.8))
    assert penalty[5] == penalty[4]  # sign of Δλ is irrelevant
    assert np.all(np.diff(prior.penalty_kg(np.linspace(0.0, 180.0, 50))) >= -1e-12)  # monotone
    assert float(prior.penalty_kg(7.3)) == pytest.approx(prior.kg_per_deg_total * 2.5)
    # the round trip through a document keeps every target
    document = {
        "schema_version": "1.0.0",
        "targets": {
            "phase_deg_p75": 4.8,
            "phase_deg_median": 2.7,
            "kg_per_deg": 3.0,
            "days_per_deg": 0.4,
            "exchange_kg_per_day": 0.25,
        },
    }
    again = HarvestPhasePrior.from_document(document, source="doc")
    assert again.penalty_kg(12.0) == pytest.approx(prior.penalty_kg(12.0))
    assert again.summary()["source"] == "doc"


# -- the DP prefers the aligned hop at comparable propellant ---------------------------------------


def test_collect_dp_ranking_flips_to_the_phase_aligned_hop_at_comparable_propellant() -> None:
    """Camp 13 collected first, then 11 and 12 in either order.  13 -> 11 is the marginally
    cheaper hop (0.30 vs 0.32 km/s: ~1 kg) but departs 12 deg out of phase; 13 -> 12 departs
    aligned.  Without the prior the DP takes the cheaper misaligned hop; with the reference
    prior (p75 4.8 deg, 3 kg/deg) the aligned order wins and the accounting stays honest."""

    ids = [11, 12, 13]
    dear = 3.0
    costs = {
        (13, 11): 0.30,
        (11, 12): 0.30,
        (13, 12): 0.32,
        (12, 11): 0.30,
        (11, 13): dear,
        (12, 13): dear,
    }
    phases = {(13, 11): 12.0, (11, 12): 1.0, (13, 12): 1.0, (12, 11): 1.0}
    deployed = [(11, T0 + 5.0 * YEAR), (12, T0 + 5.5 * YEAR), (13, T0 + 6.0 * YEAR)]
    plain = _phase_table(ids, costs, phases, None)
    without = plan_collect_tour(plain, deployed, 13, T0 + 6.0 * YEAR, 1500.0)
    assert without is not None and without.order == (13, 11, 12)
    assert without.phase_penalty_kg == 0.0 and without.hop_phase_deg == []
    prior = _prior(p75=4.8, kg_per_deg=3.0)
    priced = _phase_table(ids, costs, phases, prior)
    with_prior = plan_collect_tour(priced, deployed, 13, T0 + 6.0 * YEAR, 1500.0)
    assert with_prior is not None and with_prior.order == (13, 12, 11)  # the ranking flipped
    assert with_prior.hop_phase_deg == pytest.approx([1.0, 1.0])
    assert with_prior.phase_penalty_kg == 0.0  # the aligned tour pays nothing
    # objective = collected - w x (propellant + penalty); the propellant excludes the penalty
    assert with_prior.objective_kg == pytest.approx(
        with_prior.collected_proxy_kg - with_prior.propellant_proxy_kg - with_prior.phase_penalty_kg
    )
    assert with_prior.propellant_proxy_kg > without.propellant_proxy_kg  # it paid ~1 kg more
    assert with_prior.propellant_proxy_kg - without.propellant_proxy_kg < prior.penalty_kg(12.0)
    # a prior too weak to cover the propellant difference leaves the cheaper hop in place, and
    # the misaligned tour is then charged exactly its penalty
    weak = _phase_table(ids, costs, phases, _prior(p75=4.8, kg_per_deg=0.01))
    unchanged = plan_collect_tour(weak, deployed, 13, T0 + 6.0 * YEAR, 1500.0)
    assert unchanged is not None and unchanged.order == (13, 11, 12)
    assert unchanged.hop_phase_deg == pytest.approx([12.0, 1.0])
    assert unchanged.phase_penalty_kg == pytest.approx(0.01 * (12.0 - 4.8))
    assert unchanged.objective_kg == pytest.approx(
        unchanged.collected_proxy_kg - unchanged.propellant_proxy_kg - unchanged.phase_penalty_kg
    )
    # deterministic
    again = plan_collect_tour(priced, deployed, 13, T0 + 6.0 * YEAR, 1500.0)
    assert again is not None and again.order == with_prior.order and again.hops == with_prior.hops


def test_chain_score_subtracts_the_scored_tours_phase_penalty() -> None:
    ids = [1, 2, 3]
    costs = {(a, b): 1.0 for a in ids for b in ids if a != b}
    phases = {(a, b): 14.8 for a in ids for b in ids if a != b}  # every hop 10 deg off
    deployed = [(1, T0 + 4.5 * YEAR), (2, T0 + 5.0 * YEAR), (3, T0 + 5.5 * YEAR)]
    settings = SearchSettings(chain_tour_scoring=True, propellant_weight=0.15)
    plain = _StubSearch(_phase_table(ids, costs, phases, None, n_t=20, return_dv=1.0), settings)
    prior = _prior(p75=4.8, kg_per_deg=2.0)
    priced = _StubSearch(_phase_table(ids, costs, phases, prior, n_t=20, return_dv=1.0), settings)
    a = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=100.0)
    b = _partial(deployed, T0 + 6.0 * YEAR, 1500.0, hop_propellant=100.0)
    a.score = b.score = 0.0
    base = plain._chain_score(a)
    scored = priced._chain_score(b)
    # two collect hops, each 10 deg beyond p75 at 2 kg/deg: 40 kg off the chain score
    assert b.chain_phase_kg == pytest.approx(40.0)
    assert scored == pytest.approx(base - 40.0)
    assert priced.chain_tour_stats["phase_kg"] == pytest.approx(40.0)
    assert math.isnan(a.chain_phase_kg) or a.chain_phase_kg == 0.0


# -- extraction from the references ---------------------------------------------------------------


@requires_data
def test_harvest_phase_extraction_is_reproducible_and_matches_the_committed_document() -> None:
    paths = reference_solution_files(data_directory())
    if not paths:
        pytest.skip("archived reference solutions not present")
    catalogue = load_catalogue()
    first = extract_harvest_phase(catalogue, paths[:1])
    second = extract_harvest_phase(catalogue, paths[:1])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["hops_decoded"] > 0 and first["sources"][0]["file"] in REFERENCE_FILES
    targets = first["targets"]
    assert 0.0 < targets["phase_deg_median"] <= targets["phase_deg_p75"] <= targets["phase_deg_p90"]
    assert targets["kg_per_deg"] >= 0.0 and targets["days_per_deg"] >= 0.0
    assert targets["exchange_kg_per_day"] == pytest.approx(
        targets["asteroids_per_ship_median"] * C.MINING_RATE_KG_PER_YEAR / C.YEAR_DAYS
    )
    # the histogram covers every hop once and its fractions sum to one
    bins = first["distributions"]["histogram"]
    assert sum(b["count"] for b in bins) == first["hops_decoded"]
    assert sum(b["fraction"] for b in bins) == pytest.approx(1.0)
    # every recorded hop's phase is the pair's |Δλ| at its departure
    hop = first["hops"][0]
    leg_phase = phase_deg_at(catalogue, hop["from"], hop["to"], hop["departure_epoch"])
    assert hop["phase_deg"] == pytest.approx(leg_phase)
    # the committed document was produced by this extraction on the pinned files
    committed = load_harvest_phase(COMMITTED)
    document = json.loads(open(COMMITTED, encoding="utf-8").read())
    full = extract_harvest_phase(catalogue, paths)
    assert document["hops_decoded"] == full["hops_decoded"]
    assert [s["sha256"] for s in document["sources"]] == [s["sha256"] for s in full["sources"]]
    fresh = HarvestPhasePrior.from_document(full)
    for name in ("p75_deg", "median_deg", "kg_per_deg", "days_per_deg", "exchange_kg_per_day"):
        assert getattr(committed, name) == pytest.approx(getattr(fresh, name))
    # the references' harvest hops are phase-aligned: p75 under 10 deg, and misalignment costs
    assert committed.p75_deg < 10.0 and committed.kg_per_deg > 0.0
