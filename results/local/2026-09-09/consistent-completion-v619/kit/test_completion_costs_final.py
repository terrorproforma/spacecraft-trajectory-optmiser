"""Completion must retain cost-model provenance at the actual forward mass."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.collectdp import CollectDPSettings, CollectPairTable, CollectTour
from spacepdhcg.gtoc12.hopcalib import InflationFit
from spacepdhcg.gtoc12.screening import (
    exhaust_velocity_km_s,
    return_inflation_model,
    thrust_authority_km_s,
)
from spacepdhcg.gtoc12.search import PlannedLeg, RouteSearch, SearchSettings, _Partial

T0 = C.MISSION_START_MJD


def fixture(mass=1100.0, *, fit=True, swept=False):
    search = RouteSearch(None, np.asarray([11, 12]), SearchSettings())
    model = InflationFit((0.95, 0.55, 0.1, 0.2, 0.1), 0.5) if fit else None
    table = CollectPairTable(None, CollectDPSettings(inflation_fit=model))
    departure = T0 + 3000.0
    table._geometry[(12, 11, departure, 1)] = (0.02, np.asarray([0.3]))
    search._collect_table = table
    if swept:
        shape = (len(table.epochs), len(table.return_tofs))
        # A precomputed valid-cell map: no Lambert calculation or native library is needed.
        table.return_sweeps[11] = object()
        table._return_overrides[11] = (np.full(shape, 0.83), np.ones(shape, dtype=bool))
    earth = PlannedLeg(0, 11, T0, T0 + 600.0, 1.0, 0.8, "earth_out")
    deploy = PlannedLeg(11, 12, T0 + 600.0, T0 + 750.0, 0.5, 1.2, "deploy_hop")
    partial = _Partial([earth, deploy], 12, T0 + 750.0, mass, [(11, T0 + 600.0), (12, T0 + 750.0)])
    tour = CollectTour(
        order=(12, 11),
        collect_epochs={12: departure, 11: departure + 180.0},
        hops=[(12, 11, departure, 180.0, 1.0)],
        reposition=False,
        objective_kg=0.0,
        collected_proxy_kg=0.0,
        propellant_proxy_kg=0.0,
        return_departure=departure + 180.0,
        return_tof=450.0,
        return_dv=4.0,
    )
    return search, partial, tour


def expected_masses(search, partial, tour, *, fitted, swept=False):
    cargo = {a: C.maximum_collected_mass(tour.collect_epochs[a] - t) for a, t in partial.deployed}
    first_mass = partial.mass + cargo[12]
    if fitted:
        ratio = 1.0 / float(thrust_authority_km_s(first_mass, 180.0, 1.0))
        first_inflation = max(
            0.5,
            0.95
            + 0.55 * ratio
            + 0.1 * 180.0 / C.YEAR_DAYS
            + 0.2 * 0.02 / 0.1
            + 0.1 * 0.3 / math.pi,
        )
    else:
        first_inflation = search.settings.hop_inflation
    hop_propellant = first_mass * (1.0 - math.exp(-first_inflation / exhaust_velocity_km_s()))
    return_mass = first_mass - hop_propellant + cargo[11]
    return_inflation = (
        0.83
        if swept
        else float(
            return_inflation_model(
                450.0, 4.0 / float(thrust_authority_km_s(return_mass, 450.0, 1.0))
            )
        )
    )
    return_propellant = return_mass * (
        1.0 - math.exp(-4.0 * return_inflation / exhaust_velocity_km_s())
    )
    return (
        cargo,
        first_inflation,
        return_inflation,
        hop_propellant + return_propellant,
        return_mass - return_propellant,
    )


def test_dp_completion_uses_calibrated_fit_at_actual_mass_and_records_costs():
    search, partial, tour = fixture()
    expected = expected_masses(search, partial, tour, fitted=True)
    original_prefix = tuple(partial.legs)
    plan = search._plan_from_tour(partial, tour)
    assert plan is not None
    hop = next(leg for leg in plan.legs if leg.role == "collect_hop")
    assert hop.inflation == pytest.approx(expected[1], rel=1e-14)
    assert hop.inflation != pytest.approx(search.settings.hop_inflation)
    assert plan.legs[-1].inflation == pytest.approx(expected[2], rel=1e-14)
    assert plan.collected_mass == expected[0]
    assert plan.final_mass_proxy_kg == pytest.approx(expected[4], rel=1e-14)
    prefix_spent = search.settings.initial_mass - partial.mass - 2 * C.MINER_MASS_KG
    assert plan.propellant_proxy_kg == pytest.approx(prefix_spent + expected[3], rel=1e-14)
    assert plan.legs[:2] == original_prefix and tuple(partial.legs) == original_prefix
    assert (
        plan.deploy_epochs == dict(partial.deployed) and plan.collect_epochs == tour.collect_epochs
    )


def test_heuristic_completion_keeps_beam_model_when_dp_table_exists():
    search, partial, tour = fixture()
    expected = expected_masses(search, partial, tour, fitted=False)
    forward = [
        PlannedLeg(
            12, 11, tour.collect_epochs[12], tour.collect_epochs[11], 1.0, 9.0, "collect_hop"
        ),
        PlannedLeg(
            11, 0, tour.return_departure, tour.return_departure + 450.0, 4.0, 9.0, "earth_return"
        ),
    ]
    plan = search._finish(partial, dict(partial.deployed), tour.collect_epochs, forward)
    assert plan is not None
    assert plan.legs[-2].inflation == search.settings.hop_inflation
    assert plan.legs[-1].inflation == pytest.approx(expected[2], rel=1e-14)
    assert plan.final_mass_proxy_kg == pytest.approx(expected[4], rel=1e-14)
    assert [leg.inflation for leg in forward] == [9.0, 9.0]  # Input provenance is immutable.


def test_dp_builder_prices_each_flight_once_at_its_actual_mass(monkeypatch):
    search, partial, tour = fixture()
    cargo, hop_factor, return_factor, _, final_mass = expected_masses(
        search, partial, tour, fitted=True
    )
    calls = []
    original_hop = search._dp_hop_inflation
    original_return = search.collect_table.return_inflation_at

    def hop(source, target, departure, dv, mass, tof):
        calls.append(("hop", mass))
        return original_hop(source, target, departure, dv, mass, tof)

    def earth_return(source, departure, tof, dv, mass):
        calls.append(("return", mass))
        return original_return(source, departure, tof, dv, mass)

    monkeypatch.setattr(search, "_dp_hop_inflation", hop)
    monkeypatch.setattr(search.collect_table, "return_inflation_at", earth_return)
    plan = search._plan_from_tour(partial, tour)
    first_mass = partial.mass + cargo[12]
    return_mass = first_mass * math.exp(-hop_factor / exhaust_velocity_km_s()) + cargo[11]
    assert [role for role, _ in calls] == ["hop", "return"]
    assert [mass for _, mass in calls] == pytest.approx([first_mass, return_mass], rel=1e-14)
    assert plan is not None and plan.final_mass_proxy_kg == pytest.approx(final_mass, rel=1e-14)
    assert plan.legs[-2].inflation == hop_factor and plan.legs[-1].inflation == return_factor


@pytest.mark.parametrize("mass", [1000.0, 1400.0])
def test_dp_completion_preserves_measured_return_cell_inflation(mass):
    search, partial, tour = fixture(mass, swept=True)
    expected = expected_masses(search, partial, tour, fitted=True, swept=True)
    plan = search._plan_from_tour(partial, tour)
    assert plan is not None and plan.legs[-1].inflation == 0.83
    assert plan.final_mass_proxy_kg == pytest.approx(expected[4], rel=1e-14)
    assert search.collect_table._return_overrides[11][1].all()


def test_certified_return_cell_does_not_bypass_existing_forward_authority_gate():
    search, partial, tour = fixture(swept=True)
    tour.return_dv = 100.0
    assert search._plan_from_tour(partial, tour) is None
    assert search.last_failure == "leg_authority"


def test_generic_return_model_is_evaluated_after_collection_burns():
    search, partial, tour = fixture(fit=False)
    expected = expected_masses(search, partial, tour, fitted=False)
    guessed_mass = partial.mass + sum(expected[0].values())
    guessed_inflation = float(
        return_inflation_model(450.0, 4.0 / float(thrust_authority_km_s(guessed_mass, 450.0, 1.0)))
    )
    plan = search._plan_from_tour(partial, tour)
    assert plan is not None
    assert abs(guessed_inflation - expected[2]) > 1e-4
    assert plan.legs[-1].inflation == pytest.approx(expected[2], rel=1e-14)
    assert plan.final_mass_proxy_kg == pytest.approx(expected[4], rel=1e-14)


def test_nonfinite_calibrated_inflation_cannot_produce_an_accepted_plan():
    search, partial, tour = fixture()
    search.collect_table.settings = dataclasses.replace(
        search.collect_table.settings, inflation_fit=InflationFit((float("nan"), 0, 0, 0, 0), 0.5)
    )
    assert search._plan_from_tour(partial, tour) is None
    assert search.last_failure == "invalid_inflation"


@pytest.mark.parametrize(
    "violation", ["uncollected", "stay_too_short", "mass_below_dry_plus_collected"]
)
def test_completion_keeps_mining_and_mass_rejections(violation):
    search, partial, tour = fixture()
    collect = dict(tour.collect_epochs)
    if violation == "uncollected":
        collect.pop(12)
    elif violation == "stay_too_short":
        collect[12] = dict(partial.deployed)[12]
    else:
        partial.mass = 400.0
    leg = PlannedLeg(11, 0, collect[11], collect[11] + 450.0, 0.0, 1.0, "earth_return")
    assert search._finish(partial, dict(partial.deployed), collect, [leg]) is None
    assert search.last_failure == violation
