"""Independent verifier: synthetic rule checks and exact reproduction of official scores."""

from __future__ import annotations

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import (
    AsteroidCatalogue,
    data_available,
    load_bonus_table,
    load_catalogue,
    official_example_solution,
    verified_path,
)
from spacepdhcg.gtoc12.ephemeris import earth_state
from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
from spacepdhcg.gtoc12.solution import Event, ShipTrajectory, Solution, StateLine
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier, LagrangeThrust

EARTH_PERIOD_DAYS = 2.0 * np.pi * np.sqrt(C.EARTH.semi_major_axis_km**3 / C.MU_SUN_KM3_S2) / C.DAY_S


def _empty_catalogue() -> AsteroidCatalogue:
    n = C.ASTEROID_COUNT
    return AsteroidCatalogue(
        ids=np.arange(1, n + 1),
        epoch_mjd=np.full(n, 64328.0),
        semi_major_axis_km=np.full(n, 2.8 * C.AU_KM),
        eccentricity=np.zeros(n),
        inclination_rad=np.zeros(n),
        ascending_node_rad=np.zeros(n),
        argument_of_perihelion_rad=np.zeros(n),
        mean_anomaly_rad=np.zeros(n),
        source_sha256="",
    )


def _coasting_earth_ship(vinf: float = 0.0, mass_drop: float = 0.0) -> Solution:
    """Launch with Earth's velocity and meet Earth again one sidereal period later (score 0)."""

    t0 = 64400.0
    t1 = t0 + EARTH_PERIOD_DAYS
    r0, v0 = earth_state(t0)
    r1, v1 = earth_state(t1)
    direction = v0 / np.linalg.norm(v0)
    launch = Event(
        0, StateLine(t0, r0, v0, 3000.0), StateLine(t0, r0, v0 + vinf * direction, 3000.0)
    )
    flyby = Event(
        -3, StateLine(t1, r1, v1, 3000.0 - mass_drop), StateLine(t1, r1, v1, 3000.0 - mass_drop)
    )
    return Solution([ShipTrajectory(1, [launch, flyby])])


def test_coasting_ship_passes_with_zero_score() -> None:
    report = Gtoc12Verifier(_empty_catalogue()).verify(_coasting_earth_ship())
    assert report.ok, report.violations
    assert report.ship_count == 1 and report.total_mass_kg == 0.0
    assert report.legs[0].position_error_km < 1.0
    assert report.legs[0].velocity_error_km_s < 1e-8


def test_launch_vinf_and_mass_rules_are_enforced() -> None:
    verifier = Gtoc12Verifier(_empty_catalogue())
    codes = {item.code for item in verifier.verify(_coasting_earth_ship(vinf=6.5)).violations}
    assert "Error505" in codes
    codes = {item.code for item in verifier.verify(_coasting_earth_ship(mass_drop=1.0)).violations}
    assert "Error203" in codes  # propagated mass disagrees with the file


def test_lagrange_interpolant_reproduces_cubic_exactly() -> None:
    epochs = np.arange(0.0, 10.0)
    coefficients = np.array(
        [[0.1, -0.2, 0.3], [0.01, 0.02, -0.03], [1e-3, 2e-3, 3e-3], [1e-4, -1e-4, 2e-4]]
    )
    thrust = sum(coefficients[k][None, :] * (epochs[:, None] ** k) for k in range(4))
    interpolant = LagrangeThrust(epochs, thrust)
    for t in (0.5, 2.3, 4.999, 8.75):
        expected = sum(coefficients[k] * t**k for k in range(4))
        assert np.allclose(interpolant(t), expected, atol=1e-12)
    # duplicate epochs (zero-length partial day, as in the JPL file) are collapsed
    duplicated = LagrangeThrust(np.array([0.0, 1.0, 2.0, 2.0, 3.0]), np.ones((5, 3)))
    assert duplicated.epochs.tolist() == [0.0, 1.0, 2.0, 3.0]


requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")


@requires_data
def test_official_example_is_verified_with_zero_mass() -> None:
    report = Gtoc12Verifier(load_catalogue()).verify_file(official_example_solution())
    assert report.ok, report.violations
    assert (
        report.ship_count == 1 and report.mined_asteroid_count == 0 and report.total_mass_kg == 0.0
    )
    assert report.legs[0].to_event == 40239 and report.legs[0].passed


@requires_data
@pytest.mark.skipif(
    not official_verifier_available(), reason="official verifier binary unavailable"
)
def test_official_wrapper_on_example() -> None:
    result = run_official_verifier(official_example_solution())
    assert (
        result.ok
        and result.ships == 1
        and result.mined_asteroids == 0
        and result.total_mass_kg == 0.0
    )


@requires_data
@pytest.mark.parametrize(
    ("name", "ships", "asteroids", "total"),
    [
        ("39_mass_optimal.txt", 39, 356, 28975.140269),
        ("37_mass_optimal_self_cleaning.txt", 37, 338, 27045.268330),
        ("GTOC12_JPL_merged_solution_36sc.txt", 36, 320, 26062.646065),
    ],
)
def test_reference_solutions_reproduce_official_scores_exactly(
    name: str, ships: int, asteroids: int, total: float
) -> None:
    catalogue = load_catalogue()
    bonus = load_bonus_table()
    path = verified_path(name)
    report = Gtoc12Verifier(catalogue, bonus=bonus, rtol=1e-11).verify_file(path)
    assert report.ok, report.violations[:5]
    assert report.ship_count == ships
    assert report.mined_asteroid_count == asteroids
    assert report.total_mass_kg == pytest.approx(total, abs=1e-5)
    assert report.ship_count <= report.ship_limit
    if official_verifier_available():
        official = run_official_verifier(path)
        assert official.ok and official.ships == ships and official.mined_asteroids == asteroids
        assert set(official.score_data) == set(report.scored_masses)
        assert (
            max(abs(official.score_data[k] - report.scored_masses[k]) for k in report.scored_masses)
            < 1e-9
        )
