"""Ephemeris against published example states, Kepler propagation, and solution-file round trips."""

from __future__ import annotations

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.ephemeris import (
    earth_state,
    elements_to_state,
    planet_state,
    propagate_kepler,
    solve_kepler,
)
from spacepdhcg.gtoc12.solution import (
    Event,
    ShipTrajectory,
    Solution,
    SolutionFormatError,
    StateLine,
    format_solution,
    make_burn_arc,
    parse_solution,
)

# Lines 1-2 of the organisers' example Result.txt (GTOC12_Verification_Program.zip): the Earth
# state at 64328 MJD written by their own ephemeris.
EXAMPLE_EARTH_POSITION = np.array(
    [-25267390.158699154854, 144918560.832322776318, -11774.074663434592]
)
EXAMPLE_EARTH_VELOCITY = np.array([-29.830354405016, -5.218793306247, -0.000385365490])


def test_earth_state_matches_official_example_launch_line() -> None:
    r, v = earth_state(64328.0)
    assert np.linalg.norm(r - EXAMPLE_EARTH_POSITION) < 1e-3  # km
    assert np.linalg.norm(v - EXAMPLE_EARTH_VELOCITY) < 1e-8  # km/s


def test_kepler_solver_and_element_roundtrip() -> None:
    mean = np.linspace(0.0, 2.0 * np.pi, 17)
    ecc = np.full_like(mean, 0.3)
    eccentric = solve_kepler(mean, ecc)
    assert np.allclose(eccentric - ecc * np.sin(eccentric), np.mod(mean, 2 * np.pi), atol=1e-13)
    r, v = elements_to_state(
        np.array([C.AU_KM]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    assert np.allclose(r[0], [C.AU_KM, 0.0, 0.0])
    assert v[0, 1] == pytest.approx(np.sqrt(C.MU_SUN_KM3_S2 / C.AU_KM))


def test_kepler_propagation_matches_element_propagation() -> None:
    r0, v0 = planet_state(C.MARS, 64328.0)
    for days in (10.0, 100.0, 1000.0, -300.0):
        r1, v1 = propagate_kepler(r0, v0, days * C.DAY_S)
        r2, v2 = planet_state(C.MARS, 64328.0 + days)
        assert np.linalg.norm(r1 - r2) < 1e-3  # km
        assert np.linalg.norm(v1 - v2) < 1e-9  # km/s


def test_kepler_propagation_hyperbolic_is_stable() -> None:
    r0 = np.array([C.AU_KM, 0.0, 0.0])
    v0 = np.array([0.0, 60.0, 0.0])  # well above escape speed
    r1, v1 = propagate_kepler(r0, v0, 200.0 * C.DAY_S)
    back, vback = propagate_kepler(r1, v1, -200.0 * C.DAY_S)
    assert np.linalg.norm(back - r0) < 1e-3
    assert np.linalg.norm(vback - v0) < 1e-9


def _synthetic_solution() -> Solution:
    r_e, v_e = earth_state(64400.0)
    launch = Event(
        0,
        StateLine(64400.0, r_e, v_e, 3000.0),
        StateLine(64400.0, r_e, v_e + np.array([1.0, 2.0, 0.5]), 3000.0),
    )
    arc = make_burn_arc(
        np.array([64400.0, 64401.0, 64402.0, 64402.5]), np.tile([0.1, -0.2, 0.3], (4, 1))
    )
    rdv = Event(
        123,
        StateLine(64500.0, np.array([1e8, 2e8, 3e6]), np.array([-20.0, 10.0, 0.1]), 2900.0),
        StateLine(64500.0, np.array([1e8, 2e8, 3e6]), np.array([-20.0, 10.0, 0.1]), 2860.0),
    )
    return Solution([ShipTrajectory(1, [launch, arc, rdv])])


def test_solution_round_trip_is_lossless_to_tolerance() -> None:
    solution = _synthetic_solution()
    text = format_solution(solution)
    assert not text.endswith("\n")  # the official verifier rejects a trailing empty line
    parsed = parse_solution(text)
    assert parsed.ship_count == 1
    ship = parsed.ships[0]
    assert [type(item).__name__ for item in ship.items] == ["Event", "BurnArc", "Event"]
    assert ship.events[1].event_id == 123
    assert ship.events[1].before.mass == pytest.approx(2900.0)
    assert np.allclose(
        ship.events[0].after.velocity, solution.ships[0].events[0].after.velocity, atol=1e-12
    )
    epochs, thrust = ship.burns[0].interior_arrays()
    assert epochs.tolist() == [64400.0, 64401.0, 64402.0, 64402.5]
    assert np.allclose(thrust, np.tile([0.1, -0.2, 0.3], (4, 1)), atol=1e-13)


def test_parser_tolerates_commas_and_reports_format_errors() -> None:
    text = (
        "1 0 64400.0 1 2 3 4 5 6 3000\n1 0 64400.0 1 2 3 4 5 7 3000\n"
        "1 -1 64400.0 0.0, 0.0, 0.0\n1 -1 64400.0 0.1 0 0\n1 -1 64401.0 0.1 0 0\n1 -1 64401.0 0 0 0"
    )
    parsed = parse_solution(text)
    assert len(parsed.ships[0].burns) == 1
    with pytest.raises(SolutionFormatError) as error:
        parse_solution(
            "1 0 64400.0 1 2 3 4 5 6 3000\n1 0 64400.0 1 2 3 4 5 7 3000\n1 -1 64400.0 0.1 0 0"
        )
    assert error.value.code == "ErrorA04"
    with pytest.raises(SolutionFormatError) as error:
        parse_solution("1 0 64400.0 1 2 3\n")
    assert error.value.code == "ErrorA09"
    with pytest.raises(SolutionFormatError) as error:
        parse_solution("   \n")
    assert error.value.code == "ErrorA02"
