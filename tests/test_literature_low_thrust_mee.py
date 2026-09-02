"""Modified-equinoctial-element multi-revolution low-thrust path (Gap 2 of the reproduction)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.literature import external_sources
from spacepdhcg.literature import low_thrust as lt
from spacepdhcg.literature import low_thrust_mee as mee

ROOT = Path(__file__).resolve().parents[1]


def _document(name: str) -> dict:
    return json.loads(
        (ROOT / "benchmarks" / "literature" / "profiles" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )


def _dionysus() -> lt.LowThrustProblem:
    return lt.problem_from_document(_document("tafazzol-taheri-earth-dionysus"))


def test_mee_cartesian_round_trip_is_exact_on_both_dionysus_boundaries() -> None:
    problem = _dionysus()
    for state in (problem.initial_state, problem.final_state):
        elements = mee.cartesian_to_mee(problem.mu, np.asarray(state))
        assert elements[0] > 0.0
        assert 0.0 <= elements[5] < 2.0 * math.pi
        back = mee.mee_to_cartesian(problem.mu, elements)
        assert np.allclose(back, state, atol=1.0e-10)


def test_complex_step_jacobians_match_central_differences() -> None:
    problem = _dionysus()
    dynamics = mee.MEEDynamics(problem.mu, problem.exhaust_velocity)
    x = np.array([1.3, 0.1, -0.3, 0.02, 0.05, 2.3, 0.8])
    u = np.array([0.3, 0.7, 0.2, 0.8]) * problem.max_thrust
    A, B = dynamics.jacobians(x, u)
    eps = 1.0e-6
    A_fd = np.column_stack(
        [(dynamics.f(x + eps * e, u) - dynamics.f(x - eps * e, u)) / (2 * eps) for e in np.eye(7)]
    )
    B_fd = np.column_stack(
        [(dynamics.f(x, u + eps * e) - dynamics.f(x, u - eps * e)) / (2 * eps) for e in np.eye(4)]
    )
    assert np.max(np.abs(A - A_fd)) < 1.0e-7
    assert np.max(np.abs(B - B_fd)) < 1.0e-7


def test_mee_vector_field_is_the_pushforward_of_the_cartesian_field() -> None:
    """d/dt (MEE -> Cartesian) along the MEE flow equals the Cartesian two-body field."""

    problem = _dionysus()
    dynamics = mee.MEEDynamics(problem.mu, problem.exhaust_velocity)
    cartesian = lt.LowThrustDynamics(problem)
    x = np.array([1.3, 0.1, -0.3, 0.02, 0.05, 2.3, 0.8])
    u = np.array([0.3, 0.7, 0.2, 0.8]) * problem.max_thrust
    frame = mee.rtn_frame(problem.mu, x)
    thrust_inertial = u[0] * frame[0] + u[1] * frame[1] + u[2] * frame[2]
    xc = np.concatenate([mee.mee_to_cartesian(problem.mu, x), [x[6]]])
    expected = cartesian.f(xc, np.concatenate([thrust_inertial, [u[3]]]))
    h = 1.0e-6
    dx = dynamics.f(x, u)
    forward = mee.mee_to_cartesian(problem.mu, x + h * dx)
    backward = mee.mee_to_cartesian(problem.mu, x - h * dx)
    observed = (forward - backward) / (2.0 * h)
    assert np.max(np.abs(observed - expected[:6])) < 1.0e-6
    assert math.isclose(dx[6], expected[6])


def test_initial_guess_encodes_the_requested_revolution_count() -> None:
    problem = _dionysus()
    for revolutions in (4, 5, 6):
        states, controls, final_longitude = mee.mee_initial_guess(problem, 120, revolutions, 60.0)
        x0 = mee.cartesian_to_mee(problem.mu, np.asarray(problem.initial_state))
        xf = mee.cartesian_to_mee(problem.mu, np.asarray(problem.final_state))
        assert np.allclose(states[0, :6], x0)
        assert np.allclose(states[-1, :5], xf[:5])
        assert math.isclose(states[-1, 5], final_longitude)
        turns = (final_longitude - xf[5]) / (2.0 * math.pi)
        assert math.isclose(turns, round(turns), abs_tol=1.0e-9)
        assert round(turns) == revolutions
        assert np.all(np.diff(states[:, 5]) > 0.0)  # longitude is monotone
        assert controls.shape == (120, 4)
    natural = mee.natural_revolutions(problem, problem.time_of_flight)
    assert 5.0 < natural < 7.0
    assert set(mee.default_revolution_candidates(problem)) >= {5, 6, 7}


def test_dionysus_five_revolution_rendezvous_reproduces_published_mass_within_envelope() -> None:
    """Coarse 200-node grid: 2715.7 kg against the published 2718.33 kg (five revolutions)."""

    problem = _dionysus()
    result = mee.solve_low_thrust_mee(problem, revolutions=5, nodes=200)
    assert result.converged, result.outcome.extras
    assert result.outcome.replay_defect_inf < 1.0e-5
    assert result.replay_terminal_position_error < 1.0e-4
    assert result.replay_terminal_velocity_error < 1.0e-4
    assert result.max_path_violation < 1.0e-8
    assert result.final_mass_si is not None
    # Discretisation envelope declared for the coarse grid: 200-node FOH quantises the
    # bang-bang switches to ~0.3 TU; 400 nodes recovers 2717.5 kg (see the reproduction report).
    assert abs(result.final_mass_si - 2718.33) < 3.5
    assert result.final_mass_si < 2718.33 + 0.5  # never "better" than the published optimum


@pytest.fixture(scope="module")
def tops_problems() -> dict[str, lt.LowThrustProblem]:
    from spacepdhcg.literature import tops

    try:
        external_sources.fetch("tops.twobody")
    except external_sources.ArtifactUnavailable as error:  # pragma: no cover - offline CI
        pytest.skip(f"TOPS artifact unavailable offline: {error}")
    return {
        p.key: tops.to_low_thrust_problem(p)
        for p in tops.ingest()
        if p.family == "two_body_cartesian"
    }


def test_tops_p3_multirev_fixed_time_converges_with_two_revolutions(
    tops_problems: dict[str, lt.LowThrustProblem],
) -> None:
    problem = tops_problems["two_body_cartesian:P3"]
    assert problem.fixed_time
    result = mee.solve_low_thrust_mee(problem, revolutions=2, nodes=150)
    assert result.converged, result.outcome.extras
    assert result.replay_terminal_position_error < 1.0e-4
    assert result.replay_terminal_velocity_error < 1.0e-4
    assert 0.65 < result.final_mass < 0.72


def test_tops_p1_highly_elliptic_free_time_converges_with_one_revolution(
    tops_problems: dict[str, lt.LowThrustProblem],
) -> None:
    """The highly elliptic guess needs the robust schedule (stiff first stage + hard radius)."""

    problem = tops_problems["two_body_cartesian:P1"]
    assert not problem.fixed_time
    result = mee.solve_low_thrust_mee(
        problem,
        revolutions=1,
        nodes=150,
        trust_schedule=mee.ROBUST_TRUST_SCHEDULE,
        hard_trust_radius=mee.ROBUST_HARD_TRUST_RADIUS,
    )
    assert result.converged, result.outcome.extras
    assert result.replay_terminal_position_error < 1.0e-4
    assert result.replay_terminal_velocity_error < 1.0e-4
    assert problem.tof_bounds[0] < result.time_of_flight < problem.tof_bounds[1]
    assert 0.85 < result.final_mass < 0.90


def test_tops_p1_zero_revolutions_is_infeasible_evidence(
    tops_problems: dict[str, lt.LowThrustProblem],
) -> None:
    """N = 0 would require holding true longitude for >= 60 TU: the transcription cannot close."""

    problem = tops_problems["two_body_cartesian:P1"]
    result = mee.solve_low_thrust_mee(
        problem,
        revolutions=0,
        nodes=100,
        trust_schedule=((1.0e-1, 12),),
        hard_trust_radius=mee.ROBUST_HARD_TRUST_RADIUS,
    )
    assert not (
        result.converged
        and result.replay_terminal_position_error < 1.0e-4
        and result.replay_terminal_velocity_error < 1.0e-4
    )
