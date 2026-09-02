"""P1-E Earth-to-Mars reproduction (fast configuration) and problem scaling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spacepdhcg.literature import low_thrust as lt

ROOT = Path(__file__).resolve().parents[1]


def _document(name: str) -> dict:
    return json.loads(
        (ROOT / "benchmarks" / "literature" / "profiles" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )


def test_si_scaling_matches_tops_conventions() -> None:
    problem = lt.problem_from_document(_document("tafazzol-taheri-earth-dionysus"))
    # TOPS MEE P0 (same physical case) lists max_thrust 0.013490919... and veff 0.98774597...
    assert abs(problem.max_thrust - 0.0134909) < 2.0e-6
    assert abs(problem.exhaust_velocity - 0.987746) < 2.0e-5
    assert abs(problem.time_of_flight - 60.7909) < 2.0e-3
    assert np.isclose(np.linalg.norm(problem.initial_state[:3]), 0.98376, atol=1e-3)


def test_earth_mars_reproduces_published_final_mass_within_envelope() -> None:
    problem = lt.problem_from_document(_document("tafazzol-taheri-earth-mars"))
    result = lt.solve_low_thrust(problem, nodes=60, max_iterations=20)
    assert result.outcome.status in {"converged", "maximum_iterations"}
    assert result.outcome.replay_defect_inf < 1.0e-6
    assert result.max_path_violation < 1.0e-8
    # Coarse 60-node FOH grid: within 1 kg of the published 603.935 kg.
    assert abs(result.final_mass_si - 603.935) < 1.0


def test_element_interpolation_guess_hits_boundaries_and_revolutions() -> None:
    problem = lt.problem_from_document(_document("tafazzol-taheri-earth-dionysus"))
    states, controls = lt.element_interpolation_guess(problem, 200, revolutions=5)
    assert np.allclose(states[0, :6], problem.initial_state)
    assert np.allclose(states[-1, :6], problem.final_state)
    angles = np.unwrap(np.arctan2(states[:, 1], states[:, 0]))
    assert angles[-1] - angles[0] > 2.0 * np.pi * 5.0 - 1.0
    assert controls.shape == (200, 4)
