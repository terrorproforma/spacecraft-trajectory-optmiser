"""P1-C Mars 3-DoF literature profile: lossless-convexification reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spacepdhcg.literature import pd3_acikmese_ploen as pd3

ROOT = Path(__file__).resolve().parents[1]


def _profile(name: str) -> pd3.MarsDescentProfile:
    document = json.loads(
        (ROOT / "benchmarks" / "literature" / "profiles" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )
    return pd3.profile_from_document(document)


def test_vehicle_constants_match_blackmore_2010_eq_72() -> None:
    profile = _profile("acikmese-ploen-2007-pd3")
    assert abs(profile.rho_min - 4972.0) < 1.0
    assert abs(profile.rho_max - 13260.0) < 2.0
    assert abs(profile.replace(alpha_convention="isp-only").alpha - 4.53e-4) < 1.0e-6


def test_lossless_socp_reproduces_2007_glide_slope_fuel_within_envelope() -> None:
    profile = _profile("acikmese-ploen-2007-pd3")
    result = pd3.solve_lossless_convexification(profile, dt=1.0)
    assert result.solver_status in {"Solved", "AlmostSolved"}
    assert abs(result.fuel_used - 399.5) < 2.0  # declared envelope
    assert abs(result.replay_fuel_used - result.fuel_used) < 1.0e-2
    assert result.replay_terminal_position_error < 1.0e-6
    assert result.replay_terminal_velocity_error < 1.0e-8
    assert result.min_throttle_violation == 0.0
    assert result.max_throttle_violation == 0.0
    assert result.glide_slope_violation == 0.0


def test_lossless_socp_reproduces_2007_no_glide_slope_case() -> None:
    profile = _profile("acikmese-ploen-2007-pd3").replace(time_of_flight=72.0, glide_slope_deg=None)
    result = pd3.solve_lossless_convexification(profile, dt=1.0)
    assert abs(result.fuel_used - 387.9) < 1.0


def test_isp_only_alpha_convention_does_not_reproduce_the_paper() -> None:
    profile = _profile("acikmese-ploen-2007-pd3").replace(alpha_convention="isp-only")
    result = pd3.solve_lossless_convexification(profile, dt=1.0)
    assert abs(result.fuel_used - 399.5) > 30.0


def test_frame_mapping_round_trip() -> None:
    state = pd3.paper_to_repository_state((1500.0, 0.0, 2000.0), (-75.0, 0.0, 100.0), 1905.0)
    assert np.allclose(state, [2000.0, 0.0, 1500.0, 100.0, 0.0, -75.0, 1905.0])
    controls = np.array([[1.0, 2.0, 3.0, 9.0]])
    assert np.allclose(pd3.repository_to_paper_thrust(controls), [[3.0, 2.0, 1.0]])


def test_replay_hold_modes_are_consistent() -> None:
    profile = _profile("acikmese-ploen-2007-pd3")
    result = pd3.solve_lossless_convexification(profile, dt=1.0)
    accelerations = result.thrust / np.exp(result.log_mass[:-1])[:, None]
    states, fuel = pd3.replay_zoh(profile, accelerations, 1.0, hold="acceleration")
    assert abs(fuel - result.fuel_used) < 1.0e-2
    assert states.shape == (82, 7)
