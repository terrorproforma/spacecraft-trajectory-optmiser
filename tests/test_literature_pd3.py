"""P1-C Mars 3-DoF literature profile: lossless-convexification reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_forward_euler_diagnostic_isolates_the_discretisation_share_of_the_gap() -> None:
    profile = _profile("acikmese-ploen-2007-pd3")
    exact = pd3.solve_lossless_convexification(profile, dt=1.0)
    euler = pd3.solve_lossless_convexification(profile, dt=1.0, discretisation="forward_euler")
    # Measured 2026-09: 404.48 vs 400.63 kg; the Euler map alone costs several kilograms.
    assert 2.0 < euler.fuel_used - exact.fuel_used < 6.0


def test_module_blackmore_constant_matches_profile_document() -> None:
    document = _profile("blackmore-2010-pd3-case1")
    assert pd3.BLACKMORE_2010_CASE1 == document


def test_repository_scvx_accurate_option_closes_the_fuel_gap_2007() -> None:
    """Regression for the 6 kg gap: rk4 + multiple-shooting merit reaches the published value."""

    profile = _profile("acikmese-ploen-2007-pd3")
    result = pd3.solve_repository_scvx(profile, dt=1.0, max_iterations=80)
    assert result.discretisation == "rk4"
    assert result.merit_mode == "multiple_shooting"
    assert result.status == "converged", result.termination_reason
    # Published 399.5 kg (declared envelope 2.0 kg); measured 399.36 kg.
    assert abs(result.fuel_used - 399.5) <= 0.5
    assert abs(result.replay_fuel_used - result.fuel_used) < 1.0e-3
    assert result.replay_terminal_position_error < 1.0e-2
    assert result.replay_terminal_velocity_error < 1.0e-3
    assert result.path_max_violation < 1.0e-6
    lossless = pd3.solve_lossless_convexification(profile, dt=1.0)
    # The SCvx solves the non-relaxed problem, so it may only undercut the lossless SOCP by the
    # conservatism of the convex throttle bounds (measured 1.27 kg), never exceed it by more
    # than the discretisation envelope.
    assert -2.0 < result.fuel_used - lossless.fuel_used < 0.5


def test_repository_scvx_accurate_option_closes_the_fuel_gap_blackmore_2010() -> None:
    profile = _profile("blackmore-2010-pd3-case1")
    # dt = 0.8 s (98 intervals) keeps the test under half a minute; the report runs 0.4/0.2 s.
    result = pd3.solve_repository_scvx(profile, dt=0.8, max_iterations=120)
    assert result.status == "converged", result.termination_reason
    # Published 399.4 kg (declared envelope 2.0 kg); measured 398.84 kg at dt = 0.4 s.
    assert abs(result.fuel_used - 399.4) <= 1.0
    assert result.replay_terminal_position_error < 1.0e-2


def test_frozen_default_configuration_is_untouched() -> None:
    """The frozen benchmark fixtures rely on these defaults; the accurate path is opt-in."""

    from spacepdhcg.scvx import PoweredDescentOuterConfig
    from spacepdhcg.transcription import PoweredDescentSCvxConfig

    config = PoweredDescentSCvxConfig()
    assert config.discretisation == "forward_euler"
    assert config.integration_substeps == 1
    outer = PoweredDescentOuterConfig()
    assert outer.merit_mode == "single_shooting"
    assert outer.accept_almost_solved is False
    assert outer.stall_merit_tolerance == 0.0
    with pytest.raises(ValueError):
        PoweredDescentSCvxConfig(discretisation="midpoint")
    with pytest.raises(ValueError):
        PoweredDescentOuterConfig(merit_mode="shooting")
