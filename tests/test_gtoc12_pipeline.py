"""Lambert parity, reduced-instance determinism, SCvx leg certification and search determinism."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import data_available, load_catalogue
from spacepdhcg.gtoc12.ephemeris import elements_to_state, propagate_kepler
from spacepdhcg.gtoc12.lambert import NativeLambert, lambert_batch
from spacepdhcg.gtoc12.low_thrust import LegBoundary, ScvxSettings, certify_leg, solve_leg
from spacepdhcg.gtoc12.reduced_instance import DEFAULT_RULE_PATH, build_reduced_instance, load_rule

requires_data = pytest.mark.skipif(not data_available(), reason="pinned GTOC12 data not fetched")
REDUCED_RULE_SHA256 = "718dd7e76f8f09295ae53de58b56626c5d8eb42fa397a27ab190b6511b39bd25"
REDUCED_SELECTION_SHA256 = "e2bbbca1ca31afdcb8272fbecb54c932884b343f394a39ec91e5cbc5da5d7781"


def _circular_states(radius_au: float, phase: float, inclination: float = 0.0):
    r, v = elements_to_state(
        np.array([radius_au * C.AU_KM]),
        np.array([0.0]),
        np.array([inclination]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([phase]),
    )
    return r[0], v[0]


def test_numpy_lambert_closes_under_kepler_propagation() -> None:
    rng = np.random.default_rng(3)
    n = 64
    r1 = np.stack([_circular_states(1.0, p)[0] for p in rng.uniform(0, 2 * np.pi, n)])
    r2 = np.stack([_circular_states(2.7, p, 0.05)[0] for p in rng.uniform(0, 2 * np.pi, n)])
    tof = rng.uniform(200, 700, n) * C.DAY_S
    result = lambert_batch(r1, r2, tof)
    assert result.feasible.all()
    rp, vp = propagate_kepler(r1, result.departure_velocity, tof)
    assert np.max(np.linalg.norm(rp - r2, axis=1)) < 0.05  # km
    assert np.max(np.linalg.norm(vp - result.arrival_velocity, axis=1)) < 1e-6


@pytest.mark.skipif(
    shutil.which("c++") is None and shutil.which("g++") is None, reason="no C++ compiler"
)
def test_numpy_lambert_matches_native_cpu_kernel(tmp_path: Path) -> None:
    native = NativeLambert(tmp_path / "libspacepdhcg_c_api.so")
    rng = np.random.default_rng(5)
    n = 40
    r1 = np.stack([_circular_states(1.0, p)[0] for p in rng.uniform(0, 2 * np.pi, n)])
    r2 = np.stack([_circular_states(2.8, p, 0.1)[0] for p in rng.uniform(0, 2 * np.pi, n)])
    tof = rng.uniform(150, 900, n) * C.DAY_S
    for long_way in (False, True):
        ours = lambert_batch(r1, r2, tof, long_way=long_way)
        for k in range(n):
            theirs = native.solve(r1[k], r2[k], tof[k], long_way=long_way)
            assert (theirs is None) == (not ours.feasible[k])
            if theirs is not None:
                assert np.allclose(theirs[0], ours.departure_velocity[k], atol=1e-10)
                assert np.allclose(theirs[1], ours.arrival_velocity[k], atol=1e-10)
    stride, rows = native.family_batch(r1[:8], r2[:8], tof[:8])
    assert stride == 2 and len(rows) == 16
    assert all(row["status"] == 0 for row in rows)


def test_reduced_instance_rule_is_pinned() -> None:
    rule, digest = load_rule(DEFAULT_RULE_PATH)
    assert digest == REDUCED_RULE_SHA256
    assert rule["defined_before_any_search"] is True
    assert rule["selection"]["count"] == 1000 and rule["fleet"]["ships"] == 1


@requires_data
def test_reduced_instance_selection_is_deterministic() -> None:
    catalogue = load_catalogue()
    first = build_reduced_instance(catalogue)
    second = build_reduced_instance(catalogue)
    assert first.asteroid_ids.tolist() == second.asteroid_ids.tolist()
    assert first.asteroid_ids.shape == (1000,)
    assert first.selection_sha256 == REDUCED_SELECTION_SHA256
    assert first.eligible_count == 9803
    a_au = catalogue.semi_major_axis_km[first.asteroid_ids - 1] / C.AU_KM
    assert a_au.min() >= 2.5 and a_au.max() <= 3.2
    assert np.rad2deg(catalogue.inclination_rad[first.asteroid_ids - 1]).max() <= 6.0


def test_scvx_leg_between_synthetic_circular_orbits_is_certified() -> None:
    """A 150-day hop between near-coplanar circular orbits at 2.70 and 2.73 AU."""

    tof_days = 150.0
    r0, v0 = _circular_states(2.70, 0.0)
    # the target starts in phase with the ship and drifts with its own mean motion
    n_target = np.sqrt(C.MU_SUN_KM3_S2 / (2.73 * C.AU_KM) ** 3)
    rf, vf = _circular_states(2.73, n_target * tof_days * C.DAY_S, 0.002)
    boundary = LegBoundary(65000.0, r0, v0, 65000.0 + tof_days, rf, vf, 2500.0)
    solution = solve_leg(boundary, ScvxSettings(max_iterations=30, time_limit_s=300.0))
    assert solution.status == "converged", (solution.status, solution.diagnostic)
    assert solution.final_mass_kg < 2500.0
    certificate = certify_leg(solution)
    assert certificate.within_tolerance
    assert certificate.position_error_km < 50.0
    assert certificate.velocity_error_km_s < 1e-4
    assert certificate.maximum_thrust_n <= C.THRUST_MAX_N + 1e-9
    assert len(solution.burn_arcs()) >= 1


@requires_data
def test_search_is_deterministic_on_small_subset() -> None:
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
    from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids[:60]
    settings = SearchSettings(
        beam_width=4,
        max_deploys=2,
        neighbours=8,
        launch_epochs=(64328.0, 64508.0),
        earth_leg_tofs=(400.0, 600.0),
        hop_tofs=(90.0, 180.0),
    )
    first = RouteSearch(catalogue, ids, settings).run()
    second = RouteSearch(catalogue, ids, settings).run()
    assert [item.summary() for item in first.candidates] == [
        item.summary() for item in second.candidates
    ]
    assert first.lambert_evaluations == second.lambert_evaluations
