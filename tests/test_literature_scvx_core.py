"""Free-final-time SCvx core: discretisation exactness, Jacobians, and a small end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from spacepdhcg.literature import low_thrust as lt
from spacepdhcg.literature import pd6_szmuk_2018 as pd6
from spacepdhcg.literature.scvx_core import discretise_interval, propagate_interval


def test_linear_system_discretisation_matches_matrix_exponential() -> None:
    A = np.array([[0.0, 1.0], [-2.0, -0.3]])
    B = np.array([[0.0], [1.0]])
    sigma = 1.7
    d_tau = 0.25

    def f(x, u):
        return A @ x + B @ u

    def jac(x, u):
        return A, B

    x0 = np.array([0.3, -0.2])
    u0 = np.array([0.5])
    u1 = np.array([-0.4])
    interval = discretise_interval(f, jac, x0, u0, u1, sigma, d_tau, substeps=16)
    # Exact STM for the dilated system over d_tau.
    assert np.allclose(interval.A, expm(sigma * A * d_tau), atol=1e-8)
    # For a linear system the affine map is exact: reproduce the propagated state.
    reconstructed = (
        interval.A @ x0 + interval.B @ u0 + interval.C @ u1 + interval.S * sigma + interval.z
    )
    assert np.allclose(reconstructed, interval.propagated, atol=1e-9)
    replayed = propagate_interval(f, x0, u0, u1, sigma, d_tau, substeps=200)
    assert np.allclose(replayed, interval.propagated, atol=1e-8)


def test_nonlinear_discretisation_is_consistent_at_the_reference() -> None:
    p = pd6.Szmuk2018Parameters()
    dyn = pd6.Szmuk2018Dynamics(p)
    states, controls = pd6.initial_guess(p)
    k = 10
    interval = discretise_interval(
        dyn.f, dyn.jacobians, states[k], controls[k], controls[k + 1], 4.0, 1.0 / 49, substeps=8
    )
    reconstructed = (
        interval.A @ states[k]
        + interval.B @ controls[k]
        + interval.C @ controls[k + 1]
        + interval.S * 4.0
        + interval.z
    )
    assert np.allclose(reconstructed, interval.propagated, atol=1e-9)
    # The dilation column equals the sensitivity of the propagated state to sigma.
    eps = 1e-6
    plus = discretise_interval(
        dyn.f,
        dyn.jacobians,
        states[k],
        controls[k],
        controls[k + 1],
        4.0 + eps,
        1.0 / 49,
        substeps=8,
    ).propagated
    minus = discretise_interval(
        dyn.f,
        dyn.jacobians,
        states[k],
        controls[k],
        controls[k + 1],
        4.0 - eps,
        1.0 / 49,
        substeps=8,
    ).propagated
    assert np.allclose((plus - minus) / (2 * eps), interval.S, atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1])
def test_szmuk_jacobians_match_finite_differences(seed: int) -> None:
    rng = np.random.default_rng(seed)
    dyn = pd6.Szmuk2018Dynamics(pd6.Szmuk2018Parameters())
    x = np.concatenate([[1.5], rng.normal(size=6), rng.normal(size=4), 0.3 * rng.normal(size=3)])
    x[7:11] /= np.linalg.norm(x[7:11])
    u = np.array([2.0, 0.3, -0.2]) + 0.1 * rng.normal(size=3)
    A, B = dyn.jacobians(x, u)
    eps = 1e-6
    for i in range(14):
        d = np.zeros(14)
        d[i] = eps
        column = (dyn.f(x + d, u) - dyn.f(x - d, u)) / (2 * eps)
        assert np.allclose(A[:, i], column, atol=1e-6)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        column = (dyn.f(x, u + d) - dyn.f(x, u - d)) / (2 * eps)
        assert np.allclose(B[:, i], column, atol=1e-6)


def test_low_thrust_jacobians_and_element_roundtrip() -> None:
    problem = lt.LowThrustProblem(
        initial_state=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        final_state=(-1.5, 0.2, 0.1, -0.1, -0.8, 0.05),
        initial_mass=1.0,
        max_thrust=0.05,
        exhaust_velocity=0.7,
        tof_bounds=(6.0, 6.0),
    )
    dyn = lt.LowThrustDynamics(problem)
    x = np.array([1.1, -0.3, 0.05, 0.2, 0.9, -0.01, 0.8])
    u = np.array([0.01, -0.02, 0.005, 0.03])
    A, B = dyn.jacobians(x, u)
    eps = 1e-6
    for i in range(7):
        d = np.zeros(7)
        d[i] = eps
        assert np.allclose(A[:, i], (dyn.f(x + d, u) - dyn.f(x - d, u)) / (2 * eps), atol=1e-6)
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        assert np.allclose(B[:, i], (dyn.f(x, u + d) - dyn.f(x, u - d)) / (2 * eps), atol=1e-6)
    for state in (problem.initial_state, problem.final_state):
        elements = lt.cartesian_to_elements(1.0, np.asarray(state))
        assert np.allclose(lt.elements_to_cartesian(1.0, *elements), state, atol=1e-10)


def test_small_free_final_time_run_is_dynamically_feasible() -> None:
    p = pd6.Szmuk2018Parameters(nodes=12, max_iterations=6)
    result = pd6.reproduce(p, tf_guess=3.0, substeps=4)
    assert result.outcome.status in {"converged", "maximum_iterations"}
    assert result.time_of_flight > 0.0
    assert np.isfinite(result.fuel_used)
    assert result.outcome.replay_defect_inf < 1e-2
    assert result.outcome.iterations[-1].virtual_l1 < 1e-6
