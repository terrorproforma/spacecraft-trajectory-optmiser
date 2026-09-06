"""Opt-in native GTOC12 interval parity and independent trajectory qualification."""

from __future__ import annotations

import os
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.gpu_discretisation import GpuDiscretisation
from spacepdhcg.gtoc12.low_thrust import (
    LegBoundary,
    ScvxSettings,
    _Discretisation,
    _Model,
    certify_leg,
    solve_leg,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires explicit CUDA GTOC12 tests; serialize with other GPU work",
)


def fixture(intervals, hold):
    rng = np.random.default_rng(9823 + intervals)
    times = 13.0 + np.r_[0.0, np.cumsum(rng.uniform(0.001, 0.04, intervals))]
    states = rng.normal(size=(intervals + 1, 7)) * 0.1
    states[:, :3] += [2.7, 0.0, 0.0]
    states[:, 6] = rng.uniform(0.3, 1.0, intervals + 1)
    controls = rng.uniform(-0.3, 0.3, (intervals + 1, 4))
    controls[::3, :3] = 0.0
    controls[:, 3] = np.linalg.norm(controls[:, :3], axis=1) + 0.03
    return _Model(2500.0), times, states, controls


@pytest.mark.parametrize("hold", ["zoh", "lagrange"])
@pytest.mark.parametrize("intervals", [3, 17, 257, 2001])
@pytest.mark.parametrize("substeps", [1, 8, 16])
def test_interval_coefficients_match_cpu_reference(hold, intervals, substeps):
    model, times, states, controls = fixture(intervals, hold)
    reference = _Discretisation(model, times, substeps, hold)
    with closing(GpuDiscretisation(model, times, substeps, hold)) as gpu:
        np.testing.assert_array_equal(gpu.stencils, reference.stencils)
        expected = reference.linearise(states, controls)
        actual = gpu.linearise(states, controls)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(
            gpu.propagate(states, controls),
            reference.propagate(states, controls),
            atol=2e-12,
            rtol=2e-12,
        )
        a, b, c, propagated = actual
        closure = (
            np.einsum("nij,nj->ni", a, states[:-1])
            + np.einsum("nsij,nsj->ni", b, controls[gpu.stencils])
            + c
        )
        np.testing.assert_allclose(closure, propagated, atol=2e-13, rtol=2e-13)
        if hold == "zoh":
            # Independent analytic mass evolution; Gamma intentionally differs
            # from |u_xyz|. The surrogate sensitivity is a separate invariant.
            mass = states[:-1, 6] - model.lam * np.linalg.norm(controls[:-1, :3], axis=1) * np.diff(
                times
            )
            np.testing.assert_allclose(propagated[:, 6], mass, atol=3e-15, rtol=3e-15)
            np.testing.assert_allclose(b[:, 0, 6, 3], -model.lam * np.diff(times), atol=1e-15)


def test_workspace_reuse_polish_and_bad_inputs():
    model, times, states, controls = fixture(17, "lagrange")
    gpu = GpuDiscretisation(model, times, 8, "lagrange")
    try:
        previous = gpu.linearise(states, controls)
        saved = [a.copy() for a in previous]
        for repeat in range(6):
            gpu.substeps = 16 if repeat % 2 else 8
            controls[:, 0] += 0.005
            actual = gpu.linearise(states, controls)
            expected = _Discretisation(model, times, gpu.substeps, "lagrange").linearise(
                states, controls
            )
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
        for a, b in zip(previous, saved, strict=True):
            np.testing.assert_array_equal(a, b)
        for value in [-1.0, np.nan, 0.0]:
            corrupt = states.copy()
            corrupt[0, 6] = value
            with pytest.raises(RuntimeError, match="status 3"):
                gpu.linearise(corrupt, controls)
        # Invalid dynamics flag resets; the workspace is reusable after failure.
        np.testing.assert_allclose(
            gpu.propagate(states, controls),
            _Discretisation(model, times, gpu.substeps, "lagrange").propagate(states, controls),
            atol=2e-12,
            rtol=2e-12,
        )
        with pytest.raises(ValueError, match="states"):
            gpu.propagate(states[:-1], controls)
    finally:
        gpu.close()
        gpu.close()
    with pytest.raises(RuntimeError, match="closed"):
        gpu.propagate(states, controls)


def test_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.delenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", raising=False)
    model, times, _, _ = fixture(3, "zoh")
    with pytest.raises(RuntimeError, match="SPACEPDHCG_GTOC12_CUDA_LIBRARY"):
        GpuDiscretisation(model, times, 8)


def test_topology_is_owned_and_read_only():
    model, times, states, controls = fixture(17, "zoh")
    expected = _Discretisation(model, times.copy(), 8).propagate(states, controls)
    with closing(GpuDiscretisation(model, times, 8)) as gpu:
        times[:] = 0.0
        assert not gpu.node_times.flags.writeable
        assert not gpu.stencils.flags.writeable
        np.testing.assert_allclose(
            gpu.propagate(states, controls), expected, atol=2e-12, rtol=2e-12
        )


def test_workspace_closes_when_initial_reference_fails(monkeypatch):
    from spacepdhcg.gtoc12 import low_thrust

    closed = []
    original_close = GpuDiscretisation.close

    def close(gpu):
        original_close(gpu)
        closed.append(not gpu._finalizer.alive)

    def fail(*args):
        raise RuntimeError("injected reference failure")

    monkeypatch.setattr(GpuDiscretisation, "close", close)
    monkeypatch.setattr(low_thrust, "_ballistic_reference", fail)
    with pytest.raises(RuntimeError, match="injected reference failure"):
        solve_leg(synthetic_boundary(), ScvxSettings(discretisation_backend="cuda"))
    assert closed == [True]


def test_cuda_leg_retains_outer_time_limit():
    result = solve_leg(
        synthetic_boundary(), ScvxSettings(time_limit_s=0.0, discretisation_backend="cuda")
    )
    assert result.status == "timeout" and result.iterations == 0
    assert result.discretisation_backend == "cuda"


def synthetic_boundary():
    from spacepdhcg.gtoc12.ephemeris import elements_to_state

    def circular(radius, phase, inclination=0.0):
        r, v = elements_to_state(
            np.array([radius * C.AU_KM]),
            np.array([0.0]),
            np.array([inclination]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([phase]),
        )
        return r[0], v[0]

    r0, v0 = circular(2.7, 0.0)
    mean_motion = np.sqrt(C.MU_SUN_KM3_S2 / (2.73 * C.AU_KM) ** 3)
    rf, vf = circular(2.73, mean_motion * 150.0 * C.DAY_S, 0.002)
    return LegBoundary(65000.0, r0, v0, 65150.0, rf, vf, 2500.0)


def test_full_leg_retains_independent_physics_qualification():
    boundary = synthetic_boundary()
    settings = ScvxSettings(max_iterations=30, time_limit_s=300.0)
    reference = solve_leg(boundary, settings)
    actual = solve_leg(boundary, replace(settings, discretisation_backend="cuda"))
    assert reference.status == actual.status == "converged"
    assert reference.discretisation_backend == "numpy"
    assert actual.discretisation_backend == "cuda"
    for solution in [reference, actual]:
        certificate = certify_leg(solution)
        assert certificate.within_tolerance
        assert certificate.maximum_thrust_n <= C.THRUST_MAX_N + 1e-9
    assert abs(actual.final_mass_kg - reference.final_mass_kg) <= 1e-5
    np.testing.assert_allclose(actual.states_scaled, reference.states_scaled, atol=1e-7, rtol=0)
