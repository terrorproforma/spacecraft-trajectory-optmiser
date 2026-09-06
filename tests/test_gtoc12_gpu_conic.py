"""GPU conic assembly parity, changing-data layout and complete-leg qualification."""

from __future__ import annotations

import os
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from test_gtoc12_gpu_discretisation import fixture, synthetic_boundary

from spacepdhcg.gtoc12.gpu_conic import GpuConvexProblem
from spacepdhcg.gtoc12.low_thrust import (
    ScvxSettings,
    _ConvexProblem,
    _Discretisation,
    certify_leg,
    solve_leg,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="explicit serialized CUDA tests",
)


def boundary_and_weights(states, times):
    boundary = dict(
        r0=states[0, :3].copy(), v0=states[0, 3:6].copy(),
        rf=states[-1, :3].copy(), vf=states[-1, 3:6].copy(),
    )
    weights = np.r_[np.diff(times), 0.0] * 0.03
    return boundary, weights


def assert_same(actual, expected):
    a, b, q, cones, p = actual
    ea, eb, eq, ec, ep = expected
    for mat, ref in [(a, ea), (p, ep)]:
        assert mat.shape == ref.shape
        delta = mat - ref
        np.testing.assert_allclose(delta.data, 0.0, atol=2e-12, rtol=0.0)
        assert mat.has_canonical_format
    np.testing.assert_allclose(b, eb, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(q, eq)
    assert [str(c) for c in cones] == [str(c) for c in ec]


@pytest.mark.parametrize("hold", ["zoh", "lagrange"])
@pytest.mark.parametrize("intervals", [3, 17, 257, 2001])
@pytest.mark.parametrize(
    "free_dep,free_arr", [(False, False), (True, False), (False, True), (True, True)]
)
def test_conic_values_and_topology_reuse(hold, intervals, free_dep, free_arr):
    model, times, states, controls = fixture(intervals, hold)
    bnd, weights = boundary_and_weights(states, times)
    ref = _ConvexProblem(len(times), free_dep, free_arr)
    with closing(GpuConvexProblem(model, times, hold, free_dep, free_arr, bnd, weights)) as gpu:
        old = None
        topology = [a.copy() for a in gpu._topology]
        for repeat in range(3):
            # Include coast-to-thrust changes, zero smoothness, and polish RK4 steps.
            controls[:, :3] = 0.0 if repeat == 0 else controls[:, :3] + 0.02
            states[:, 0] += 0.0001
            substeps = 1 if repeat == 0 else 8 if repeat == 1 else 16
            parameters = (
                0.1 / (repeat + 1),
                0.3 / (repeat + 1),
                13.0 + repeat,
                0.3,
                0.05,
                0.02,
                0.001 * repeat,
            )
            cpu_disc = _Discretisation(model, times, substeps, hold)
            phi, psi, c, _ = cpu_disc.linearise(states, controls)
            expected = ref.build(
                phi,
                psi,
                c,
                cpu_disc.stencils,
                states,
                controls,
                bnd,
                *parameters[:6],
                weights,
                parameters[6],
                zero_last_control=hold == "zoh",
            )
            actual = gpu.build_linearised(states, controls, substeps, *parameters)
            assert_same(actual, expected)
            for current, saved in zip(gpu._topology, topology, strict=True):
                np.testing.assert_array_equal(current, saved)
                assert not current.flags.writeable
            if old is not None:
                np.testing.assert_array_equal(old[0].data, old[1])
            old = (actual[0], actual[0].data.copy())


def test_conic_rejects_invalid_and_recovers():
    model, times, states, controls = fixture(17, "lagrange")
    bnd, weights = boundary_and_weights(states, times)
    gpu = GpuConvexProblem(model, times, "lagrange", True, True, bnd, weights)
    parameters = (0.1, 0.3, 13.0, 0.3, 0.05, 0.02, 0.001)
    try:
        for index in range(7):
            bad = list(parameters)
            bad[index] = float("nan")
            with pytest.raises(RuntimeError, match="status 3"):
                gpu.build_linearised(states, controls, 8, *bad)
        for index in [0, -1]:
            bad_states = states.copy()
            bad_states[index, :3] = 0.0
            with pytest.raises(RuntimeError, match="status 3"):
                gpu.build_linearised(bad_states, controls, 8, *parameters)
        with pytest.raises(ValueError, match="states"):
            gpu.build_linearised(states[:-1], controls, 8, *parameters)
        with pytest.raises(ValueError, match="substeps"):
            gpu.build_linearised(states, controls, 0, *parameters)
        actual = gpu.build_linearised(states, controls, 8, *parameters)
        assert np.isfinite(actual[0].data).all()
    finally:
        gpu.close()
        gpu.close()
    with pytest.raises(RuntimeError, match="closed"):
        gpu.build_linearised(states, controls, 8, *parameters)


@pytest.mark.parametrize("hold,node_days", [("zoh", 2.0), ("lagrange", 2.0), ("zoh", 8.0)])
def test_complete_leg_assembly_same_qualified_accuracy(hold, node_days):
    boundary = synthetic_boundary()
    settings = ScvxSettings(
        max_iterations=30,
        time_limit_s=300.0,
        hold=hold,
        node_days=node_days,
        discretisation_backend="cuda",
    )
    baseline = solve_leg(boundary, settings)
    candidate = solve_leg(boundary, replace(settings, assembly_backend="cuda"))
    assert baseline.status == candidate.status == "converged"
    assert candidate.assembly_backend == "cuda"
    for solution in [baseline, candidate]:
        assert certify_leg(solution).within_tolerance
    assert abs(candidate.final_mass_kg - baseline.final_mass_kg) <= 1e-5


def test_conic_closes_on_solver_exception(monkeypatch):
    from spacepdhcg.gtoc12 import low_thrust

    closed = []
    original = GpuConvexProblem.close

    def close(self):
        closed.append(self)
        original(self)

    def fail(*args):
        raise RuntimeError("injected solve error")

    monkeypatch.setattr(GpuConvexProblem, "close", close)
    monkeypatch.setattr(low_thrust, "_clarabel_solve", fail)
    with pytest.raises(RuntimeError, match="injected solve error"):
        solve_leg(
            synthetic_boundary(),
            ScvxSettings(discretisation_backend="cuda", assembly_backend="cuda"),
        )
    assert len(closed) == 1 and not closed[0]._finalizer.alive
