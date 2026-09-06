"""Native GPU solver integration, independent objective checks and rejection gates.

Coast fixtures verify a known zero-cost optimum. Full transfer qualification and
matched objective/timing evidence are recorded separately; plumbing tests do not
assert that every transfer converges.
"""

from __future__ import annotations

import os
from contextlib import closing

import numpy as np
import pytest
from test_gtoc12_gpu_discretisation import synthetic_boundary

from spacepdhcg.gtoc12.gpu_qoco import GpuQocoProblem
from spacepdhcg.gtoc12.low_thrust import (
    ScvxSettings,
    _ConvexProblem,
    _Discretisation,
    _Model,
    solve_leg,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires serialized GPU QOCO runtime",
)


def coast_fixture(hold):
    model = _Model(2500.0)
    times = np.arange(4) * 0.003
    omega = 2.7**-1.5
    states = np.zeros((4, 7))
    states[:, 0] = 2.7 * np.cos(omega * times)
    states[:, 1] = 2.7 * np.sin(omega * times)
    states[:, 3] = -2.7 * omega * np.sin(omega * times)
    states[:, 4] = 2.7 * omega * np.cos(omega * times)
    states[:, 6] = 1.0
    controls = np.zeros((4, 4))
    boundary = dict(r0=states[0, :3], v0=states[0, 3:6], rf=states[-1, :3], vf=states[-1, 3:6])
    weights = np.full(4, 0.003)
    if hold == "zoh":
        weights[-1] = 0.0
    return model, times, states, controls, boundary, weights


@pytest.mark.parametrize("hold", ["zoh", "lagrange"])
@pytest.mark.parametrize(
    "free_dep,free_arr", [(False, False), (True, False), (False, True), (True, True)]
)
def test_known_coast_optimum_and_device_updates(hold, free_dep, free_arr):
    model, times, states, controls, boundary, weights = coast_fixture(hold)
    with closing(
        GpuQocoProblem(model, times, hold, free_dep, free_arr, boundary, weights, 1e-9)
    ) as gpu:
        previous = None
        for repeat in range(3):
            penalty = 13.0 + repeat
            ok, _, x = gpu.solve_linearised(
                states, controls, 8, 0.1 / (repeat + 1), 0.3, penalty, 0.3, 0.05, 0.02, 0.001
            )
            r = gpu.last_report
            assert ok and r["qualified"] == 1, r
            assert max(r["primal_residual"], r["dual_residual"], r["relative_gap"]) <= 1e-9
            assert r["workspace_creations"] == 1 and r["device_numeric_updates"] == repeat
            if previous is not None:
                # Matrix values stay on device during successful updates. Only
                # topology/conversion flags and six audit scalars cross here.
                assert r["adapter_d2h_bytes"] - previous["adapter_d2h_bytes"] <= 64
            u = x[gpu.iu : gpu.inu].reshape(4, 4)
            objective = float(
                weights @ u[:, 3]
                + penalty * np.sum(x[gpu.isl : gpu.ivd])
                + 0.001 * np.sum(np.diff(u[:, :3], axis=0) ** 2)
            )
            assert abs(objective) <= 1e-7
            assert abs(objective - r["primal_objective"]) <= 1e-12
            assert abs(x[7 * 3 + 6] - 1.0) <= 1e-5
            # Audit the ORIGINAL uneliminated reference constraints, including
            # the redundant final-control cone and bounds omitted by QOCO.
            disc = _Discretisation(model, times, 8, hold)
            phi, psi, affine, _ = disc.linearise(states, controls)
            a, b, q, _, p = _ConvexProblem(4, free_dep, free_arr).build(
                phi,
                psi,
                affine,
                disc.stencils,
                states,
                controls,
                boundary,
                0.1 / (repeat + 1),
                0.3,
                penalty,
                0.3,
                0.05,
                0.02,
                weights,
                0.001,
                zero_last_control=hold == "zoh",
            )
            slack = b - a @ x
            d = gpu.dimensions
            assert np.max(np.abs(slack[: d.equalities])) <= 1e-9
            assert np.min(slack[d.equalities : d.equalities + d.inequalities]) >= -1e-9
            soc = slack[d.equalities + d.inequalities :].reshape(-1, 4)
            assert np.max(np.linalg.norm(soc[:, 1:], axis=1) - soc[:, 0]) <= 1e-9
            original_objective = float(q @ x + x @ p @ x - 0.5 * np.dot(p.diagonal(), x * x))
            assert abs(original_objective - r["primal_objective"]) <= 1e-12
            previous = r


def test_unqualified_output_and_bad_input_recovery():
    model, times, states, controls, boundary, weights = coast_fixture("zoh")
    params = (0.1, 0.3, 13.0, 0.3, 0.05, 0.02, 0.001)
    with closing(GpuQocoProblem(model, times, "zoh", True, True, boundary, weights, 1e-30)) as gpu:
        ok, _, x = gpu.solve_linearised(states, controls, 8, *params)
        assert not ok and np.isnan(x).all()
        assert gpu.last_report["qualified"] == 0 and gpu.last_report["api_status"] == 4
    with closing(GpuQocoProblem(model, times, "zoh", True, True, boundary, weights, 1e-9)) as gpu:
        assert gpu.solve_linearised(states, controls, 8, *params)[0]
        bad = states.copy()
        bad[0, 6] = -1.0
        with pytest.raises(RuntimeError, match="status 3"):
            gpu.solve_linearised(bad, controls, 8, *params)
        assert not gpu.last_report["qualified"]
        assert gpu.last_report["relative_gap"] is None
        assert gpu.last_report["absolute_primal_residual"] is None
        assert gpu.last_report["absolute_dual_residual"] is None
        assert gpu.solve_linearised(states, controls, 8, *params)[0]
    with pytest.raises(RuntimeError, match="closed"):
        gpu.solve_linearised(states, controls, 8, *params)


def test_missing_extensions_fail_without_fallback(tmp_path, monkeypatch):
    model, times, states, controls, boundary, weights = coast_fixture("zoh")
    fake = tmp_path / "not_a_solver.so"
    fake.write_bytes(b"deliberately missing QOCO extensions")
    monkeypatch.setenv("SPACEPDHCG_QOCO_LIBRARY", str(fake))
    with closing(GpuQocoProblem(model, times, "zoh", True, True, boundary, weights, 1e-9)) as gpu:
        with pytest.raises(RuntimeError, match="status 5"):
            gpu.solve_linearised(states, controls, 8, 0.1, 0.3, 13.0, 0.3, 0.05, 0.02, 0.001)


def test_assembly_validation_rejects_finite_bad_parameters_and_recovers(monkeypatch, capfd):
    model, times, states, controls, boundary, weights = coast_fixture("zoh")
    params = (0.1, 0.3, 13.0, 0.3, 0.05, 0.02, 0.001)
    guarded = os.environ.get("SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION") == "1"
    queued = guarded and all(
        os.environ.get(name) == "1"
        for name in (
            "SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION",
            "SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY",
            "SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY",
        )
    )
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION_TRACE", "1")
    cases = []
    for i in range(len(params)):
        for value in (-1.0, np.nan):
            changed = list(params)
            changed[i] = value
            cases.append((states, controls, changed))
    for slot, value in ((6, -1.0), (6, np.nan), (0, np.nan)):
        changed = states.copy()
        changed[0, slot] = value
        cases.append((changed, controls, params))
    changed = controls.copy()
    changed[1, 0] = np.nan
    cases.append((states, changed, params))

    with closing(GpuQocoProblem(model, times, "zoh", True, True, boundary, weights, 1e-9)) as gpu:
        # Invalid initial data must never reach host setup either.
        with pytest.raises(RuntimeError, match="status 3"):
            gpu.solve_linearised(states, controls, 8, -1.0, *params[1:])
        assert gpu.last_report["workspace_creations"] == 0
        for bad_states, bad_controls, bad_params in cases:
            # Rebuild, prime zero-Ruiz scaling, then exercise an actual replay.
            for _ in range(3):
                ok, _, x = gpu.solve_linearised(states, controls, 8, *params)
                assert ok, gpu.last_report
                assert (
                    max(
                        gpu.last_report[k]
                        for k in ("primal_residual", "dual_residual", "relative_gap")
                    )
                    <= 1e-9
                )
                assert abs(x[7 * 3 + 6] - 1.0) <= 1e-5
            solved = gpu.last_report["solves"]
            with pytest.raises(RuntimeError, match="status 3"):
                gpu.solve_linearised(bad_states, bad_controls, 8, *bad_params)
            report = gpu.last_report
            assert not report["qualified"] and report["iterations"] == 0
            assert all(
                report[k] is None
                for k in (
                    "primal_residual",
                    "dual_residual",
                    "absolute_primal_residual",
                    "absolute_dual_residual",
                    "primal_objective",
                    "dual_objective",
                    "relative_gap",
                )
            )
            if guarded:
                assert report["qoco_status"] == 3 and report["solves"] == solved
        assert gpu.solve_linearised(states, controls, 8, *params)[0]
    trace = capfd.readouterr().err
    if queued:
        assert trace.count("GTOC12_ASSEMBLY_VALIDATION queued=1 invalid=1") == len(cases)


def test_outer_loop_routes_directly_to_native_and_closes(monkeypatch):
    from spacepdhcg.gtoc12 import low_thrust

    closed = []
    original_close = GpuQocoProblem.close

    def close(self):
        closed.append(self)
        original_close(self)

    def no_cpu(*args, **kwargs):
        raise AssertionError("CPU conic assembly or solver was called")

    monkeypatch.setattr(GpuQocoProblem, "close", close)
    monkeypatch.setattr(low_thrust, "_clarabel_solve", no_cpu)
    monkeypatch.setattr(low_thrust._ConvexProblem, "build", no_cpu)
    result = solve_leg(
        synthetic_boundary(),
        ScvxSettings(
            discretisation_backend="cuda",
            assembly_backend="cuda",
            convex_solver_backend="qoco",
            max_iterations=1,
            polish_iterations=0,
        ),
    )
    assert result.convex_solver_backend == "qoco" and len(result.solver_reports) == 1
    assert len(closed) == 1 and not closed[0]._finalizer.alive


def test_outer_solver_exception_closes_workspace(monkeypatch):
    closed = []
    original_close = GpuQocoProblem.close

    def close(self):
        closed.append(self)
        original_close(self)

    def fail(*args):
        raise RuntimeError("injected native wrapper failure")

    monkeypatch.setattr(GpuQocoProblem, "close", close)
    monkeypatch.setattr(GpuQocoProblem, "solve_linearised", fail)
    with pytest.raises(RuntimeError, match="injected native"):
        solve_leg(
            synthetic_boundary(),
            ScvxSettings(
                discretisation_backend="cuda", assembly_backend="cuda", convex_solver_backend="qoco"
            ),
        )
    assert len(closed) == 1 and not closed[0]._finalizer.alive
