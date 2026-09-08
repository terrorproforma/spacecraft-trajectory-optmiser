"""Independent end-to-end qualification for the retained CUDA outer loop."""

import os
from dataclasses import replace

import numpy as np
import pytest
from test_gtoc12_gpu_discretisation import synthetic_boundary

from spacepdhcg.gtoc12.low_thrust import ScvxSettings, certify_leg, solve_leg

GPU = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires serialized GPU native runtime",
)


def settings(**kwargs):
    return ScvxSettings(
        discretisation_backend="cuda",
        assembly_backend="cuda",
        convex_solver_backend="qoco",
        outer_loop_backend="cuda",
        **kwargs,
    )


def test_invalid_outer_backend_fails_before_seed():
    with pytest.raises(ValueError, match="outer_loop_backend"):
        solve_leg(synthetic_boundary(), ScvxSettings(outer_loop_backend="bogus"))
    with pytest.raises(ValueError, match="CUDA outer loop requires"):
        solve_leg(synthetic_boundary(), ScvxSettings(outer_loop_backend="cuda"))


@GPU
def test_native_solution_passes_batched_independent_cuda_certificate():
    from spacepdhcg.gtoc12.gpu_verifier import certify_legs_cuda

    solution = solve_leg(synthetic_boundary(), settings(max_iterations=30, time_limit_s=30))
    assert solution.converged
    reference = certify_leg(solution)
    certificates = certify_legs_cuda([solution] * 17)
    for certificate in certificates:
        assert certificate.within_tolerance
        assert abs(certificate.final_mass_kg-reference.final_mass_kg) < 1e-7
        assert abs(certificate.position_error_km-reference.position_error_km) < 0.001


@GPU
@pytest.mark.parametrize("device_scheduling", [False, True])
@pytest.mark.parametrize("deferred_reports", [False, True])
@pytest.mark.parametrize("state_origin", [False, True])
def test_native_transfer_has_no_host_trajectory_iterations(
    monkeypatch, device_scheduling, deferred_reports, state_origin
):
    from spacepdhcg.gtoc12 import low_thrust
    from spacepdhcg.gtoc12.gpu_discretisation import GpuDiscretisation
    from spacepdhcg.gtoc12.gpu_qoco import GpuQocoProblem

    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_DEVICE_SCHEDULING", str(int(device_scheduling)))
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_DEFERRED_REPORTS", str(int(deferred_reports)))
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(int(state_origin)))

    def forbidden(*args, **kwargs):
        raise AssertionError("host numerical iteration must not run")

    monkeypatch.setattr(GpuQocoProblem, "solve_linearised", forbidden)
    monkeypatch.setattr(GpuDiscretisation, "propagate", forbidden)
    monkeypatch.setattr(low_thrust, "_ballistic_reference", forbidden)
    sol = solve_leg(synthetic_boundary(), settings(max_iterations=30, time_limit_s=30))
    certificate = certify_leg(sol)
    assert sol.converged and certificate.within_tolerance, (sol.status, certificate)
    assert abs(sol.final_mass_kg - 2445.3111007852112) <= 1e-5
    assert sol.accepted_iterations == sum(r["accepted"] for r in sol.history)
    assert sol.iterations == len(sol.history) == len(sol.solver_reports)
    assert sol.outer_transfer_bytes["trajectory_upload_bytes"] == 0
    assert sol.seed_backend == "cuda"
    assert sol.outer_transfer_bytes["trajectory_download_bytes"] == len(sol.states_scaled) * 11 * 8
    assert sol.outer_transfer_bytes["control_download_bytes"] <= (sol.iterations + 2) * 16
    if device_scheduling or deferred_reports:
        assert sol.outer_transfer_bytes["control_download_bytes"] == (sol.iterations + 1) * 8


@GPU
@pytest.mark.parametrize("mode", ["auto", "graph", "dispatch"])
def test_cli_execution_policy_reaches_native_loop(monkeypatch, mode):
    from spacepdhcg.cli import build_parser
    from spacepdhcg.gtoc12.cli import _scvx_settings, _with_screening_backend

    # Contradict inherited switches: the public option must own the execution
    # policy, including disabling an inherited graph for a dispatch comparison.
    inherited = "1" if mode == "dispatch" else "0"
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_OUTER_GRAPH", inherited)
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH", inherited)
    args = build_parser().parse_args([
        "gtoc12", "run", "--run-id", "execution-test", "--output", "unused",
        "--discretisation-backend", "cuda", "--assembly-backend", "cuda",
        "--convex-solver", "qoco", "--outer-loop-backend", "cuda",
        "--gpu-execution", mode,
    ])

    @_with_screening_backend
    def command(parsed):
        return solve_leg(
            synthetic_boundary(),
            replace(_scvx_settings(parsed), max_iterations=30, time_limit_s=30),
        )

    solution = command(args)
    assert solution.converged and certify_leg(solution).within_tolerance
    assert abs(solution.final_mass_kg - 2445.3111007852112) <= 1e-5
    graph_rows = [r for r in solution.solver_reports if r["solve_seconds"] is None]
    assert bool(graph_rows) == (mode != "dispatch")
    if graph_rows:
        priming = len(solution.solver_reports) - len(graph_rows)
        assert solution.outer_transfer_bytes["control_download_bytes"] == 8 * (priming + 1) + 16
    assert os.environ["SPACEPDHCG_TEST_GTOC12_OUTER_GRAPH"] == inherited
    assert os.environ["SPACEPDHCG_TEST_QOCO_IPM_GRAPH"] == inherited


@GPU
def test_native_zero_time_preserves_seed_and_never_claims_convergence():
    sol = solve_leg(synthetic_boundary(), settings(time_limit_s=0))
    assert sol.status == "timeout" and sol.iterations == sol.accepted_iterations == 0
    assert not sol.history and not sol.solver_reports
    assert np.isfinite(sol.states_scaled).all() and sol.virtual_inf == float("inf")


@GPU
@pytest.mark.parametrize("state_origin", [False, True])
@pytest.mark.parametrize("ruiz", [0, 5])
@pytest.mark.parametrize("early_graph", [False, True])
def test_gpu_outer_graph_retains_physics_and_objective(
    monkeypatch, state_origin, ruiz, early_graph
):
    from spacepdhcg.gtoc12 import low_thrust
    from spacepdhcg.gtoc12.gpu_discretisation import GpuDiscretisation
    from spacepdhcg.gtoc12.gpu_qoco import GpuQocoProblem

    # This test measures cold priming; cross-leg reuse has a separate matrix.
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")

    for flag in (
        "GTOC12_OUTER_GRAPH",
        "QOCO_DEVICE_VALIDATION",
        "QOCO_NATIVE_NUMERIC_REPLAY",
        "QOCO_NATIVE_REPLAY",
        "QOCO_IPM_GRAPH",
    ):
        monkeypatch.setenv("SPACEPDHCG_TEST_" + flag, "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(int(state_origin)))
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH", str(int(early_graph)))

    def forbidden(*args, **kwargs):
        raise AssertionError("host numerical iteration must not run")

    monkeypatch.setattr(GpuQocoProblem, "solve_linearised", forbidden)
    monkeypatch.setattr(GpuDiscretisation, "propagate", forbidden)
    monkeypatch.setattr(low_thrust, "_ballistic_reference", forbidden)
    sol = solve_leg(
        synthetic_boundary(),
        settings(max_iterations=30, time_limit_s=30, qoco_ruiz_iterations=ruiz),
    )
    certificate = certify_leg(sol)
    assert sol.converged and certificate.within_tolerance, (sol.status, certificate)
    assert abs(sol.final_mass_kg - 2445.3111007852112) <= 1e-5
    assert sol.iterations == len(sol.history) == len(sol.solver_reports)
    assert sol.accepted_iterations == sum(row["accepted"] for row in sol.history)
    graph_rows = [row for row in sol.solver_reports if row["solve_seconds"] is None]
    assert graph_rows and all(row["update_seconds"] is None for row in graph_rows)
    # One initial command, one per host priming solve, then a single 16-byte
    # graph exit. A four-attempt solve can equal the old per-attempt byte count;
    # graph execution still performs no per-iteration command downloads.
    priming_count = len(sol.solver_reports) - len(graph_rows)
    assert priming_count in (2, 3)
    if ruiz == 0 and not state_origin:
        device_init = os.environ.get("SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION", "1") != "0"
        assert priming_count == (2 if early_graph or device_init else 3)
    assert sol.outer_transfer_bytes["control_download_bytes"] == 8 * (priming_count + 1) + 16
    assert sol.outer_transfer_bytes["trajectory_upload_bytes"] == 0
    assert graph_rows[-1]["workspace_creations"] == 1
    assert graph_rows[-1]["solves"] == sum(row["iterations"] > 0 for row in sol.solver_reports)


@GPU
def test_gpu_outer_graph_zero_budget_and_missing_extension_are_explicit(monkeypatch):
    for flag in (
        "GTOC12_OUTER_GRAPH",
        "QOCO_DEVICE_VALIDATION",
        "QOCO_NATIVE_NUMERIC_REPLAY",
        "QOCO_NATIVE_REPLAY",
        "QOCO_IPM_GRAPH",
    ):
        monkeypatch.setenv("SPACEPDHCG_TEST_" + flag, "1")
    sol = solve_leg(synthetic_boundary(), settings(time_limit_s=0))
    assert sol.status == "timeout" and sol.iterations == sol.accepted_iterations == 0
    assert not sol.history and not sol.solver_reports
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH", "0")
    with pytest.raises(RuntimeError, match="status 5"):
        solve_leg(synthetic_boundary(), settings())


@GPU
@pytest.mark.parametrize(
    "field,value",
    [("substeps", 0), ("minimum_trust", 0), ("shrink_factor", 1), ("virtual_weight", float("nan"))],
)
def test_native_invalid_settings_are_not_silently_replaced(field, value):
    with pytest.raises(RuntimeError, match="status 1"):
        solve_leg(synthetic_boundary(), replace(settings(), **{field: value}))
