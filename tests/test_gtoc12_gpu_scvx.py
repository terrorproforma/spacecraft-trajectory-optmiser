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
@pytest.mark.parametrize("device_scheduling", [False, True])
@pytest.mark.parametrize("deferred_reports", [False, True])
def test_native_transfer_has_no_host_trajectory_iterations(monkeypatch, device_scheduling, deferred_reports):
    from spacepdhcg.gtoc12 import low_thrust
    from spacepdhcg.gtoc12.gpu_discretisation import GpuDiscretisation
    from spacepdhcg.gtoc12.gpu_qoco import GpuQocoProblem

    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_DEVICE_SCHEDULING", str(int(device_scheduling)))
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_DEFERRED_REPORTS", str(int(deferred_reports)))

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
def test_native_zero_time_preserves_seed_and_never_claims_convergence():
    sol = solve_leg(synthetic_boundary(), settings(time_limit_s=0))
    assert sol.status == "timeout" and sol.iterations == sol.accepted_iterations == 0
    assert not sol.history and not sol.solver_reports
    assert np.isfinite(sol.states_scaled).all() and sol.virtual_inf == float("inf")


@GPU
@pytest.mark.parametrize(
    "field,value",
    [("substeps", 0), ("minimum_trust", 0), ("shrink_factor", 1), ("virtual_weight", float("nan"))],
)
def test_native_invalid_settings_are_not_silently_replaced(field, value):
    with pytest.raises(RuntimeError, match="status 1"):
        solve_leg(synthetic_boundary(), replace(settings(), **{field: value}))
