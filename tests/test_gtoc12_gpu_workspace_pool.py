"""Cross-leg reuse must refresh physics inputs and discard failed workspaces."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_gtoc12_gpu_discretisation import synthetic_boundary
from test_gtoc12_gpu_scvx import GPU, settings

from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.low_thrust import certify_leg, solve_leg


@GPU
@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("ruiz", [0, 5])
@pytest.mark.parametrize("hold", ["zoh", "lagrange"])
def test_pool_refreshes_inputs_and_discards_timeout(monkeypatch, origin, ruiz, hold):
    first = synthetic_boundary()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    changed = replace(
        first,
        departure_position=rotation @ first.departure_position,
        departure_velocity=rotation @ first.departure_velocity,
        arrival_position=rotation @ first.arrival_position,
        arrival_velocity=rotation @ first.arrival_velocity,
        arrival_epoch=first.arrival_epoch - 0.5,
        initial_mass=first.initial_mass - 50.0,
    )
    config = settings(max_iterations=30, time_limit_s=30, qoco_ruiz_iterations=ruiz, hold=hold)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(origin))
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH", "0")
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    with using_gpu_execution(policy):
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
        references = [certify_leg(solve_leg(b, config)) for b in (first, changed)]
        assert all(c.within_tolerance for c in references)
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "1")
        for i, boundary in enumerate((first, changed, first)):
            solution = solve_leg(boundary, config)
            certificate = certify_leg(solution)
            assert solution.converged and certificate.within_tolerance
            assert abs(certificate.final_mass_kg - references[i % 2].final_mass_kg) < 1e-5
            assert solution.solver_reports[-1]["workspace_creations"] == (
                0 if i and ruiz == 0 else 1
            )
            assert solution.iterations == len(solution.solver_reports) == len(solution.history)
            assert solution.accepted_iterations == sum(r["accepted"] for r in solution.history)
        failed = solve_leg(first, replace(config, time_limit_s=0))
        assert failed.status == "timeout" and failed.iterations == 0
        recovered = solve_leg(first, config)
        assert recovered.converged and certify_leg(recovered).within_tolerance
        assert recovered.solver_reports[-1]["workspace_creations"] == 1
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
        fresh = solve_leg(first, config)
        assert fresh.converged and certify_leg(fresh).within_tolerance
        assert fresh.solver_reports[-1]["workspace_creations"] == 1


@GPU
def test_default_graph_execution_reuses_solver(monkeypatch):
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    config = settings(max_iterations=30, time_limit_s=30)
    with using_gpu_execution(policy):
        # Clear any prior cached test workspace through the supported opt-out.
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
        assert solve_leg(synthetic_boundary(), config).converged
        monkeypatch.delenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL")
        first = solve_leg(synthetic_boundary(), config)
        second = solve_leg(synthetic_boundary(), config)
        for solution in (first, second):
            assert solution.converged and certify_leg(solution).within_tolerance
            assert abs(solution.final_mass_kg - 2445.3111007852112) < 1e-5
        assert first.solver_reports[-1]["workspace_creations"] == 1
        assert second.solver_reports[-1]["workspace_creations"] == 0
