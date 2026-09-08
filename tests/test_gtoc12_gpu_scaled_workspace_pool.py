"""Scaled reuse must refresh geometry, isolate scaling counts and discard failures."""

import os
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
@pytest.mark.parametrize("hold", ["zoh", "lagrange"])
@pytest.mark.parametrize("replay", [0, 1])
def test_scaled_pool_refreshes_and_discards_failed_leg(monkeypatch, origin, hold, replay):
    first = synthetic_boundary()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    changed = replace(
        first,
        departure_position=rotation @ first.departure_position,
        departure_velocity=rotation @ first.departure_velocity,
        arrival_position=rotation @ first.arrival_position,
        arrival_velocity=rotation @ first.arrival_velocity,
        arrival_epoch=first.arrival_epoch - 0.5,
        initial_mass=first.initial_mass - 50,
    )
    config = settings(max_iterations=30, time_limit_s=30, qoco_ruiz_iterations=2, hold=hold)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(origin))
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY", str(replay))
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH", "0")
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    with using_gpu_execution(policy):
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
        references = [certify_leg(solve_leg(b, config)) for b in (first, changed)]
        assert all(c.within_tolerance for c in references)
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "1")
        for i, boundary in enumerate((first, changed, first)):
            result = solve_leg(boundary, config)
            certificate = certify_leg(result)
            assert result.converged and certificate.within_tolerance
            assert abs(certificate.final_mass_kg - references[i % 2].final_mass_kg) < 1e-5
            assert result.solver_reports[-1]["workspace_creations"] == (0 if i else 1)
            assert result.iterations == len(result.solver_reports) == len(result.history)
            assert result.accepted_iterations == sum(row["accepted"] for row in result.history)
        failed = solve_leg(first, replace(config, time_limit_s=0))
        assert failed.status == "timeout" and failed.iterations == 0
        recovered = solve_leg(first, config)
        assert recovered.converged and certify_leg(recovered).within_tolerance
        assert recovered.solver_reports[-1]["workspace_creations"] == 1


@GPU
def test_scaled_pool_keys_ruiz_count(monkeypatch):
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL", "1")
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    with using_gpu_execution(policy):
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
        boundary = synthetic_boundary()
        assert solve_leg(boundary, settings(max_iterations=30, time_limit_s=30)).converged
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "1")
        for ruiz, created in [(2, 1), (5, 1), (2, 0), (0, 1), (5, 0)]:
            config = settings(max_iterations=30, time_limit_s=30, qoco_ruiz_iterations=ruiz)
            result = solve_leg(boundary, config)
            assert result.converged and certify_leg(result).within_tolerance
            assert result.solver_reports[-1]["workspace_creations"] == created


@GPU
def test_scaled_pool_requires_library_capability(monkeypatch):
    old_library = os.environ.get("SPACEPDHCG_GTOC12_OLD_QOCO_LIBRARY")
    if not old_library:
        pytest.skip("requires an objective-preserving library without the capability export")
    monkeypatch.setenv("SPACEPDHCG_QOCO_LIBRARY", old_library)
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "1")
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    with using_gpu_execution(policy):
        for _ in range(2):
            config = settings(max_iterations=30, time_limit_s=30, qoco_ruiz_iterations=2)
            result = solve_leg(synthetic_boundary(), config)
            assert result.converged and certify_leg(result).within_tolerance
            assert result.solver_reports[-1]["workspace_creations"] == 1
