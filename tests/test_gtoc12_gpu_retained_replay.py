"""A reused solver must refresh its first QP while retaining device replay."""

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
def test_retained_first_solve_refreshes_and_replays(monkeypatch, capfd, origin, hold):
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
    config = settings(max_iterations=30, time_limit_s=30, hold=hold)
    policy = SimpleNamespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN", str(origin))
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH", "0")
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE", "1")
    with using_gpu_execution(policy):
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "0")
        references = [certify_leg(solve_leg(b, config)) for b in (first, changed)]
        assert all(c.within_tolerance for c in references)
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", "1")
        for i, boundary in enumerate((first, changed, first)):
            capfd.readouterr()
            solution = solve_leg(boundary, config)
            trace = capfd.readouterr().err
            certificate = certify_leg(solution)
            assert solution.converged and certificate.within_tolerance
            assert abs(certificate.final_mass_kg - references[i % 2].final_mass_kg) < 1e-5
            report = solution.solver_reports[0]
            assert report["solves"] == 1 and report["qualified"]
            assert report["workspace_creations"] == int(i == 0)
            if i:
                assert trace.count("RETAINED_REPLAY submitted=0 updated=1") == 1
                assert trace.count("RETAINED_REPLAY ") == 1
            else:
                assert "RETAINED_REPLAY " not in trace
