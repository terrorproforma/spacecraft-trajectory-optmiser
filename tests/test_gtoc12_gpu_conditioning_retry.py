"""Captured return regression for device-selected cold-retry conditioning."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.low_thrust import LegBoundary, ScvxSettings, certify_leg, solve_leg
from spacepdhcg.gtoc12.pipeline import clamp_thrust

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="serialized CUDA runtime required"
)


def captured_return():
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads(
        (root / "results/local/2026-09-09/return-replay-v672/fixture.json").read_text()
    )
    shared = dict(fixture["boundary_shared"])
    for name in (
        "departure_position",
        "departure_velocity",
        "arrival_position",
        "arrival_velocity",
    ):
        shared[name] = np.array(shared[name], dtype=np.float64)
    mass = float.fromhex(fixture["variants"]["mesh_candidate1"]["initial_mass_hex"])
    return LegBoundary(**shared, initial_mass=mass), ScvxSettings(**fixture["settings"])


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("pool", [False, True])
def test_captured_return_converges_and_certifies_after_rebinding(monkeypatch, graph, pool):
    from argparse import Namespace

    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_QOCO_POOL", str(int(pool)))
    boundary, settings = captured_return()
    execution = Namespace(
        gpu_execution="graph" if graph else "dispatch", outer_loop_backend="cuda", workers=1
    )
    masses = []
    with using_gpu_execution(execution):
        for _ in range(2):
            solution = solve_leg(boundary, settings)
            assert solution.converged, (solution.status, solution.diagnostic)
            assert solution.iterations <= settings.max_iterations + settings.polish_iterations
            assert len(solution.solver_reports) == solution.iterations
            assert any(r["conditioning_retry"] for r in solution.solver_reports)
            assert all(
                r["ruiz_iterations"] == (5 if r["conditioning_retry"] else 0)
                for r in solution.solver_reports
            )
            assert all(
                r["qualified"]
                for r, h in zip(solution.solver_reports, solution.history, strict=True)
                if h.get("accepted")
            )
            clamp_thrust(solution)
            certificate = certify_leg(solution)
            assert certificate.within_tolerance
            assert certificate.final_mass_kg >= boundary.minimum_final_mass
            masses.append(certificate.final_mass_kg)
    assert abs(masses[0] - masses[1]) < 1e-5


def test_explicit_ruiz_policy_is_not_overridden(monkeypatch):
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY", "1")
    boundary, settings = captured_return()
    with pytest.raises(RuntimeError, match="CUDA SCvx failed"):
        solve_leg(boundary, replace(settings, qoco_ruiz_iterations=5))
