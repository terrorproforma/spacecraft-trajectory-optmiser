"""Native regression for convergence after trust-region-limited SCvx steps.

Uses the same opt-in GPU environment as test_planner_gpu. Run serialized with
other GPU work, selecting the intended core through LD_LIBRARY_PATH.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_PLANNER_GPU_TESTS") != "1"
    or not os.environ.get("SPACEPDHCG_PLAN_EXECUTABLE")
    or not os.environ.get("SPACEPDHCG_QOCO_LIBRARY"),
    reason="requires explicitly enabled native planner GPU tests and QOCO",
)


def _solve(tmp_path: Path, maximum_iterations: int, step_tolerance: float) -> dict:
    problem = json.loads(
        (ROOT / "artifacts/performance/qoco-gpu-tree-input-20.json").read_text()
    )
    problem["solver"].setdefault("trust_region", {})["initial_radius"] = 0.015625
    problem["solver"]["maximum_outer_iterations"] = maximum_iterations
    problem["solver"]["step_tolerance"] = step_tolerance
    source, output = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(json.dumps(problem))
    run = subprocess.run(
        [os.environ["SPACEPDHCG_PLAN_EXECUTABLE"], str(source), "--quiet", "--output", str(output)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert output.exists(), (run.returncode, run.stdout, run.stderr)
    result = json.loads(output.read_text())
    assert result["backend"]["hidden_cpu_fallback"] is False
    accepted = [row for row in result["iterations"] if row["accepted"]]
    assert accepted, result["status"]
    last = accepted[-1]
    # Final replay must not replace the accepted displacement with zero.
    assert result["summary"]["trajectory_step"] > 0
    assert result["summary"]["trajectory_step"] == pytest.approx(
        last["step_fraction"] * last["trust_radius_before"], rel=1e-12, abs=1e-15
    )
    return result


def test_small_trust_region_needs_an_interior_step(tmp_path: Path) -> None:
    result = _solve(tmp_path, 30, 0.02)
    assert result["certificate"]["certified"], result["certificate"]
    assert result["summary"]["objective"] == pytest.approx(0.51297569119164033, rel=0, abs=1e-8)
    accepted = [row for row in result["iterations"] if row["accepted"]]
    assert any(row["step_fraction"] >= 0.8 for row in accepted[:-1])
    assert accepted[-1]["step_fraction"] < 0.8


@pytest.mark.parametrize("step_tolerance", [0.02, 1e-12])
def test_final_replay_preserves_iteration_limit(tmp_path: Path, step_tolerance: float) -> None:
    result = _solve(tmp_path, 2, step_tolerance)
    assert result["status"]["solver_status"] == "maximum_iterations"
    assert not result["certificate"]["certified"]


def test_feasible_initial_point_does_not_erase_cancellation(tmp_path: Path) -> None:
    proxy = tmp_path / "cancel.so"
    subprocess.run(
        ["g++", "-std=c++17", "-shared", "-fPIC", "-I" + str(ROOT / "cpp/cuda/include"),
         "-I" + str(ROOT / "cpp/include"),
         str(ROOT / "cpp/cuda/tests/scvx_cancel_before_solve_probe.cpp"),
         "-ldl", "-o", str(proxy)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    problem = json.loads((ROOT / "examples/planner/hcw_rendezvous.json").read_text())
    problem.pop("units")
    problem["initial_state"] = [0.0] * 6
    problem["horizon"]["intervals"] = 4
    source, output = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(json.dumps(problem))
    env = dict(os.environ, LD_PRELOAD=str(proxy))
    run = subprocess.run(
        [os.environ["SPACEPDHCG_PLAN_EXECUTABLE"], str(source), "--quiet", "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert "cancel_before_solve" in run.stderr, (run.returncode, run.stdout, run.stderr)
    result = json.loads(output.read_text())
    assert result["status"]["solver_status"] == "cancelled"
    assert result["summary"]["outer_iterations"] == 0
    assert not result["certificate"]["certified"]
