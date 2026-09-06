"""Opt-in, serialized native GPU tests for work accounting on failure exits."""

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


@pytest.fixture(scope="module")
def fault_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    proxy = tmp_path_factory.mktemp("failure-probe") / "fault.so"
    subprocess.run(
        ["g++", "-std=c++17", "-shared", "-fPIC", "-I" + str(ROOT / "cpp/cuda/include"),
         "-I" + str(ROOT / "cpp/include"), "-I/usr/local/cuda-12.8/include",
         str(ROOT / "cpp/cuda/tests/scvx_qoco_failure_timing_probe.cpp"),
         "-ldl", "-o", str(proxy)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return proxy


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_completed_inner_work_is_accounted_on_every_exit(
    tmp_path: Path, fault_probe: Path, fail_at: int
) -> None:
    output = tmp_path / "result.json"
    env = dict(os.environ, LD_PRELOAD=str(fault_probe),
               SPACEPDHCG_TEST_FAIL_QOCO_ORDINAL=str(fail_at))
    run = subprocess.run(
        [os.environ["SPACEPDHCG_PLAN_EXECUTABLE"],
         str(ROOT / "artifacts/performance/qoco-gpu-tree-input-20.json"),
         "--quiet", "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=180, check=False,
    )
    records = [json.loads(line) for line in run.stderr.splitlines()
               if line.startswith('{"case":"qoco_after_solve_fault"')]
    assert records and all(r["actual_status"] == 0 for r in records), run.stderr
    result = json.loads(output.read_text())
    if fail_at:
        assert run.returncode == 3 and records[-1]["injected"]
        assert len(records) == fail_at
        assert result["status"]["solver_status"] == "inner_failure"
        assert not result["certificate"]["certified"]
        # The first valid inner candidate may be rejected by nonlinear SCvx;
        # it still consumed real work and must not be counted twice on failure.
        assert result["summary"]["accepted_steps"] <= fail_at - 1
        assert result["summary"]["accepted_steps"] == sum(
            bool(row["accepted"]) for row in result["iterations"]
        )
    else:
        assert run.returncode == 0 and result["certificate"]["certified"]
        assert result["summary"]["objective"] == pytest.approx(
            0.51297569119164033, rel=0, abs=1e-8
        )
    assert result["summary"]["inner_iterations"] == sum(r["iterations"] for r in records)
    timings = result["timings"]
    assert timings["solve_seconds"] == timings["qoco_solve_seconds"] > 0
    for name in ("solve_seconds", "update_seconds", "residual_seconds"):
        assert timings[name] == records[-1][name]
    components = ("update_seconds", "scaling_seconds", "solve_seconds", "residual_seconds",
                  "qoco_conversion_seconds", "qoco_setup_seconds")
    assert timings["cqp_total_seconds"] == pytest.approx(sum(timings[k] for k in components))
    assert 0 < timings["cqp_total_seconds"] <= timings["scvx_total_seconds"]
    assert timings["scvx_total_seconds"] <= timings["solve_wall_seconds"] + 1e-5
