"""Native GPU planner correctness (short horizons) against the CPU reference.

These tests execute the real ``spacepdhcg_plan`` CUDA executable.  They are skipped unless
``SPACEPDHCG_PLAN_EXECUTABLE`` names the executable *and* ``SPACEPDHCG_PLANNER_GPU_TESTS=1``
is set, so GPU use stays deliberately serialized with other device work.  The pure-QOCO
backend additionally needs ``SPACEPDHCG_QOCO_LIBRARY``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.planner import FAMILIES, PlanOptions, PlanResult, load_problem, plan
from spacepdhcg.planner.native_runner import capabilities, find_executable

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "planner"
EXAMPLE_FILES = {
    "hcw": "hcw_rendezvous.json",
    "powered_descent_3dof": "powered_descent_3dof.json",
    "powered_descent_6dof": "powered_descent_6dof.json",
    "low_thrust": "low_thrust.json",
}
SHORT_INTERVALS = {
    "hcw": 20,
    "powered_descent_3dof": 20,
    "powered_descent_6dof": 10,
    "low_thrust": 20,
}

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SPACEPDHCG_PLANNER_GPU_TESTS") != "1"
        or "SPACEPDHCG_PLAN_EXECUTABLE" not in os.environ,
        reason=(
            "GPU planner tests require SPACEPDHCG_PLANNER_GPU_TESTS=1 and "
            "SPACEPDHCG_PLAN_EXECUTABLE"
        ),
    ),
    pytest.mark.usefixtures("planner_native_library"),
]


def _executable() -> Path:
    executable = find_executable()
    assert executable is not None
    info = capabilities(executable)
    if int(info.get("cuda_device_count", 0)) <= 0:
        pytest.skip("no CUDA device visible to spacepdhcg_plan")
    return executable


def _short(family: str, backend: str = "pure_qoco") -> dict:
    document = load_problem(EXAMPLES / EXAMPLE_FILES[family])
    document["horizon"]["intervals"] = SHORT_INTERVALS[family]
    document["solver"]["backend"] = backend
    document["solver"].pop("preset", None)
    return document


def _require_qoco() -> None:
    if not os.environ.get("SPACEPDHCG_QOCO_LIBRARY"):
        pytest.skip("SPACEPDHCG_QOCO_LIBRARY is required for the pure_qoco backend")


@pytest.fixture(scope="module")
def gpu_results(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[PlanResult, PlanResult]]:
    _require_qoco()
    executable = _executable()
    results: dict[str, tuple[PlanResult, PlanResult]] = {}
    for family in FAMILIES:
        document = _short(family)
        output = tmp_path_factory.mktemp(f"gpu-{family}")
        gpu = plan(
            document,
            PlanOptions(
                executable=executable, output_directory=output / "gpu", time_limit_seconds=600
            ),
        )
        cpu = plan(document, PlanOptions(backend="cpu_reference", output_directory=output / "cpu"))
        results[family] = (gpu, cpu)
    return results


@pytest.mark.parametrize("family", FAMILIES)
def test_native_plans_converge_certify_and_match_cpu_reference(
    gpu_results: dict[str, tuple[PlanResult, PlanResult]], family: str
) -> None:
    gpu, cpu = gpu_results[family]
    assert gpu.backend_execution == "native_cuda", gpu.document["backend"]
    assert gpu.document["backend"]["hidden_cpu_fallback"] is False
    assert gpu.converged, gpu.document["status"]
    assert gpu.certified, gpu.failed_gates
    assert cpu.certified, cpu.failed_gates
    tolerance = gpu.document["certificate"]["tolerance"]
    for result in (gpu, cpu):
        assert result.independent_replay["terminal_residual"] <= tolerance
        assert result.independent_replay["path_violation"] <= tolerance
        assert result.independent_replay["dynamics_defect"] <= tolerance
    assert (
        gpu.independent_replay["replay_parity"]
        <= gpu.document["certificate"]["replay_parity_tolerance"]
    )
    assert gpu.document["solver_residuals"]["coefficient_parity_relative"] <= 5.0e-12
    # Both solvers must land on the same optimum: objective and trust-scaled node states.
    assert gpu.objective == pytest.approx(cpu.objective, rel=1.0e-4, abs=1.0e-6)
    scales = np.asarray(
        gpu.document["problem"]["transcription"].get(
            "state_trust_scales", np.ones(gpu.states.shape[1])
        )
    )
    scaled_difference = np.max(np.abs((gpu.states - cpu.states) * scales))
    assert scaled_difference <= 2.0e-3, scaled_difference
    assert gpu.document["timings"]["cuda_startup_seconds"] >= 0.0
    assert gpu.document["timings"]["scvx_total_seconds"] > 0.0
    assert gpu.document["backend"]["qoco_workspace_creations"] == 1
    assert gpu.document["backend"]["topology_allocation_count_after_create"] == 0


def test_native_warm_start_reuses_a_previous_plan(
    gpu_results: dict[str, tuple[PlanResult, PlanResult]],
) -> None:
    executable = _executable()
    previous, _ = gpu_results["powered_descent_3dof"]
    result = plan(
        _short("powered_descent_3dof"),
        PlanOptions(executable=executable, warm_start=previous, time_limit_seconds=600),
    )
    assert result.certified, result.failed_gates
    assert result.document["problem"]["warm_start_supplied"] is True
    assert result.outer_iterations <= previous.outer_iterations
    assert result.objective == pytest.approx(previous.objective, rel=1.0e-4, abs=1.0e-6)


def test_native_pdhcg_backend_selection_is_recorded() -> None:
    executable = _executable()
    result = plan(
        _short("hcw", backend="pdhcg"), PlanOptions(executable=executable, time_limit_seconds=300)
    )
    assert result.backend_execution == "native_cuda"
    assert result.requested_backend == "pdhcg"
    assert result.document["backend"]["device_policy"] == "adaptive_pdhcg"
    assert result.document["backend"]["qoco_workspace_creations"] == 0
    assert result.certified, result.failed_gates


def test_native_infeasible_target_is_reported_honestly(tmp_path: Path) -> None:
    _require_qoco()
    executable = _executable()
    document = _short("powered_descent_3dof")
    document["terminal"]["state"] = [5000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    document["solver"]["maximum_outer_iterations"] = 6
    result = plan(
        document,
        PlanOptions(executable=executable, output_directory=tmp_path, time_limit_seconds=600),
    )
    assert not result.certified
    assert result.status in {
        "trust_region_exhausted",
        "maximum_iterations",
        "solver_failure",
        "converged_not_certified",
    }
    assert result.exit_code in {2, 3}
    assert result.document["backend"]["native_exit_code"] in {2, 3}
    written = json.loads((tmp_path / "plan-result.json").read_text(encoding="utf-8"))
    assert written["certificate"]["certified"] is False


def test_native_time_limit_is_reported(tmp_path: Path) -> None:
    executable = _executable()
    document = _short("low_thrust", backend="pdhcg_recovery")
    document["solver"]["maximum_outer_iterations"] = 50
    document["solver"]["tolerance"] = 1.0e-12
    document["solver"]["step_tolerance"] = 1.0e-12
    result = plan(
        document,
        PlanOptions(executable=executable, output_directory=tmp_path, time_limit_seconds=2.0),
    )
    assert result.status in {
        "time_limit",
        "converged_not_certified",
        "maximum_iterations",
        "trust_region_exhausted",
    }
    if result.status == "time_limit":
        assert result.document["status"]["time_limit_triggered"] is True
    assert not result.certified


def test_native_rejects_cpu_reference_backend_documents(tmp_path: Path) -> None:
    import subprocess

    executable = _executable()
    request = copy.deepcopy(_short("hcw", backend="cpu_reference"))
    from spacepdhcg.planner.problem import apply_defaults, dump_canonical, normalise_problem

    path = tmp_path / "request.json"
    path.write_text(dump_canonical(apply_defaults(normalise_problem(request))), encoding="utf-8")
    completed = subprocess.run(
        [str(executable), str(path), "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 64
    payload = json.loads(completed.stdout)
    assert payload["status"]["code"] == "invalid_problem"
    assert payload["certificate"]["certified"] is False
