"""CPU reference planner: every family converges and certifies; warm start; honest failures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.planner import (
    FAMILIES,
    GPUUnavailableError,
    PlannerTranscription,
    PlanOptions,
    PlanResult,
    load_problem,
    load_result,
    normalise_problem,
    plan,
)
from spacepdhcg.planner.cli import main as cli_main
from spacepdhcg.planner.problem import apply_defaults

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "planner"
EXAMPLE_FILES = {
    "hcw": "hcw_rendezvous.json",
    "powered_descent_3dof": "powered_descent_3dof.json",
    "powered_descent_6dof": "powered_descent_6dof.json",
    "low_thrust": "low_thrust.json",
}

pytestmark = pytest.mark.usefixtures("planner_native_library")


def example(family: str) -> dict:
    return load_problem(EXAMPLES / EXAMPLE_FILES[family])


def _short(document: dict, intervals: int) -> dict:
    result = copy.deepcopy(document)
    result["horizon"]["intervals"] = intervals
    return result


@pytest.fixture(scope="module")
def cpu_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, PlanResult]:
    results: dict[str, PlanResult] = {}
    for family in FAMILIES:
        output = tmp_path_factory.mktemp(f"cpu-{family}")
        results[family] = plan(
            example(family),
            PlanOptions(
                backend="cpu_reference", output_directory=output, export_viewer=output / "viewer"
            ),
        )
    return results


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_plans_converge_and_certify(
    cpu_results: dict[str, PlanResult], family: str
) -> None:
    result = cpu_results[family]
    document = result.document
    assert result.backend_execution == "cpu_reference"
    assert result.requested_backend == "cpu_reference"
    assert result.converged, document["status"]
    assert result.certified, document["certificate"]["failed_gates"]
    assert result.status == "certified"
    assert result.exit_code == 0
    tolerance = document["certificate"]["tolerance"]
    replay = result.independent_replay
    assert replay["terminal_residual"] <= tolerance
    assert replay["path_violation"] <= tolerance
    assert replay["dynamics_defect"] <= tolerance
    assert replay["replay_parity"] <= document["certificate"]["replay_parity_tolerance"]
    assert document["solver_residuals"]["canonical_residual"] <= tolerance
    assert result.accepted_steps >= 1
    # Node histories have the documented shapes and are finite.
    info = result.family_info
    assert result.states.shape == (
        document["problem"]["horizon"]["intervals"] + 1,
        info.state_dimension,
    )
    assert result.controls.shape == (
        document["problem"]["horizon"]["intervals"],
        info.control_dimension,
    )
    assert np.all(np.isfinite(result.states)) and np.all(np.isfinite(result.controls))
    assert result.replay_states.shape[0] == document["problem"]["horizon"]["intervals"] * 10 + 1
    # Timings, telemetry and disposition blocks are present.
    assert document["timings"]["plan_wall_seconds"] > 0.0
    assert len(result.iterations) == result.outer_iterations
    assert document["backend"]["hidden_cpu_fallback"] is False
    assert document["backend"]["inner_solver"] == "clarabel"


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_outputs_are_written(cpu_results: dict[str, PlanResult], family: str) -> None:
    result = cpu_results[family]
    directory = result.output_directory
    assert directory is not None
    for name in (
        "plan-result.json",
        "plan-summary.md",
        "states.csv",
        "controls.csv",
        "replay.csv",
        "iterations.csv",
        "problem.json",
    ):
        assert (directory / name).is_file(), name
    reloaded = load_result(directory / "plan-result.json")
    assert reloaded.certified == result.certified
    assert reloaded.objective == pytest.approx(result.objective)
    summary = (directory / "plan-summary.md").read_text(encoding="utf-8")
    assert "Certified: **yes**" in summary
    assert family in summary
    header = (directory / "states.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "time," + ",".join(result.family_info.state_names)
    # The canonical problem echo is itself a valid problem document.
    canonical = json.loads((directory / "problem.json").read_text(encoding="utf-8"))
    assert normalise_problem(canonical)["family"] == family


def test_warm_start_from_a_previous_result_reconverges_quickly(
    cpu_results: dict[str, PlanResult],
) -> None:
    previous = cpu_results["powered_descent_3dof"]
    result = plan(
        example("powered_descent_3dof"), PlanOptions(backend="cpu_reference", warm_start=previous)
    )
    assert result.certified, result.failed_gates
    assert result.document["problem"]["warm_start_supplied"] is True
    assert result.outer_iterations <= previous.outer_iterations
    assert result.objective == pytest.approx(previous.objective, rel=1e-3)
    # The warm start is re-propagated from the initial state, so node states remain consistent.
    assert np.max(np.abs(result.states[0] - previous.states[0])) == 0.0


def test_warm_start_document_shape_is_validated() -> None:
    document = example("hcw")
    document["warm_start"] = {"states": [[0.0] * 6] * 3, "controls": [[0.0] * 3] * 2}
    from spacepdhcg.planner import ProblemValidationError

    with pytest.raises(ProblemValidationError, match="warm_start"):
        normalise_problem(document)


def test_infeasible_target_is_reported_honestly() -> None:
    document = example("powered_descent_3dof")
    document["terminal"]["state"] = [5000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    document["solver"]["maximum_outer_iterations"] = 6
    result = plan(document, PlanOptions(backend="cpu_reference"))
    assert not result.certified
    assert result.status in {
        "trust_region_exhausted",
        "maximum_iterations",
        "converged_not_certified",
        "not_certified",
        "solver_failure",
    }
    assert result.exit_code in {2, 3}
    if result.status == "solver_failure":
        assert "inner solver failure" in result.message
    else:
        assert (
            "converged" in result.failed_gates
            or "independent_terminal_residual" in result.failed_gates
        )
    assert result.independent_replay["terminal_position_error"] > 1.0
    # The retained trajectory is still the last physically consistent reference.
    assert np.all(np.isfinite(result.states))


def test_time_limit_is_reported_as_time_limit() -> None:
    document = _short(example("low_thrust"), 20)
    document["solver"]["maximum_outer_iterations"] = 50
    document["solver"]["tolerance"] = 1.0e-12
    document["solver"]["step_tolerance"] = 1.0e-12
    result = plan(document, PlanOptions(backend="cpu_reference", time_limit_seconds=1.0e-3))
    assert result.status == "time_limit"
    assert result.document["status"]["time_limit_triggered"] is True
    assert not result.certified


def test_gpu_backend_without_native_executable_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SPACEPDHCG_PLAN_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(GPUUnavailableError, match="pure_qoco"):
        plan(_short(example("hcw"), 8), PlanOptions(backend="pure_qoco"))


def test_gpu_backend_fallback_is_labelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SPACEPDHCG_PLAN_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = plan(
        _short(example("hcw"), 8), PlanOptions(backend="pdhcg", allow_cpu_reference_fallback=True)
    )
    assert result.backend_execution == "cpu_reference"
    assert result.document["backend"]["fallback_from"] == "pdhcg"
    assert "CPU reference fallback" in result.message
    assert result.certified


def test_backend_override_drops_incompatible_presets() -> None:
    document = example("hcw")
    document["solver"]["preset"] = "frozen_adaptive_pure_qoco"
    result = plan(_short(document, 8), PlanOptions(backend="cpu_reference"))
    assert result.requested_backend == "cpu_reference"


def test_transcription_dimensions_match_the_frozen_layouts() -> None:
    canonical = apply_defaults(normalise_problem(_short(example("powered_descent_3dof"), 4)))
    with PlannerTranscription(canonical) as transcription:
        dims = transcription.dimensions
        assert (dims.state_dimension, dims.control_dimension, dims.intervals) == (7, 4, 4)
        assert dims.variables == 5 * 7 + 4 * 4 + 2 * 4 * 7
        assert dims.scalar_rows == 7 + 4 * 7 + 6 + 2 * 4 * 7 + 4
        assert dims.affine_rows == 4 * 4 + 3 * 5 + 12 * 4 + 8
        assert transcription.structure.n_variables == dims.variables
        states, controls = transcription.initial_reference()
        replay = transcription.rollout(states[0], controls, 1)
        assert np.max(np.abs(replay - states)) == 0.0
        dense = transcription.rollout(states[0], controls, 5)
        assert dense.shape == (4 * 5 + 1, 7)
        evaluation = transcription.evaluate(states, controls)
        assert set(evaluation.path_components) == {
            "thrust_epigraph",
            "throttle_lower",
            "throttle_upper",
            "tilt",
            "minimum_mass",
            "altitude",
            "glide_slope",
        }
        values = transcription.values(states, controls, 1.0)
        assert values.quadratic.size == dims.variables
        assert values.lower.size == dims.scalar_rows
        assert values.affine_offset.size == dims.affine_rows


def test_cli_plan_and_validate_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    problem = EXAMPLES / EXAMPLE_FILES["hcw"]
    assert cli_main(["validate", str(problem), "--quiet"]) == 0
    assert capsys.readouterr().out.strip() == "valid"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0.0", "family": "hcw"}), encoding="utf-8")
    assert cli_main(["validate", str(bad)]) == 64
    output = tmp_path / "out"
    code = cli_main(
        ["plan", str(problem), "--backend", "cpu_reference", "--output", str(output), "--no-csv"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "certified:   True" in printed
    assert (output / "plan-result.json").is_file()
    assert not (output / "states.csv").exists()
    assert cli_main(["summary", str(output / "plan-result.json")]) == 0
    assert "# Plan summary" in capsys.readouterr().out
    assert cli_main(["capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu_reference_available"] is True
