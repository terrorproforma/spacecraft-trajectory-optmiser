"""Plan results: typed access to the strict result document plus file writers."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.planner.problem import FAMILY_INFO, FamilyInfo

FloatArray = NDArray[np.float64]
RESULT_SCHEMA_VERSION = "1.0.0"
RESULT_KIND = "spacepdhcg_plan_result"


def json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats into strict JSON values."""

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    return value


@dataclass(slots=True)
class PlanResult:
    """One planner result (native CUDA or CPU reference) in the strict document format."""

    document: dict[str, Any]
    output_directory: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)

    # -- typed accessors ---------------------------------------------------

    @property
    def status(self) -> str:
        return str(self.document.get("status", {}).get("code", "unknown"))

    @property
    def message(self) -> str:
        return str(self.document.get("status", {}).get("message", ""))

    @property
    def solver_status(self) -> str:
        return str(self.document.get("status", {}).get("solver_status", "unknown"))

    @property
    def exit_code(self) -> int:
        return int(self.document.get("status", {}).get("exit_code", 70))

    @property
    def certified(self) -> bool:
        return bool(self.document.get("certificate", {}).get("certified", False))

    @property
    def failed_gates(self) -> list[str]:
        return list(self.document.get("certificate", {}).get("failed_gates", []))

    @property
    def converged(self) -> bool:
        return self.solver_status == "converged"

    @property
    def family(self) -> str:
        return str(self.document.get("problem", {}).get("family", "unknown"))

    @property
    def family_info(self) -> FamilyInfo:
        return FAMILY_INFO[self.family]

    @property
    def backend_execution(self) -> str:
        return str(self.document.get("backend", {}).get("execution", "unknown"))

    @property
    def requested_backend(self) -> str:
        return str(self.document.get("backend", {}).get("requested_backend", "unknown"))

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self.document.get("summary", {}))

    @property
    def objective(self) -> float:
        return float(self.summary.get("objective", float("nan")))

    @property
    def outer_iterations(self) -> int:
        return int(self.summary.get("outer_iterations", 0))

    @property
    def accepted_steps(self) -> int:
        return int(self.summary.get("accepted_steps", 0))

    @property
    def independent_replay(self) -> dict[str, Any]:
        return dict(self.document.get("independent_replay", {}))

    @property
    def terminal_residual(self) -> float:
        value = self.independent_replay.get("terminal_residual")
        return float("nan") if value is None else float(value)

    @property
    def terminal_position_error(self) -> float:
        value = self.independent_replay.get("terminal_position_error")
        return float("nan") if value is None else float(value)

    @property
    def wall_seconds(self) -> float:
        return float(self.document.get("timings", {}).get("plan_wall_seconds", float("nan")))

    @property
    def times(self) -> FloatArray:
        return np.asarray(self.document.get("trajectory", {}).get("times", []), dtype=np.float64)

    @property
    def states(self) -> FloatArray:
        rows = self.document.get("trajectory", {}).get("states", [])
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)

    @property
    def controls(self) -> FloatArray:
        rows = self.document.get("trajectory", {}).get("controls", [])
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)

    @property
    def replay_times(self) -> FloatArray:
        return np.asarray(self.document.get("dense_replay", {}).get("times", []), dtype=np.float64)

    @property
    def replay_states(self) -> FloatArray:
        rows = self.document.get("dense_replay", {}).get("states", [])
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)

    @property
    def iterations(self) -> list[dict[str, Any]]:
        return list(self.document.get("iterations", []))

    def warm_start_document(self) -> dict[str, Any]:
        """Node histories in the ``warm_start`` shape accepted by problem documents."""

        return {
            "source": f"PlanResult {self.family} {self.status}",
            "states": json_safe(self.states),
            "controls": json_safe(self.controls),
        }

    # -- writers -------------------------------------------------------------

    def to_json(self, *, indent: int | None = 2) -> str:
        return (
            json.dumps(json_safe(self.document), indent=indent, sort_keys=False, allow_nan=False)
            + "\n"
        )

    def write(
        self, directory: str | Path, *, write_csv: bool = True, write_summary: bool = True
    ) -> dict[str, Path]:
        """Write ``plan-result.json`` (+ CSV tables + ``plan-summary.md``) into ``directory``."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}
        result_path = target / "plan-result.json"
        result_path.write_text(self.to_json(), encoding="utf-8")
        files["result"] = result_path
        if write_csv and "trajectory" in self.document:
            files.update(self._write_csv(target))
        if write_summary:
            summary_path = target / "plan-summary.md"
            summary_path.write_text(self.summary_markdown(), encoding="utf-8")
            files["summary"] = summary_path
        self.output_directory = target
        self.files.update(files)
        return files

    def _write_csv(self, target: Path) -> dict[str, Path]:
        info = self.family_info
        files: dict[str, Path] = {}
        states_path = target / "states.csv"
        with states_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", *info.state_names])
            for time, row in zip(self.times, self.states, strict=True):
                writer.writerow([repr(float(time)), *[repr(float(value)) for value in row]])
        files["states_csv"] = states_path
        controls_path = target / "controls.csv"
        with controls_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", *info.control_names])
            for time, row in zip(self.times[:-1], self.controls, strict=True):
                writer.writerow([repr(float(time)), *[repr(float(value)) for value in row]])
        files["controls_csv"] = controls_path
        if "dense_replay" in self.document:
            replay_path = target / "replay.csv"
            with replay_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", *info.state_names])
                for time, row in zip(self.replay_times, self.replay_states, strict=True):
                    writer.writerow([repr(float(time)), *[repr(float(value)) for value in row]])
            files["replay_csv"] = replay_path
        iterations = self.iterations
        if iterations:
            iterations_path = target / "iterations.csv"
            keys = list(iterations[0].keys())
            with iterations_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(keys)
                for record in iterations:
                    writer.writerow([json_safe(record.get(key)) for key in keys])
            files["iterations_csv"] = iterations_path
        return files

    def summary_markdown(self) -> str:
        """Human-readable summary of the plan and its certificate."""

        document = self.document
        problem = document.get("problem", {})
        summary = self.summary
        independent = self.independent_replay
        certificate = document.get("certificate", {})
        residuals = document.get("solver_residuals", {})
        timings = document.get("timings", {})
        backend = document.get("backend", {})
        info = FAMILY_INFO.get(self.family)
        units = problem.get("units", {})

        def fmt(value: Any, digits: int = 6) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, int):
                return str(value)
            if isinstance(value, float):
                if not math.isfinite(value):
                    return "n/a"
                return f"{value:.{digits}g}"
            return str(value)

        lines = [
            f"# Plan summary: {problem.get('name') or self.family}",
            "",
            f"- Family: `{self.family}`"
            + (f" ({info.physical_family}; {info.frame})" if info is not None else ""),
            f"- Status: **{self.status}** — {self.message}",
            f"- Certified: **{fmt(self.certified)}**"
            + (f" (failed gates: {', '.join(self.failed_gates)})" if self.failed_gates else ""),
            f"- Execution: `{self.backend_execution}` via backend `{self.requested_backend}`"
            f" (policy `{backend.get('device_policy', backend.get('policy', 'n/a'))}`)",
            "",
            "## Problem",
            "",
            f"- Intervals: {problem.get('horizon', {}).get('intervals')}, final time "
            f"{fmt(problem.get('horizon', {}).get('final_time'))} {units.get('time', 's')}, "
            f"step {fmt(problem.get('horizon', {}).get('step_seconds'))} s",
            f"- Initial state: {json.dumps(json_safe(problem.get('initial_state')))}",
            f"- Terminal target: {json.dumps(json_safe(problem.get('terminal', {}).get('state')))}",
            f"- Terminal fixed flags: {json.dumps(problem.get('terminal', {}).get('fixed'))}",
            f"- Canonical units: {json.dumps(units)}",
            "",
            "## Result",
            "",
            f"- Objective: {fmt(summary.get('objective'), 9)} "
            f"({summary.get('objective_definition', '')})",
            f"- Outer iterations: {summary.get('outer_iterations')} "
            f"(accepted {summary.get('accepted_steps')}, "
            f"rejected {summary.get('rejected_steps')}), "
            f"inner iterations {summary.get('inner_iterations')}",
            f"- Final trust radius: {fmt(summary.get('final_trust_radius'))}",
            f"- Propellant used: {fmt(summary.get('propellant_used'))}, "
            f"final mass {fmt(summary.get('final_mass'))}",
            "- Terminal position error (independent replay): "
            f"{fmt(independent.get('terminal_position_error'), 4)} "
            f"{units.get('position', '')}; velocity error "
            f"{fmt(independent.get('terminal_velocity_error'), 4)} "
            f"{units.get('velocity', '')}",
            "",
            "## Residuals",
            "",
            "| quantity | solver | independent replay |",
            "|---|---:|---:|",
        ]
        residual_rows = (
            ("canonical residual", residuals.get("canonical_residual"), None),
            (
                "dynamics defect (scaled)",
                residuals.get("dynamics_defect"),
                independent.get("dynamics_defect"),
            ),
            (
                "path violation (normalised)",
                residuals.get("path_violation"),
                independent.get("path_violation"),
            ),
            (
                "terminal residual (scaled)",
                residuals.get("terminal_residual"),
                independent.get("terminal_residual"),
            ),
            ("virtual control", residuals.get("virtual_control"), None),
            ("replay parity", None, independent.get("replay_parity")),
            (
                "continuous-time violation (dense replay)",
                None,
                independent.get("continuous_time_violation"),
            ),
        )
        for label, solver_value, replay_value in residual_rows:
            solver_text = "—" if solver_value is None else fmt(solver_value, 4)
            replay_text = "—" if replay_value is None else fmt(replay_value, 4)
            lines.append(f"| {label} | {solver_text} | {replay_text} |")
        lines.extend(
            [
                "",
                "## Certificate gates",
                "",
                "| gate | passed | value | limit |",
                "|---|---|---:|---:|",
            ]
        )
        for name, gate in certificate.get("gates", {}).items():
            lines.append(
                f"| {name} | {fmt(gate.get('passed'))} | {fmt(gate.get('value'), 4)} | "
                f"{fmt(gate.get('limit'), 4)} |"
            )
        lines.extend(
            [
                "",
                "## Timings (seconds)",
                "",
                "| stage | seconds |",
                "|---|---:|",
            ]
        )
        for key in (
            "cuda_startup_seconds",
            "topology_seconds",
            "coefficient_seconds",
            "workspace_create_seconds",
            "solve_seconds",
            "recovery_seconds",
            "replay_seconds",
            "acceptance_seconds",
            "scvx_total_seconds",
            "independent_replay_seconds",
            "plan_wall_seconds",
        ):
            if key in timings:
                lines.append(f"| {key} | {fmt(timings.get(key), 5)} |")
        lines.append("")
        return "\n".join(lines)


def load_result(path: str | Path) -> PlanResult:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("result_kind") != RESULT_KIND:
        raise ValueError(f"{path} is not a spacepdhcg plan result document")
    return PlanResult(document=document, output_directory=Path(path).parent)
