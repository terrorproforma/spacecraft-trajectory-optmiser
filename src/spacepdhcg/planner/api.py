"""``plan(problem, options) -> PlanResult`` and its options."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spacepdhcg.planner.cpu_reference import solve_cpu_reference
from spacepdhcg.planner.native_runner import (
    PlanExecutionError,
    find_executable,
    gpu_availability,
    run_native_plan,
)
from spacepdhcg.planner.problem import (
    BACKENDS,
    ProblemValidationError,
    apply_defaults,
    dump_canonical,
    load_problem,
    normalise_problem,
)
from spacepdhcg.planner.result import PlanResult

GPU_BACKENDS = ("pure_qoco", "pdhcg", "pdhcg_recovery")


class GPUUnavailableError(RuntimeError):
    """Raised when a GPU backend was requested but no native GPU path is available."""


@dataclass(slots=True)
class PlanOptions:
    """Run-time options that are not part of the problem document."""

    backend: str | None = None
    output_directory: str | Path | None = None
    executable: str | Path | None = None
    time_limit_seconds: float | None = None
    allow_cpu_reference_fallback: bool = False
    export_viewer: str | Path | None = None
    warm_start: PlanResult | Mapping[str, Any] | None = None
    write_csv: bool = True
    write_summary: bool = True
    quiet: bool = True
    progress: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if self.backend is not None and self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, received {self.backend!r}")


def _warm_start_document(source: PlanResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, PlanResult):
        return source.warm_start_document()
    document = dict(source)
    if "states" not in document or "controls" not in document:
        raise ProblemValidationError("warm_start must provide 'states' and 'controls' rows")
    return document


def plan(problem: str | Path | Mapping[str, Any], options: PlanOptions | None = None) -> PlanResult:
    """Plan one trajectory.

    ``problem`` is a path to a problem JSON document or an already loaded mapping.  The
    document is schema-validated and normalised to canonical units; GPU backends run the
    native ``spacepdhcg_plan`` executable, ``cpu_reference`` runs the Python/Clarabel
    reference over the native transcription ABI.  A GPU backend never silently degrades:
    when no GPU path is available the call raises :class:`GPUUnavailableError` unless
    ``allow_cpu_reference_fallback`` is set, in which case the result is clearly labelled
    ``execution: cpu_reference`` with ``fallback_from`` recorded.
    """

    options = options or PlanOptions()
    report = options.progress or (lambda _message: None)
    document = load_problem(problem)
    if options.backend is not None:
        solver = document.setdefault("solver", {})
        solver["backend"] = options.backend
        # A backend override invalidates any preset tied to the previous backend.
        preset = solver.get("preset")
        if preset is not None:
            compatible = {
                "pure_qoco": {"frozen_adaptive_pure_qoco"},
                "cpu_reference": set(),
                "pdhcg": {"frozen_adaptive_pdhcg", "fixed_tight_pdhcg"},
                "pdhcg_recovery": {"fixed_tight_pdhcg"},
            }[options.backend]
            if compatible and preset not in compatible:
                solver.pop("preset")
    if options.warm_start is not None:
        document["warm_start"] = _warm_start_document(options.warm_start)
    canonical = normalise_problem(document)
    apply_defaults(canonical, time_limit_seconds=options.time_limit_seconds)
    backend = canonical["solver"].get("backend", "pure_qoco")
    output_directory = (
        Path(options.output_directory) if options.output_directory is not None else None
    )

    if backend == "cpu_reference":
        report("running the CPU reference planner (Clarabel SCvx over the native transcription)")
        result = solve_cpu_reference(canonical, progress=report)
    else:
        executable = find_executable(options.executable)
        available, reason, capabilities = gpu_availability(executable)
        if not available:
            if not options.allow_cpu_reference_fallback:
                raise GPUUnavailableError(
                    f"backend {backend!r} needs the native CUDA planner but it is unavailable: "
                    f"{reason}. Pass --backend cpu_reference (or "
                    "allow_cpu_reference_fallback=True) to run the clearly labelled CPU "
                    "reference instead."
                )
            report(f"GPU unavailable ({reason}); running the labelled CPU reference fallback")
            fallback = copy.deepcopy(canonical)
            fallback["solver"]["backend"] = "cpu_reference"
            result = solve_cpu_reference(fallback, progress=report)
            result.document["backend"]["fallback_from"] = backend
            result.document["backend"]["fallback_reason"] = reason
            result.document["status"]["message"] += (
                f" [CPU reference fallback: requested backend {backend!r} was unavailable: "
                f"{reason}]"
            )
        else:
            assert executable is not None
            report(f"running native planner {executable} (backend {backend})")
            result = run_native_plan(
                executable,
                canonical,
                output_directory=output_directory,
                quiet=options.quiet,
            )
            if capabilities is not None:
                result.document["backend"]["capabilities"] = capabilities

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "problem.json").write_text(dump_canonical(canonical), encoding="utf-8")
        result.write(
            output_directory, write_csv=options.write_csv, write_summary=options.write_summary
        )
    if options.export_viewer is not None:
        from spacepdhcg.planner.viewer_export import export_viewer_bundle

        result.files["viewer"] = export_viewer_bundle(result, options.export_viewer)
    return result


__all__ = ["GPUUnavailableError", "PlanExecutionError", "PlanOptions", "plan"]
