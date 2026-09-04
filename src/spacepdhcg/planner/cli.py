"""``spacepdhcg`` command line: ``plan``, ``validate``, ``capabilities``, ``defaults``.

Exit codes mirror the native executable: 0 certified, 2 plan produced but not certified,
3 inner solver failure, 64 invalid problem/usage, 65 I/O error, 66 GPU unavailable /
CUDA error, 70 internal error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spacepdhcg.planner.api import GPUUnavailableError, PlanOptions, plan
from spacepdhcg.planner.native_library import PlannerNativeError, native_default_document
from spacepdhcg.planner.native_runner import (
    PlanExecutionError,
    capabilities,
    find_executable,
    gpu_availability,
)
from spacepdhcg.planner.problem import (
    BACKENDS,
    FAMILIES,
    ProblemValidationError,
    dump_canonical,
    load_problem,
    normalise_problem,
)
from spacepdhcg.planner.result import PlanResult, json_safe, load_result

EXIT_CERTIFIED = 0
EXIT_NOT_CERTIFIED = 2
EXIT_SOLVER_FAILURE = 3
EXIT_INVALID_PROBLEM = 64
EXIT_IO_ERROR = 65
EXIT_GPU_UNAVAILABLE = 66
EXIT_INTERNAL = 70


PLANNER_COMMANDS = ("plan", "validate", "capabilities", "defaults", "summary")


def add_commands(commands: argparse._SubParsersAction) -> None:
    """Attach the planner sub-commands to ``commands``.

    Used by the umbrella :mod:`spacepdhcg.cli` dispatcher and by :func:`_parser`; each leaf
    stores its handler in ``func`` so a single dispatcher can serve every track.
    """

    run = commands.add_parser("plan", help="plan one problem document")
    run.add_argument("problem", type=Path, help="problem JSON (schema 1.0.0)")
    run.add_argument(
        "--output", "-o", type=Path, help="output directory for plan-result.json, CSV, summary"
    )
    run.add_argument(
        "--backend", choices=BACKENDS, help="override solver.backend from the document"
    )
    run.add_argument(
        "--executable",
        type=Path,
        help="path to spacepdhcg_plan (else $SPACEPDHCG_PLAN_EXECUTABLE / PATH)",
    )
    run.add_argument(
        "--time-limit", type=float, help="wall-clock limit in seconds (0 = unlimited; default 600)"
    )
    run.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help=(
            "when the GPU path is unavailable run the clearly labelled CPU reference instead "
            "of failing"
        ),
    )
    run.add_argument("--warm-start", type=Path, help="previous plan-result.json to warm start from")
    run.add_argument(
        "--export-viewer", type=Path, help="write a trajectory-viewer bundle into this directory"
    )
    run.add_argument("--no-csv", action="store_true", help="do not write CSV tables")
    run.add_argument("--no-summary", action="store_true", help="do not write plan-summary.md")
    run.add_argument("--json", action="store_true", help="print the full result document to stdout")
    run.add_argument(
        "--verbose", "-v", action="store_true", help="show progress and native diagnostics"
    )
    run.set_defaults(func=command_plan)

    check = commands.add_parser(
        "validate", help="validate a problem document and print the canonical form"
    )
    check.add_argument("problem", type=Path)
    check.add_argument("--quiet", action="store_true", help="only report validity")
    check.set_defaults(func=command_validate)

    caps = commands.add_parser("capabilities", help="report native executable and GPU availability")
    caps.add_argument("--executable", type=Path)
    caps.set_defaults(func=command_capabilities)

    defaults = commands.add_parser("defaults", help="print the native family defaults")
    defaults.add_argument("family", choices=FAMILIES)
    defaults.set_defaults(func=command_defaults)

    summary = commands.add_parser(
        "summary", help="print the summary of an existing plan-result.json"
    )
    summary.add_argument("result", type=Path)
    summary.set_defaults(func=command_summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spacepdhcg",
        description=(
            "SpacePDHCG trajectory planner (persistent single-GPU SCvx with CPU reference)."
        ),
    )
    add_commands(parser.add_subparsers(dest="command", required=True))
    return parser


def _print_result(result: PlanResult, *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(result.to_json())
        return
    summary = result.summary
    replay = result.independent_replay
    lines = [
        f"family:      {result.family}",
        f"execution:   {result.backend_execution} (requested backend {result.requested_backend})",
        f"status:      {result.status} - {result.message}",
        f"certified:   {result.certified}"
        + (f" (failed gates: {', '.join(result.failed_gates)})" if result.failed_gates else ""),
        f"objective:   {summary.get('objective')}",
        f"iterations:  {summary.get('outer_iterations')} outer "
        f"({summary.get('accepted_steps')} accepted), {summary.get('inner_iterations')} inner",
        f"terminal:    scaled residual {replay.get('terminal_residual')}, position error "
        f"{replay.get('terminal_position_error')}, velocity error "
        f"{replay.get('terminal_velocity_error')}",
        f"wall time:   {result.wall_seconds:.3f} s",
    ]
    if result.output_directory is not None:
        lines.append(f"output:      {result.output_directory}")
    if "viewer" in result.files:
        lines.append(f"viewer:      {result.files['viewer']}")
    sys.stdout.write("\n".join(lines) + "\n")


def _exit_code_for(result: PlanResult) -> int:
    if result.status == "certified":
        return EXIT_CERTIFIED
    if result.status == "solver_failure":
        return EXIT_SOLVER_FAILURE
    return EXIT_NOT_CERTIFIED


def command_plan(arguments: argparse.Namespace) -> int:
    warm_start = None
    if arguments.warm_start is not None:
        try:
            warm_start = load_result(arguments.warm_start)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            sys.stderr.write(f"spacepdhcg plan: cannot read warm start: {error}\n")
            return EXIT_IO_ERROR
    progress = (
        (lambda message: sys.stderr.write(f"[spacepdhcg] {message}\n"))
        if arguments.verbose
        else None
    )
    options = PlanOptions(
        backend=arguments.backend,
        output_directory=arguments.output,
        executable=arguments.executable,
        time_limit_seconds=arguments.time_limit,
        allow_cpu_reference_fallback=arguments.allow_cpu_fallback,
        export_viewer=arguments.export_viewer,
        warm_start=warm_start,
        write_csv=not arguments.no_csv,
        write_summary=not arguments.no_summary,
        quiet=not arguments.verbose,
        progress=progress,
    )
    try:
        result = plan(arguments.problem, options)
    except ProblemValidationError as error:
        sys.stderr.write(f"spacepdhcg plan: invalid problem: {error}\n")
        return EXIT_INVALID_PROBLEM
    except GPUUnavailableError as error:
        sys.stderr.write(f"spacepdhcg plan: {error}\n")
        return EXIT_GPU_UNAVAILABLE
    except PlannerNativeError as error:
        sys.stderr.write(f"spacepdhcg plan: native planner library unavailable: {error}\n")
        return EXIT_GPU_UNAVAILABLE
    except PlanExecutionError as error:
        sys.stderr.write(f"spacepdhcg plan: {error}\n")
        if error.stderr and arguments.verbose:
            sys.stderr.write(error.stderr)
        code = error.exit_code if error.exit_code in (64, 65, 66, 70) else EXIT_INTERNAL
        return code
    except OSError as error:
        sys.stderr.write(f"spacepdhcg plan: I/O error: {error}\n")
        return EXIT_IO_ERROR
    if arguments.verbose and result.document.get("backend", {}).get("native_stderr"):
        sys.stderr.write(result.document["backend"]["native_stderr"])
    _print_result(result, as_json=arguments.json)
    return _exit_code_for(result)


def command_validate(arguments: argparse.Namespace) -> int:
    try:
        canonical = normalise_problem(load_problem(arguments.problem))
    except ProblemValidationError as error:
        sys.stderr.write(f"invalid: {error}\n")
        return EXIT_INVALID_PROBLEM
    if arguments.quiet:
        sys.stdout.write("valid\n")
    else:
        sys.stdout.write(dump_canonical(canonical))
    return 0


def command_capabilities(arguments: argparse.Namespace) -> int:
    executable = find_executable(arguments.executable)
    available, reason, info = gpu_availability(executable)
    payload = {
        "executable": str(executable) if executable else None,
        "gpu_available": available,
        "reason": reason,
        "capabilities": info,
        "cpu_reference_available": True,
    }
    if executable is not None and info is None:
        try:
            payload["capabilities"] = capabilities(executable)
        except PlanExecutionError as error:
            payload["capabilities_error"] = str(error)
    sys.stdout.write(json.dumps(json_safe(payload), indent=2) + "\n")
    return 0


def command_defaults(arguments: argparse.Namespace) -> int:
    try:
        document = native_default_document(arguments.family)
    except PlannerNativeError as error:
        sys.stderr.write(f"spacepdhcg defaults: {error}\n")
        return EXIT_GPU_UNAVAILABLE
    sys.stdout.write(json.dumps(document, indent=2) + "\n")
    return 0


def command_summary(arguments: argparse.Namespace) -> int:
    try:
        result = load_result(arguments.result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.stderr.write(f"spacepdhcg summary: {error}\n")
        return EXIT_IO_ERROR
    sys.stdout.write(result.summary_markdown())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Planner-only entry point (``python -m spacepdhcg.planner.cli``).

    The ``spacepdhcg`` console script routes through :func:`spacepdhcg.cli.main`, which mounts
    the same sub-commands next to the other tracks.
    """

    arguments = _parser().parse_args(argv)
    return int(arguments.func(arguments))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
