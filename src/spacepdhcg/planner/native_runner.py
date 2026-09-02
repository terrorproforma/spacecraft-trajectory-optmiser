"""Locate and run the native ``spacepdhcg_plan`` executable."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spacepdhcg.planner.problem import dump_canonical
from spacepdhcg.planner.result import RESULT_KIND, PlanResult

EXECUTABLE_ENVIRONMENT = "SPACEPDHCG_PLAN_EXECUTABLE"
EXIT_CODES = {
    0: "certified",
    2: "not_certified",
    3: "solver_failure",
    64: "invalid_problem",
    65: "io_error",
    66: "cuda_error",
    70: "internal_error",
}


class PlanExecutionError(RuntimeError):
    """Raised when the native executable cannot run or reports a hard failure."""

    def __init__(self, message: str, *, exit_code: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


def find_executable(explicit: str | Path | None = None) -> Path | None:
    """Resolve the native planner executable (explicit path, env var, then PATH)."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    override = os.environ.get(EXECUTABLE_ENVIRONMENT)
    if override:
        candidates.append(Path(override))
    located = shutil.which("spacepdhcg_plan")
    if located:
        candidates.append(Path(located))
    for candidate in candidates:
        path = candidate.expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def capabilities(executable: Path, *, timeout: float = 60.0) -> dict[str, Any]:
    """Run ``spacepdhcg_plan --capabilities`` and parse its JSON."""

    try:
        completed = subprocess.run(
            [str(executable), "--capabilities"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PlanExecutionError(f"cannot query {executable}: {error}") from error
    if completed.returncode != 0:
        raise PlanExecutionError(
            f"{executable} --capabilities exited with {completed.returncode}",
            exit_code=completed.returncode,
            stderr=completed.stderr,
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PlanExecutionError(
            f"{executable} --capabilities returned invalid JSON: {error}"
        ) from error


def gpu_availability(executable: Path | None) -> tuple[bool, str, dict[str, Any] | None]:
    """Return (available, reason, capabilities) for the native GPU path."""

    if executable is None:
        return (
            False,
            "the native spacepdhcg_plan executable was not found (set "
            f"{EXECUTABLE_ENVIRONMENT} or add it to PATH)",
            None,
        )
    try:
        info = capabilities(executable)
    except PlanExecutionError as error:
        return False, str(error), None
    if int(info.get("cuda_device_count", 0)) <= 0:
        return False, "the native executable reports no CUDA device", info
    return True, "CUDA device available", info


def run_native_plan(
    executable: Path,
    canonical_document: Mapping[str, Any],
    *,
    output_directory: Path | None = None,
    timeout: float | None = None,
    quiet: bool = True,
) -> PlanResult:
    """Execute one native plan and parse the strict result document."""

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        request_path = output_directory / "native-request.json"
        result_path = output_directory / "native-result.json"
        context = None
    else:
        context = tempfile.TemporaryDirectory(prefix="spacepdhcg-plan-")
        request_path = Path(context.name) / "native-request.json"
        result_path = Path(context.name) / "native-result.json"
    try:
        request_path.write_text(dump_canonical(canonical_document), encoding="utf-8")
        command = [str(executable), str(request_path), "--output", str(result_path)]
        if quiet:
            command.append("--quiet")
        time_limit = float(
            canonical_document.get("solver", {}).get("time_limit_seconds", 0.0) or 0.0
        )
        effective_timeout = (
            timeout if timeout is not None else (time_limit + 300.0 if time_limit > 0 else None)
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise PlanExecutionError(
                f"spacepdhcg_plan did not finish within {effective_timeout} s and was terminated",
                exit_code=None,
                stderr=str(error.stderr or ""),
            ) from error
        except OSError as error:
            raise PlanExecutionError(f"cannot execute {executable}: {error}") from error
        document: dict[str, Any] | None = None
        if result_path.is_file():
            try:
                document = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise PlanExecutionError(
                    f"spacepdhcg_plan wrote an invalid result document: {error}",
                    exit_code=completed.returncode,
                    stderr=completed.stderr,
                ) from error
        if completed.returncode in (64, 65, 66, 70) or document is None:
            status = (document or {}).get("status", {}) if document else {}
            message = status.get("message") or completed.stderr.strip() or "unknown native failure"
            code_name = EXIT_CODES.get(completed.returncode, completed.returncode)
            raise PlanExecutionError(
                f"spacepdhcg_plan failed ({code_name}): {message}",
                exit_code=completed.returncode,
                stderr=completed.stderr,
            )
        if document.get("result_kind") != RESULT_KIND:
            raise PlanExecutionError(
                "spacepdhcg_plan returned an unexpected document kind",
                exit_code=completed.returncode,
                stderr=completed.stderr,
            )
        document.setdefault("backend", {})["executable"] = str(executable)
        document["backend"]["native_exit_code"] = completed.returncode
        document["backend"]["native_stderr"] = completed.stderr[-4000:]
        return PlanResult(document=document, output_directory=output_directory)
    finally:
        if context is not None:
            context.cleanup()
