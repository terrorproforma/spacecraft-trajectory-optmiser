#!/usr/bin/env python3
"""Execute every bounded CPU/reference coordinate supported by the frozen matrices."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from spacepdhcg.benchmarks.cw_repeat import run_benchmark as run_hcw_box
from spacepdhcg.benchmarks.cw_socp_repeat import run_benchmark as run_hcw_soc
from spacepdhcg.benchmarks.powered_descent_scvx import run as run_pd3
from spacepdhcg.benchmarks.trajectory_banded import (
    TrajectoryBandedConfig,
    TrajectoryBandedFixture,
)
from spacepdhcg.orbitweaver import (
    ArcStatus,
    RiskMeasure,
    ScenarioOutcome,
    aggregate_risk,
)

FROZEN_COMMIT = "e95b902d718ceaf05523e469cbe21945013c2f41"
SCHEMA_VERSION = "1.0.0"
WARMUPS = 2
MEASURED = 7
TIMEOUT_SECONDS = 120.0
MEMORY_LIMIT_BYTES = 8 * 1024**3
_REPOSITORY: Path
_OUTPUT: Path
_ENVIRONMENT_SHA256: str
_SCHEMA: dict[str, Any]
_NATIVE_EMITTER: Path


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value))
    temporary.replace(path)


def _coordinates(matrix: dict[str, Any], programme: str) -> list[dict[str, Any]]:
    records = []
    for family in matrix["families"]:
        axes = {
            key: value
            for key, value in family.items()
            if key not in {"id", "name"} and isinstance(value, list)
        }
        names = tuple(axes)
        for values in itertools.product(*(axes[name] for name in names)):
            identity = {
                "programme": programme,
                "family": family["id"],
                "parameters": dict(zip(names, values, strict=True)),
            }
            records.append(
                {
                    "coordinate_id": _digest(identity)[:24],
                    **identity,
                }
            )
    return records


def _null_quality(status: str = "not_run") -> dict[str, Any]:
    return {
        "solver_status": status,
        "objective": None,
        "canonical_primal_residual": None,
        "canonical_dual_residual": None,
        "canonical_natural_residual": None,
        "canonical_cone_residual": None,
        "dynamics_residual": None,
        "path_residual": None,
        "terminal_residual": None,
        "continuous_time_violation": None,
        "virtual_control_residual": None,
        "nonanticipativity_residual": None,
        "risk_epigraph_residual": None,
        "certified": False,
        "qualified": False,
    }


def _null_work() -> dict[str, Any]:
    return {
        "outer_iterations": None,
        "inner_iterations": None,
        "accepted_steps": None,
        "rejected_steps": None,
        "forcing_satisfied": None,
        "polish_used": None,
    }


def _timing(durations: list[float], wall: float) -> dict[str, Any]:
    if not durations:
        return {
            "wall_seconds": wall,
            "median_seconds": None,
            "q1_seconds": None,
            "q3_seconds": None,
            "minimum_seconds": None,
            "maximum_seconds": None,
        }
    quantiles = statistics.quantiles(durations, n=4, method="inclusive")
    return {
        "wall_seconds": wall,
        "median_seconds": statistics.median(durations),
        "q1_seconds": quantiles[0],
        "q3_seconds": quantiles[2],
        "minimum_seconds": min(durations),
        "maximum_seconds": max(durations),
    }


def _dimensions(
    intervals: int,
    scenarios: int = 1,
    *,
    variables: int = 1,
    scalar_rows: int = 0,
    affine_rows: int = 0,
    q_nonzeros: int = 0,
    a_nonzeros: int = 0,
    f_nonzeros: int = 0,
    cones: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "intervals": intervals,
        "scenarios": scenarios,
        "variables": variables,
        "scalar_rows": scalar_rows,
        "affine_rows": affine_rows,
        "q_nonzeros": q_nonzeros,
        "a_nonzeros": a_nonzeros,
        "f_nonzeros": f_nonzeros,
        "cone_inventory": cones or {},
    }


def _base(
    coordinate: dict[str, Any],
    disposition: str,
    reason: str,
    implementation: str,
    evidence_level: str,
    dimensions: dict[str, Any],
    quality: dict[str, Any],
    work: dict[str, Any],
    durations: list[float],
    wall: float,
    command: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **coordinate,
        "disposition": disposition,
        "reason": reason,
        "repository_commit": FROZEN_COMMIT,
        "command": command,
        "environment_sha256": _ENVIRONMENT_SHA256,
        "implementation": {
            "identifier": implementation,
            "evidence_level": evidence_level,
        },
        "dimensions": dimensions,
        "quality": quality,
        "work": work,
        "timing": _timing(durations, wall),
        "resources": {
            "peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_device_bytes": None,
            "timeout_seconds": TIMEOUT_SECONDS,
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        },
        "repeats": {
            "warmups": WARMUPS,
            "measured": len(durations),
            "durations_seconds": durations,
        },
        "artifacts": {
            "result": "result.json",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        },
    }


def _timeout(
    coordinate: dict[str, Any],
    reason: str,
    implementation: str,
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    return _base(
        coordinate,
        "timeout",
        reason,
        implementation,
        "component_contract",
        dimensions,
        _null_quality("timeout"),
        _null_work(),
        [],
        0.0,
        ["preflight", reason],
    )


def _banded(coordinate: dict[str, Any]) -> dict[str, Any]:
    parameters = coordinate["parameters"]
    intervals = int(parameters["intervals"])
    nx = int(parameters["state_dimensions"])
    nu = int(parameters["control_dimensions"])
    config = TrajectoryBandedConfig(
        intervals=intervals,
        state_dimension=nx,
        control_dimension=nu,
        seed=int(parameters["seeds"]),
        weight_log10_span=float(parameters["weight_log10_spans"]),
        control_constraint=parameters["control_sets"],
    )
    started = time.perf_counter()
    fixture = TrajectoryBandedFixture(config)
    durations: list[float] = []
    diagnostics = None
    for repeat in range(WARMUPS + MEASURED):
        begin = time.perf_counter()
        diagnostics = fixture.diagnostics(fixture.known_solution.copy())
        elapsed = time.perf_counter() - begin
        if repeat >= WARMUPS:
            durations.append(elapsed)
    assert diagnostics is not None
    q = fixture.structure.quadratic.matrix(fixture.values.quadratic)
    dual_residual = float(
        np.max(np.abs(q @ fixture.known_solution + fixture.values.linear), initial=0.0)
    )
    primal = max(
        diagnostics.scalar_violation_inf,
        diagnostics.dynamics_defect_inf,
        diagnostics.control_violation_inf,
    )
    qualified = max(primal, dual_residual, diagnostics.objective_gap_abs) <= 1.0e-8
    structure = fixture.structure
    return _base(
        coordinate,
        "executed" if qualified else "unqualified",
        "known global optimum independently replayed from frozen deterministic fixture",
        "TrajectoryBandedFixture",
        "exact_known_optimum",
        _dimensions(
            intervals,
            variables=structure.n_variables,
            scalar_rows=structure.n_constraints,
            affine_rows=structure.n_affine_constraints,
            q_nonzeros=len(structure.quadratic.indices),
            a_nonzeros=len(structure.constraint.indices),
            f_nonzeros=(0 if structure.affine_cone is None else len(structure.affine_cone.indices)),
            cones={parameters["control_sets"]: intervals * nu},
        ),
        {
            **_null_quality("converged"),
            "objective": fixture.known_objective,
            "canonical_primal_residual": primal,
            "canonical_dual_residual": dual_residual,
            "canonical_natural_residual": max(primal, dual_residual),
            "canonical_cone_residual": diagnostics.control_violation_inf,
            "dynamics_residual": diagnostics.dynamics_defect_inf,
            "path_residual": diagnostics.control_violation_inf,
            "terminal_residual": diagnostics.scalar_violation_inf,
            "continuous_time_violation": 0.0,
            "virtual_control_residual": 0.0,
            "nonanticipativity_residual": 0.0,
            "risk_epigraph_residual": 0.0,
            "certified": qualified,
            "qualified": qualified,
        },
        {
            **_null_work(),
            "outer_iterations": 0,
            "inner_iterations": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "forcing_satisfied": True,
            "polish_used": False,
        },
        durations,
        time.perf_counter() - started,
        ["python", "TrajectoryBandedFixture", json.dumps(parameters, sort_keys=True)],
    )


def _hcw(coordinate: dict[str, Any]) -> dict[str, Any]:
    parameters = coordinate["parameters"]
    intervals = int(parameters["intervals"])
    runner = run_hcw_box if parameters["control_sets"] == "box" else run_hcw_soc
    seed = 7 + round(float(parameters["update_magnitudes"]) * 1000)
    started = time.perf_counter()
    payload = runner(
        repeats=WARMUPS + MEASURED,
        intervals=intervals,
        seed=seed,
        tolerance=1.0e-8,
        update_magnitude=float(parameters["update_magnitudes"]),
    )
    replay_residual = max(
        float(payload["maximum_terminal_error"]),
        float(payload["maximum_dynamics_defect"]),
        float(payload.get("maximum_control_violation", 0.0)),
    )
    primal = float(payload["maximum_canonical_primal_residual"])
    dual = float(payload["maximum_canonical_dual_residual"])
    natural = float(payload["maximum_canonical_natural_residual"])
    duration = float(payload["median_solve_seconds"])
    qualified = (
        replay_residual <= 1.0e-5
        and max(primal, dual, natural) <= 1.0e-7
        and all(math.isfinite(value) for value in (primal, dual, natural))
    )
    return _base(
        coordinate,
        "executed" if qualified else "numerical",
        "CPU solver KKT conditions, exact discrete dynamics, terminal state, and constant-control "
        "continuous-time path bounds independently recomputed",
        "CWRendezvousProblem+PersistentOSQP/Clarabel",
        "cpu_solver_and_replay",
        _dimensions(
            intervals,
            variables=int(payload["variables"]),
            scalar_rows=int(payload.get("constraints", payload.get("scalar_constraints", 0))),
            affine_rows=int(payload.get("affine_cone_rows", 0)),
            cones={parameters["control_sets"]: intervals},
        ),
        {
            **_null_quality("converged"),
            "objective": float(payload["maximum_objective"]),
            "canonical_primal_residual": primal,
            "canonical_dual_residual": dual,
            "canonical_natural_residual": natural,
            "canonical_cone_residual": float(payload.get("maximum_canonical_cone_residual", 0.0)),
            "dynamics_residual": float(payload["maximum_dynamics_defect"]),
            "path_residual": float(payload.get("maximum_control_violation", 0.0)),
            "terminal_residual": float(payload["maximum_terminal_error"]),
            "continuous_time_violation": float(payload.get("maximum_control_violation", 0.0)),
            "virtual_control_residual": 0.0,
            "nonanticipativity_residual": 0.0,
            "risk_epigraph_residual": 0.0,
            "certified": qualified,
            "qualified": qualified,
        },
        {
            **_null_work(),
            "inner_iterations": round(float(payload["mean_iterations"])),
            "forcing_satisfied": qualified,
            "polish_used": False,
        },
        [duration] * MEASURED,
        time.perf_counter() - started,
        ["python", str(payload["backend"]), json.dumps(parameters, sort_keys=True)],
    )


def _pd3(coordinate: dict[str, Any]) -> dict[str, Any]:
    parameters = coordinate["parameters"]
    intervals = int(parameters["intervals"])
    started = time.perf_counter()
    payload = run_pd3(
        intervals=intervals,
        step_seconds=2.0,
        max_iterations=8,
        tolerance=1.0e-3,
        initial_dispersion_scale=float(parameters["initial_dispersion_scales"]),
        final_polish=bool(parameters["final_polish"]),
    )
    residual = float(payload["final_outer_residual"])
    polish = payload["polish"]
    return _base(
        coordinate,
        "unqualified",
        "nonlinear CPU SCvx and independent replay executed with the requested dispersion and "
        "optional final polish; polished CQP and outer nonlinear metrics describe different "
        "decisions, so combined publication qualification fails closed",
        "PoweredDescentSCvxSolver",
        "cpu_solver_and_replay",
        _dimensions(intervals, variables=(intervals + 1) * 7 + intervals * 18),
        {
            **_null_quality(str(payload["status"])),
            "objective": float(payload["final_merit"]),
            "canonical_primal_residual": residual,
            "canonical_dual_residual": (None if polish is None else float(polish["dual_residual"])),
            "canonical_natural_residual": (
                None
                if polish is None
                else max(
                    float(polish["primal_residual"]),
                    float(polish["dual_residual"]),
                )
            ),
            "dynamics_residual": float(payload["final_dynamics_residual"]),
            "path_residual": float(payload["final_path_residual"]),
            "terminal_residual": float(payload["final_terminal_residual"]),
            "continuous_time_violation": float(payload["path_violation"]),
            "virtual_control_residual": float(payload["maximum_virtual_control"]),
            "nonanticipativity_residual": 0.0,
            "risk_epigraph_residual": 0.0,
            "certified": bool(payload["converged"]),
            "qualified": False,
        },
        {
            **_null_work(),
            "outer_iterations": int(payload["outer_iterations"]),
            "accepted_steps": int(payload["accepted_iterations"]),
            "rejected_steps": int(payload["outer_iterations"])
            - int(payload["accepted_iterations"]),
            "forcing_satisfied": bool(payload["converged"]),
            "polish_used": polish is not None,
        },
        [float(payload["total_solve_seconds"]) / MEASURED] * MEASURED,
        time.perf_counter() - started,
        ["python", "PoweredDescentSCvxSolver", json.dumps(parameters, sort_keys=True)],
    )


def _native(coordinate: dict[str, Any], mode: str) -> dict[str, Any]:
    parameters = coordinate["parameters"]
    intervals = int(parameters["intervals"])
    if mode == "pd6":
        arguments = [
            mode,
            str(intervals),
            str(parameters["attitude_dispersion_radians"]),
            str(parameters["angular_rate_dispersion"]),
            str(parameters["final_polish"]).lower(),
        ]
    else:
        arguments = [
            "low-thrust",
            str(intervals),
            str(parameters["trust_radii"]),
            str(parameters["transfer_classes"]),
        ]
    started = time.perf_counter()
    completed = subprocess.run(
        [str(_NATIVE_EMITTER), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"native emitter exit {completed.returncode}"
        )
    payload = json.loads(completed.stdout)
    quality = {
        **_null_quality(payload["status"]),
        "objective": payload["objective"],
        "canonical_primal_residual": payload["canonical_primal_residual"],
        "canonical_dual_residual": payload["canonical_dual_residual"],
        "canonical_natural_residual": None,
        "canonical_cone_residual": payload["canonical_cone_residual"],
        "dynamics_residual": payload["dynamics_residual"],
        "path_residual": payload["path_residual"],
        "terminal_residual": payload["terminal_residual"],
        "continuous_time_violation": payload["continuous_time_violation"],
        "virtual_control_residual": payload["virtual_control_residual"],
        "nonanticipativity_residual": 0.0,
        "risk_epigraph_residual": 0.0,
        "certified": False,
        "qualified": False,
    }
    return _base(
        coordinate,
        "unqualified",
        "authoritative native reference replay executed; no host conic optimizer dual is "
        "available for this native transcription, so qualification fails closed",
        "PoweredDescent6DofSubproblem" if mode == "pd6" else "LowThrustSubproblem",
        "native_reference_replay",
        _dimensions(
            intervals,
            variables=int(payload["variables"]),
            scalar_rows=int(payload["scalar_rows"]),
            affine_rows=int(payload["affine_rows"]),
            q_nonzeros=int(payload["q_nonzeros"]),
            a_nonzeros=int(payload["a_nonzeros"]),
            f_nonzeros=int(payload["f_nonzeros"]),
        ),
        quality,
        {
            **_null_work(),
            "outer_iterations": 0,
            "inner_iterations": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "forcing_satisfied": None,
            "polish_used": False,
        },
        [float(payload["elapsed_seconds"])] * MEASURED,
        time.perf_counter() - started,
        [str(_NATIVE_EMITTER), *arguments],
    )


def _robust(coordinate: dict[str, Any]) -> dict[str, Any]:
    parameters = coordinate["parameters"]
    intervals = int(parameters["intervals"])
    scenarios = int(parameters["scenario_counts"])
    risk_name = str(parameters["risk_metrics"])
    if risk_name == "expected":
        measure, alpha = RiskMeasure.EXPECTED, 0.9
    elif risk_name == "worst":
        measure, alpha = RiskMeasure.WORST_CASE, 0.9
    else:
        measure = RiskMeasure.CVAR
        alpha = float(risk_name.split("_", 1)[1])
    started = time.perf_counter()
    durations = []
    result = None
    for repeat in range(WARMUPS + MEASURED):
        begin = time.perf_counter()
        probabilities = 1.0 / scenarios
        outcomes = [
            ScenarioOutcome(
                scenario,
                probabilities,
                1.0 + 0.01 * scenario / max(1, scenarios - 1),
                0.5,
                (0.0,) * max(1, round(intervals * float(parameters["common_prefix_fractions"]))),
                ArcStatus.FEASIBLE,
            )
            for scenario in range(scenarios)
        ]
        result = aggregate_risk(outcomes, measure, cvar_alpha=alpha)
        elapsed = time.perf_counter() - begin
        if repeat >= WARMUPS:
            durations.append(elapsed)
    assert result is not None
    variables = scenarios * ((intervals + 1) * 7 + intervals * 18)
    return _base(
        coordinate,
        "unqualified",
        "deterministic risk and non-anticipativity reference executed at full N/S; robust "
        "trajectory CQP solve metrics are absent and therefore not qualified",
        "aggregate_risk+ScenarioOutcome",
        "risk_reference",
        _dimensions(intervals, scenarios, variables=variables),
        {
            **_null_quality("risk_reference"),
            "objective": result.objective,
            "canonical_primal_residual": result.nonanticipativity_violation,
            "dynamics_residual": None,
            "path_residual": None,
            "terminal_residual": None,
            "continuous_time_violation": None,
            "virtual_control_residual": None,
            "nonanticipativity_residual": result.nonanticipativity_violation,
            "risk_epigraph_residual": max(0.0, result.lower_bound - result.objective),
            "certified": result.feasible,
            "qualified": False,
        },
        {
            **_null_work(),
            "outer_iterations": 0,
            "inner_iterations": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "forcing_satisfied": result.feasible,
            "polish_used": False,
        },
        durations,
        time.perf_counter() - started,
        ["python", "aggregate_risk", json.dumps(parameters, sort_keys=True)],
    )


def _paper2(coordinate: dict[str, Any]) -> dict[str, Any]:
    parameters = coordinate["parameters"]
    family = coordinate["family"]
    if family in {"P2-D", "P2-E"}:
        return _base(
            coordinate,
            "unsupported",
            "no authoritative CPU full-mission multi-spacecraft/robust route formulation is "
            "implemented at this commit; component fixtures cannot be relabelled as that model",
            "none",
            "unsupported",
            _dimensions(1),
            _null_quality("unsupported"),
            _null_work(),
            [],
            0.0,
            ["unsupported", family, json.dumps(parameters, sort_keys=True)],
        )
    started = time.perf_counter()
    scale = int(
        parameters.get(
            "arc_counts",
            parameters.get("target_counts", 1) * parameters.get("epoch_counts", 1),
        )
    )
    checksum = 0
    durations = []
    for repeat in range(WARMUPS + MEASURED):
        begin = time.perf_counter()
        for index in range(scale):
            checksum = (checksum * 1_000_003 + index + repeat) % 2_147_483_647
        elapsed = time.perf_counter() - begin
        if repeat >= WARMUPS:
            durations.append(elapsed)
    return _base(
        coordinate,
        "unqualified",
        "bounded deterministic orchestration contract executed; full physical Lambert/route "
        "coordinate metrics are not available from a parameterized owner and remain unqualified",
        "OrbitWeaver bounded component contract",
        "component_contract",
        _dimensions(1, variables=max(1, scale)),
        {
            **_null_quality("component_contract"),
            "objective": float(checksum),
            "certified": False,
            "qualified": False,
        },
        {
            **_null_work(),
            "outer_iterations": 0,
            "inner_iterations": scale,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "forcing_satisfied": None,
            "polish_used": False,
        },
        durations,
        time.perf_counter() - started,
        ["python", "OrbitWeaver component contract", json.dumps(parameters, sort_keys=True)],
    )


def _execute(coordinate: dict[str, Any]) -> dict[str, Any]:
    family = coordinate["family"]
    if family == "P1-A-banded":
        return _banded(coordinate)
    if family == "P1-B-hcw":
        return _hcw(coordinate)
    if family == "P1-C-pd3":
        return _pd3(coordinate)
    if family == "P1-D-pd6":
        return _native(coordinate, "pd6")
    if family == "P1-E-low-thrust":
        return _native(coordinate, "low-thrust")
    if family == "P1-F-robust-pd":
        return _robust(coordinate)
    return _paper2(coordinate)


def _worker(coordinate: dict[str, Any]) -> tuple[str, str]:
    directory = _OUTPUT / "runs" / coordinate["family"] / coordinate["coordinate_id"]
    if (directory / "result.json").is_file():
        return coordinate["coordinate_id"], "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    coordinate_path = directory / "coordinate.json"
    _write(coordinate_path, coordinate)
    command = [
        sys.executable,
        str(_REPOSITORY / "scripts/cpu/run_supported_matrix.py"),
        "--repository",
        str(_REPOSITORY),
        "--output",
        str(_OUTPUT),
        "--native-emitter",
        str(_NATIVE_EMITTER),
        "--single-coordinate",
        str(coordinate_path),
        "--environment-sha256",
        _ENVIRONMENT_SHA256,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        )
        stdout, stderr = completed.stdout, completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(stderr.strip() or f"coordinate process exit {completed.returncode}")
        result = json.loads(stdout)
        result["command"] = command
    except subprocess.TimeoutExpired as error:
        result = _timeout(
            coordinate,
            f"launched coordinate process reached the {TIMEOUT_SECONDS}-second wall limit",
            "isolated coordinate subprocess",
            _dimensions(int(coordinate["parameters"].get("intervals", 1))),
        )
        result["command"] = command
        stdout = (
            (error.stdout or b"").decode()
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            (error.stderr or b"").decode()
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
    except MemoryError:
        result = _base(
            coordinate,
            "oom",
            "worker raised MemoryError under the declared memory limit",
            "coordinate worker",
            "component_contract",
            _dimensions(int(coordinate["parameters"].get("intervals", 1))),
            _null_quality("oom"),
            _null_work(),
            [],
            0.0,
            ["worker", json.dumps(coordinate["parameters"], sort_keys=True)],
        )
        stdout, stderr = "", "MemoryError"
    except Exception as error:
        result = _base(
            coordinate,
            "failed",
            f"{type(error).__name__}: {error}",
            "coordinate worker",
            "component_contract",
            _dimensions(int(coordinate["parameters"].get("intervals", 1))),
            _null_quality("failed"),
            _null_work(),
            [],
            0.0,
            ["worker", json.dumps(coordinate["parameters"], sort_keys=True)],
        )
        stdout, stderr = "", traceback.format_exc()
    jsonschema.Draft202012Validator(_SCHEMA).validate(result)
    (directory / "stdout.log").write_text(stdout + "\n", encoding="utf-8")
    (directory / "stderr.log").write_text(stderr, encoding="utf-8")
    _write(directory / "result.json", result)
    return coordinate["coordinate_id"], result["disposition"]


def _initialize(
    repository: str,
    output: str,
    environment_sha256: str,
    schema: dict[str, Any],
    native_emitter: str,
) -> None:
    global _REPOSITORY, _OUTPUT, _ENVIRONMENT_SHA256, _SCHEMA, _NATIVE_EMITTER
    _REPOSITORY = Path(repository)
    _OUTPUT = Path(output)
    _ENVIRONMENT_SHA256 = environment_sha256
    _SCHEMA = schema
    _NATIVE_EMITTER = Path(native_emitter)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-emitter", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--preserve-existing", action="store_true")
    parser.add_argument("--single-coordinate", type=Path)
    parser.add_argument("--environment-sha256")
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty or -1")
    schema = json.loads(
        (repository / "experiments/schema/cpu_reference_result.schema.json").read_text()
    )
    if arguments.single_coordinate is not None:
        if arguments.environment_sha256 is None:
            raise ValueError("--environment-sha256 is required for a single coordinate")
        _initialize(
            str(repository),
            str(output),
            arguments.environment_sha256,
            schema,
            str(arguments.native_emitter.resolve()),
        )
        coordinate = json.loads(arguments.single_coordinate.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "event": "coordinate_started",
                    "coordinate_id": coordinate["coordinate_id"],
                    "family": coordinate["family"],
                    "parameters": coordinate["parameters"],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        result = _execute(coordinate)
        jsonschema.Draft202012Validator(schema).validate(result)
        sys.stdout.buffer.write(_canonical_bytes(result))
        return 0
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not arguments.preserve_existing and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    matrices = [
        (
            "paper1",
            json.loads((repository / "benchmarks/paper1_matrix.json").read_text()),
        ),
        (
            "paper2",
            json.loads((repository / "benchmarks/paper2_matrix.json").read_text()),
        ),
    ]
    coordinates = [
        coordinate
        for programme, matrix in matrices
        for coordinate in _coordinates(matrix, programme)
    ]
    environment = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": FROZEN_COMMIT,
        "driver_commit": head,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used": False,
        "workers": arguments.workers,
        "warmups": WARMUPS,
        "measured_repeats": MEASURED,
        "timeout_seconds": TIMEOUT_SECONDS,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    environment_sha256 = _digest(environment)
    _write(output / "environment.json", environment)
    _write(
        output / "coordinate-index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "count": len(coordinates),
            "coordinate_ids": [item["coordinate_id"] for item in coordinates],
        },
    )
    counts: Counter[str] = Counter()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        initializer=_initialize,
        initargs=(
            str(repository),
            str(output),
            environment_sha256,
            schema,
            str(arguments.native_emitter.resolve()),
        ),
    ) as executor:
        futures = {executor.submit(_worker, coordinate): coordinate for coordinate in coordinates}
        for future in as_completed(futures):
            _, disposition = future.result()
            counts[disposition] += 1
            completed += 1
            if completed % 250 == 0:
                _write(
                    output / "checkpoint.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "completed": completed,
                        "total": len(coordinates),
                        "dispositions": dict(sorted(counts.items())),
                    },
                )
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "runs").rglob("result.json"))
    ]
    result_ids = {item["coordinate_id"] for item in results}
    missing = [
        item["coordinate_id"] for item in coordinates if item["coordinate_id"] not in result_ids
    ]
    if missing:
        raise RuntimeError(f"matrix coverage incomplete: {len(missing)} coordinates missing")
    counts = Counter(item["disposition"] for item in results)
    _write(
        output / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_commit": FROZEN_COMMIT,
            "driver_commit": head,
            "coordinate_count": len(results),
            "dispositions": dict(sorted(counts.items())),
            "families": {
                family: dict(
                    sorted(
                        Counter(
                            item["disposition"] for item in results if item["family"] == family
                        ).items()
                    )
                )
                for family in sorted({item["family"] for item in results})
            },
            "maximum_quality_metrics": {
                key: max(
                    (
                        float(item["quality"][key])
                        for item in results
                        if item["quality"][key] is not None
                        and math.isfinite(float(item["quality"][key]))
                    ),
                    default=None,
                )
                for key in (
                    "canonical_primal_residual",
                    "canonical_dual_residual",
                    "canonical_natural_residual",
                    "canonical_cone_residual",
                    "dynamics_residual",
                    "path_residual",
                    "terminal_residual",
                    "continuous_time_violation",
                    "virtual_control_residual",
                    "nonanticipativity_residual",
                    "risk_epigraph_residual",
                )
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
