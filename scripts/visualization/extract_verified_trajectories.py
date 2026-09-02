#!/usr/bin/env python3
"""Extract compact, traceable trajectory visualisation evidence without GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0.0"
STATE_ARRAY_KEYS = frozenset(
    {"states", "state_history", "trajectory_states", "positions", "node_positions"}
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return False


def has_traceable_state_arrays(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in STATE_ARRAY_KEYS and _is_numeric_matrix(item):
                return True
            if has_traceable_state_arrays(item):
                return True
    elif isinstance(value, list):
        return any(has_traceable_state_arrays(item) for item in value)
    return False


def _is_numeric_matrix(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and all(
            isinstance(row, list)
            and len(row) >= 3
            and all(
                isinstance(component, (int, float)) and not isinstance(component, bool)
                for component in row
            )
            for row in value
        )
    )


def decimate_indices(
    times: np.ndarray,
    positions: np.ndarray,
    maximum_points: int,
    required_indices: set[int] | None = None,
) -> list[int]:
    """Select exact source points while preserving endpoints, extrema and required points."""

    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (times.size, 3):
        raise ValueError("times and positions must have shapes (N,) and (N, 3)")
    if times.size < 2 or maximum_points < 2:
        raise ValueError("at least two source and output points are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("trajectory values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    required = {0, times.size - 1}
    for axis in range(3):
        required.add(int(np.argmin(positions[:, axis])))
        required.add(int(np.argmax(positions[:, axis])))
    radii = np.linalg.norm(positions, axis=1)
    required.update({int(np.argmin(radii)), int(np.argmax(radii))})
    if required_indices:
        required.update(int(index) for index in required_indices)
    if any(index < 0 or index >= times.size for index in required):
        raise ValueError("required decimation index is outside the trajectory")
    if times.size <= maximum_points:
        return list(range(times.size))
    if len(required) > maximum_points:
        raise ValueError("maximum_points cannot preserve every required trajectory point")

    remaining = maximum_points - len(required)
    if remaining:
        candidates = np.linspace(0, times.size - 1, remaining + 2, dtype=np.int64)[1:-1]
        selected = required | {int(index) for index in candidates}
        if len(selected) < maximum_points:
            for index in np.argsort(
                np.min(
                    np.abs(
                        np.arange(times.size, dtype=np.int64)[:, None]
                        - np.asarray(sorted(selected), dtype=np.int64)[None, :]
                    ),
                    axis=1,
                )
            )[::-1]:
                selected.add(int(index))
                if len(selected) == maximum_points:
                    break
    else:
        selected = required
    return sorted(selected)


def compact_points(
    times: np.ndarray,
    states: np.ndarray,
    maximum_points: int,
    required: set[int],
) -> dict[str, Any]:
    positions = np.asarray(states, dtype=np.float64)[:, :3]
    indices = decimate_indices(times, positions, maximum_points, required)
    points = [
        [float(times[index]), *(float(value) for value in positions[index])] for index in indices
    ]
    return {
        "point_count": len(points),
        "original_point_count": int(times.size),
        "original_sha256": canonical_sha256(
            {
                "times": np.asarray(times, dtype=np.float64).tolist(),
                "states": np.asarray(states, dtype=np.float64).tolist(),
            }
        ),
        "selected_indices": indices,
        "points_txyz": points,
    }


def verify_source_checksums(
    archive: Path,
    relative_paths: list[str],
) -> list[dict[str, Any]]:
    manifest = json_load(archive / "checksums.json")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    checks: list[dict[str, Any]] = []
    for relative in relative_paths:
        expected = entries.get(relative)
        if expected is None:
            raise ValueError(f"source checksum is missing {relative}")
        path = archive / relative
        actual = file_sha256(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"source checksum mismatch for {relative}")
        checks.append(
            {
                "path": str(path),
                "bytes": expected["bytes"],
                "sha256": actual,
                "verified": True,
            }
        )
    return checks


def assert_close(actual: float, expected: float, name: str, tolerance: float = 2.0e-8) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name} mismatch: recomputed={actual!r}, archived={expected!r}")


def array_summary(values: np.ndarray, labels: list[str]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        label: {
            "minimum": float(np.min(values[:, index])),
            "maximum": float(np.max(values[:, index])),
            "mean": float(np.mean(values[:, index])),
        }
        for index, label in enumerate(labels)
    }


def _hcw_required(states: np.ndarray, controls: np.ndarray, limit: float) -> set[int]:
    required: set[int] = set()
    margin = limit - np.max(np.abs(controls), axis=1)
    for index in np.argsort(margin)[: min(8, margin.size)]:
        required.update({int(index), int(index) + 1})
    speed = np.linalg.norm(states[:, 3:6], axis=1)
    required.update({int(np.argmin(speed)), int(np.argmax(speed))})
    return required


def extract_hcw(
    cpu_source: Path, archived: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from spacepdhcg.backends import PersistentOSQP
    from spacepdhcg.cqp import CanonicalCQP, independent_canonical_residuals
    from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem
    from spacepdhcg.models.cw import discretise_cw

    parameters = archived["parameters"]
    intervals = int(parameters["intervals"])
    seed = 7 + round(float(parameters["update_magnitudes"]) * 1000)
    rng = np.random.default_rng(seed)
    problem = CWRendezvousProblem(
        CWRendezvousConfig(intervals=intervals, max_component_acceleration=5.0e-2)
    )
    base_initial = np.concatenate((rng.uniform(-100.0, 100.0, 3), rng.uniform(-0.05, 0.05, 3)))
    base_target = np.concatenate((rng.uniform(-5.0, 5.0, 3), np.zeros(3)))
    boundaries: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(9):
        initial_delta = rng.normal(size=6)
        target_delta = rng.normal(size=6)
        initial_delta /= max(float(np.linalg.norm(initial_delta)), 1.0)
        target_delta /= max(float(np.linalg.norm(target_delta)), 1.0)
        boundaries.append(
            (
                base_initial + float(parameters["update_magnitudes"]) * initial_delta,
                base_target + float(parameters["update_magnitudes"]) * target_delta,
            )
        )
    backend = PersistentOSQP(problem.canonical(*boundaries[0]))
    records: list[dict[str, Any]] = []
    previous = None
    for repeat_index, (initial, target) in enumerate(boundaries):
        values = problem.values(initial, target)
        backend.update(values)
        if previous is not None:
            backend.warm_start(previous.primal, previous.dual)
        solution = backend.solve(tolerance=1.0e-8)
        diagnostics = problem.diagnostics(solution.primal, initial, target)
        audit = independent_canonical_residuals(
            CanonicalCQP(problem.structure, values),
            solution.primal,
            solution.dual,
        )
        states = solution.primal[: problem.layout.state_variable_count].reshape(intervals + 1, 6)
        controls = solution.primal[problem.layout.state_variable_count :].reshape(intervals, 3)
        records.append(
            {
                "repeat_index": repeat_index,
                "initial": initial,
                "target": target,
                "solution": solution,
                "diagnostics": diagnostics,
                "audit": audit,
                "states": states,
                "controls": controls,
            }
        )
        previous = solution
    selected = max(records, key=lambda item: float(item["solution"].objective))
    quality = archived["quality"]
    assert_close(
        max(item["diagnostics"].terminal_error_inf for item in records),
        quality["terminal_residual"],
        "HCW terminal residual",
    )
    assert_close(
        max(item["diagnostics"].dynamics_defect_inf for item in records),
        quality["dynamics_residual"],
        "HCW dynamics residual",
    )
    assert_close(
        max(item["audit"].primal for item in records),
        quality["canonical_primal_residual"],
        "HCW primal residual",
    )
    assert_close(
        max(item["audit"].dual for item in records),
        quality["canonical_dual_residual"],
        "HCW dual residual",
    )
    assert_close(
        max(item["solution"].objective for item in records), quality["objective"], "HCW objective"
    )

    substeps = 5
    sub_ad, sub_bd = discretise_cw(
        problem.config.mean_motion,
        problem.config.step_seconds / substeps,
    )
    dense = np.empty((intervals * substeps + 1, 6), dtype=np.float64)
    dense[0] = selected["initial"]
    for interval, control in enumerate(selected["controls"]):
        for substep in range(substeps):
            index = interval * substeps + substep
            dense[index + 1] = sub_ad @ dense[index] + sub_bd @ control
    times = np.arange(intervals + 1, dtype=np.float64) * problem.config.step_seconds
    dense_times = (
        np.arange(intervals * substeps + 1, dtype=np.float64)
        * problem.config.step_seconds
        / substeps
    )
    dense_terminal = float(np.max(np.abs(dense[-1] - selected["target"])))
    required = _hcw_required(
        selected["states"],
        selected["controls"],
        problem.config.max_component_acceleration,
    )
    source = {
        "times": times,
        "states": selected["states"],
        "controls": selected["controls"],
        "dense_times": dense_times,
        "dense_states": dense,
    }
    metadata = {
        "trajectory_id": "p1b_hcw_cpu_040d020d_repeat_max_objective",
        "family": "P1-B",
        "physical_family": "HCW rendezvous",
        "source": {
            "commit": archived["repository_commit"],
            "campaign": "cpu-actual-c5a4991",
            "run_id": archived["coordinate_id"],
            "coordinate": archived["parameters"],
            "repeat_index": selected["repeat_index"],
            "solver": archived["implementation"]["identifier"],
            "policy": "persistent OSQP; tolerance=1e-8; warm-start previous repeat",
            "status": archived["disposition"],
        },
        "qualification": {"qualified": True, "label": "qualified CPU solver + replay"},
        "frame": "Hill/LVLH relative Cartesian [radial, along-track, cross-track]",
        "position_units": "m",
        "time_units": "s",
        "initial_state": selected["initial"].tolist(),
        "terminal_target": selected["target"].tolist(),
        "state_order": ["x", "y", "z", "vx", "vy", "vz"],
        "control_order": ["ax", "ay", "az"],
        "controls_summary": array_summary(selected["controls"], ["ax", "ay", "az"]),
        "mass_summary": None,
        "attitude_summary": None,
        "iteration_history": {
            "inner_iterations": int(selected["solution"].iterations),
            "outer_iterations": None,
            "accepted": None,
        },
        "path_constraint_bounds": {
            "component_acceleration_abs_max_m_s2": problem.config.max_component_acceleration
        },
        "validation": {
            "finite": True,
            "dimensions": {
                "states": list(selected["states"].shape),
                "controls": list(selected["controls"].shape),
            },
            "initial_inf": float(np.max(np.abs(selected["states"][0] - selected["initial"]))),
            "terminal_inf": selected["diagnostics"].terminal_error_inf,
            "dynamics_inf": selected["diagnostics"].dynamics_defect_inf,
            "path_inf": selected["diagnostics"].control_violation_inf,
            "continuous_time_inf": selected["diagnostics"].control_violation_inf,
            "canonical_primal": selected["audit"].primal,
            "canonical_dual": selected["audit"].dual,
            "canonical_natural": selected["audit"].natural,
            "dense_replay_terminal_inf": dense_terminal,
            "source_aggregate_reproduced": True,
        },
        "_required_transcription": required,
        "_required_replay": {index * substeps for index in required},
    }
    return metadata, source


def _rk4_pd3(model: Any, state: np.ndarray, control: np.ndarray, step: float) -> np.ndarray:
    k1 = model.dynamics(state, control)
    k2 = model.dynamics(state + 0.5 * step * k1, control)
    k3 = model.dynamics(state + 0.5 * step * k2, control)
    k4 = model.dynamics(state + step * k3, control)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def extract_pd3(archived: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from spacepdhcg.backends import PersistentClarabel
    from spacepdhcg.models import PoweredDescent3DOFModel
    from spacepdhcg.scvx import (
        ForcingRuleConfig,
        PoweredDescentOuterConfig,
        PoweredDescentSCvxSolver,
        TrustRegionConfig,
    )
    from spacepdhcg.transcription import PoweredDescent3DOFSubproblem, PoweredDescentSCvxConfig

    parameters = archived["parameters"]
    intervals = int(parameters["intervals"])
    step = 40.0 / intervals
    model = PoweredDescent3DOFModel()
    initial = np.asarray([20.0, -10.0, 120.0, 0.0, 0.0, -7.0, 2_000.0])
    initial += float(parameters["initial_dispersion_scales"]) * np.asarray(
        [20.0, -10.0, 30.0, 1.0, -0.5, -1.0, 0.0]
    )
    target_position = np.zeros(3)
    target_velocity = np.zeros(3)
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(
            intervals=intervals,
            step_seconds=step,
            trust_radius=1.0,
        ),
    )
    solver = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(
            max_iterations=8,
            convergence_tolerance=1.0e-3,
        ),
        forcing_config=ForcingRuleConfig(),
        trust_config=TrustRegionConfig(initial_radius=1.0),
    )
    result = solver.solve(initial, target_position, target_velocity)
    problem = subproblem.canonical(
        result.states,
        result.controls,
        initial,
        target_position,
        target_velocity,
    )
    polish_backend = PersistentClarabel(
        problem,
        tolerance=1.0e-8,
        iteration_limit=2_000,
        verbose=False,
    )
    polish = polish_backend.solve()
    audit = polish_backend.independent_residuals(polish.primal)
    quality = archived["quality"]
    assert_close(result.residual.dynamics, quality["dynamics_residual"], "P1-C dynamics residual")
    assert_close(result.residual.path, quality["path_residual"], "P1-C path residual")
    assert_close(result.residual.terminal, quality["terminal_residual"], "P1-C terminal residual")
    assert_close(
        result.path_diagnostics.maximum_violation,
        quality["continuous_time_violation"],
        "P1-C path violation",
    )
    assert_close(audit.primal, quality["canonical_primal_residual"], "P1-C polished primal")
    assert_close(audit.dual, quality["canonical_dual_residual"], "P1-C polished dual")

    substeps = 10
    replay_step = step / substeps
    dense = np.empty((intervals * substeps + 1, 7), dtype=np.float64)
    dense[0] = initial
    replay_controls = np.repeat(result.controls, substeps, axis=0)
    for index, control in enumerate(replay_controls):
        dense[index + 1] = _rk4_pd3(model, dense[index], control, replay_step)
    dense_path = model.path_diagnostics(dense, replay_controls)
    dense_terminal = float(
        max(
            np.max(np.abs(dense[-1, :3] - target_position)),
            np.max(np.abs(dense[-1, 3:6] - target_velocity)),
        )
    )
    times = np.arange(intervals + 1, dtype=np.float64) * step
    dense_times = np.arange(dense.shape[0], dtype=np.float64) * replay_step
    altitude = result.states[:, 2]
    horizontal = np.linalg.norm(result.states[:, :2], axis=1)
    glide_margin = model.config.glide_slope_tangent * altitude - horizontal
    required = {
        int(np.argmin(altitude)),
        int(np.argmin(glide_margin)),
        int(np.argmin(result.states[:, 6] - model.config.minimum_mass)),
    }
    sigma = result.controls[:, 3]
    tilt_margin = result.controls[:, 2] - model.config.tilt_cosine * sigma
    for index in {
        int(np.argmin(model.config.maximum_thrust - sigma)),
        int(np.argmin(tilt_margin)),
    }:
        required.update({index, index + 1})
    iteration_history = [
        {
            "iteration": record.iteration,
            "phase": record.phase.value,
            "requested_tolerance": record.requested_tolerance,
            "effective_tolerance": record.effective_tolerance,
            "solver_status": record.solver_status,
            "trust_radius_before": record.trust_radius_before,
            "trust_radius_after": record.trust_radius_after,
            "trust_action": record.trust_action,
            "agreement": record.agreement,
            "accepted": record.accepted,
            "restoration_accepted": record.restoration_accepted,
        }
        for record in result.iterations
    ]
    source = {
        "times": times,
        "states": result.states,
        "controls": result.controls,
        "dense_times": dense_times,
        "dense_states": dense,
    }
    metadata = {
        "trajectory_id": "p1c_pd3_cpu_019e151e",
        "family": "P1-C",
        "physical_family": "3-DoF powered descent",
        "source": {
            "commit": archived["repository_commit"],
            "campaign": "cpu-actual-c5a4991",
            "run_id": archived["coordinate_id"],
            "coordinate": archived["parameters"],
            "solver": archived["implementation"]["identifier"],
            "policy": "CPU SCvx; max_outer=8; convergence=1e-3; final Clarabel polish",
            "status": archived["disposition"],
        },
        "qualification": {
            "qualified": False,
            "label": "unqualified: polished CQP and retained nonlinear decisions differ",
        },
        "frame": "local-level inertial Cartesian",
        "position_units": "m",
        "time_units": "s",
        "initial_state": initial.tolist(),
        "terminal_target": [0.0, 0.0, 0.0, 0.0, 0.0, None],
        "state_order": ["x", "y", "z", "vx", "vy", "vz", "mass"],
        "control_order": ["thrust_x", "thrust_y", "thrust_z", "sigma"],
        "controls_summary": array_summary(
            result.controls, ["thrust_x", "thrust_y", "thrust_z", "sigma"]
        ),
        "mass_summary": {
            "initial": float(result.states[0, 6]),
            "final": float(result.states[-1, 6]),
            "minimum": float(np.min(result.states[:, 6])),
        },
        "attitude_summary": None,
        "iteration_history": iteration_history,
        "path_constraint_bounds": {
            "minimum_mass_kg": model.config.minimum_mass,
            "maximum_thrust_n": model.config.maximum_thrust,
            "maximum_tilt_radians": model.config.maximum_tilt_radians,
            "glide_slope_radians": model.config.glide_slope_radians,
            "minimum_altitude_m": 0.0,
        },
        "validation": {
            "finite": True,
            "dimensions": {
                "states": list(result.states.shape),
                "controls": list(result.controls.shape),
            },
            "initial_inf": float(np.max(np.abs(result.states[0] - initial))),
            "terminal_inf": result.residual.terminal,
            "dynamics_inf": result.residual.dynamics,
            "path_inf": result.residual.path,
            "continuous_time_inf": result.path_diagnostics.maximum_violation,
            "virtual_control_inf": quality["virtual_control_residual"],
            "polished_canonical_primal": audit.primal,
            "polished_canonical_dual": audit.dual,
            "polished_canonical_natural": audit.natural,
            "dense_replay_terminal_inf": dense_terminal,
            "dense_replay_path_inf": dense_path.maximum_violation,
            "source_aggregate_reproduced": True,
        },
        "_required_transcription": required,
        "_required_replay": {index * substeps for index in required},
    }
    return metadata, source


def compile_helper(source: Path, include_roots: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    include_arguments = [
        argument for root in include_roots for argument in ("-I", str(root / "cpp" / "include"))
    ]
    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            *include_arguments,
            str(source),
            "-o",
            str(output),
        ],
        check=True,
    )


def run_helper(binary: Path, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    value = json.loads(completed.stdout, parse_constant=_reject_constant)
    if not finite_tree(value):
        raise ValueError(f"helper {arguments[0]} emitted non-finite values")
    return value


def native_required(family: str, states: np.ndarray) -> set[int]:
    if family == "P1-D-pd6":
        altitude = states[:, 2]
        mass = states[:, 13]
        angular_rate = np.linalg.norm(states[:, 10:13], axis=1)
        quaternion_scalar = np.abs(states[:, 6])
        return {
            int(np.argmin(altitude)),
            int(np.argmin(mass)),
            int(np.argmax(angular_rate)),
            int(np.argmin(quaternion_scalar)),
        }
    radii = np.linalg.norm(states[:, :3], axis=1)
    return {
        int(np.argmin(radii)),
        int(np.argmax(radii)),
        int(np.argmin(states[:, 6])),
    }


def extract_native(
    archived: dict[str, Any],
    helper_payload: dict[str, Any],
    native_emitter: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters = archived["parameters"]
    if archived["family"] == "P1-D-pd6":
        native_arguments = [
            "pd6",
            str(parameters["intervals"]),
            str(parameters["attitude_dispersion_radians"]),
            str(parameters["angular_rate_dispersion"]),
            str(parameters["final_polish"]).lower(),
        ]
        state_order = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx",
            "wy",
            "wz",
            "mass",
        ]
        control_order = [
            "body_thrust_x",
            "body_thrust_y",
            "body_thrust_z",
            "torque_x",
            "torque_y",
            "torque_z",
            "sigma",
        ]
        physical_family = "6-DoF powered descent"
        identifier = "p1d_pd6_native_02863523"
        qualification = "unqualified: no host conic optimizer dual"
    else:
        native_arguments = [
            "low-thrust",
            str(parameters["intervals"]),
            str(parameters["trust_radii"]),
            str(parameters["transfer_classes"]),
        ]
        state_order = ["x", "y", "z", "vx", "vy", "vz", "mass"]
        control_order = ["thrust_x", "thrust_y", "thrust_z", "sigma"]
        physical_family = "low-thrust radius raise"
        identifier = "p1e_low_thrust_native_04fb7199"
        qualification = "unqualified: no host conic optimizer dual"
    emitter = subprocess.run(
        [str(native_emitter), *native_arguments],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    replay_metrics = json.loads(emitter.stdout, parse_constant=_reject_constant)
    for source_key, archive_key in (
        ("canonical_primal_residual", "canonical_primal_residual"),
        ("canonical_cone_residual", "canonical_cone_residual"),
        ("dynamics_residual", "dynamics_residual"),
        ("path_residual", "path_residual"),
        ("terminal_residual", "terminal_residual"),
        ("virtual_control_residual", "virtual_control_residual"),
    ):
        assert_close(
            replay_metrics[source_key],
            archived["quality"][archive_key],
            f"{archived['family']} {archive_key}",
        )
    states = np.asarray(helper_payload["states"], dtype=np.float64)
    controls = np.asarray(helper_payload["controls"], dtype=np.float64)
    times = np.asarray(helper_payload["node_times"], dtype=np.float64)
    dense_states = np.asarray(helper_payload["replay_states"], dtype=np.float64)
    dense_times = np.asarray(helper_payload["replay_times"], dtype=np.float64)
    if (
        states.shape[0] != int(parameters["intervals"]) + 1
        or controls.shape[0] + 1 != states.shape[0]
    ):
        raise ValueError(f"{archived['family']} helper dimensions are invalid")
    required = native_required(archived["family"], states)
    substeps = int(helper_payload["integration"]["replay_substeps"])
    model = helper_payload["model"]
    if archived["family"] == "P1-D-pd6":
        mass_index = 13
        attitude_summary = {
            "initial_quaternion": states[0, 6:10].tolist(),
            "final_quaternion": states[-1, 6:10].tolist(),
            "maximum_angular_rate": float(np.max(np.linalg.norm(states[:, 10:13], axis=1))),
        }
        constraints = {
            "minimum_mass_kg": model["minimum_mass"],
            "maximum_thrust_n": model["maximum_thrust"],
            "maximum_torque_nm": model["maximum_torque"],
            "maximum_angular_rate_rad_s": model["maximum_angular_rate"],
            "maximum_tilt_radians": model["maximum_tilt_radians"],
            "glide_slope_radians": model["glide_slope_radians"],
            "minimum_altitude_m": 0.0,
        }
    else:
        mass_index = 6
        attitude_summary = None
        constraints = {
            "minimum_mass_kg": model["minimum_mass"],
            "maximum_thrust": model["maximum_thrust"],
            "minimum_radius_km": model["minimum_radius"],
        }
    source = {
        "times": times,
        "states": states,
        "controls": controls,
        "dense_times": dense_times,
        "dense_states": dense_states,
    }
    metadata = {
        "trajectory_id": identifier,
        "family": archived["family"].split("-", 2)[0] + "-" + archived["family"].split("-", 2)[1],
        "physical_family": physical_family,
        "source": {
            "commit": archived["repository_commit"],
            "campaign": "cpu-actual-c5a4991",
            "run_id": archived["coordinate_id"],
            "coordinate": archived["parameters"],
            "solver": archived["implementation"]["identifier"],
            "policy": "authoritative native reference construction and nonlinear replay",
            "status": archived["disposition"],
        },
        "qualification": {"qualified": False, "label": qualification},
        "frame": helper_payload["frame"],
        "position_units": helper_payload["position_units"],
        "time_units": helper_payload["time_units"],
        "initial_state": states[0].tolist(),
        "terminal_target": states[-1].tolist(),
        "state_order": state_order,
        "control_order": control_order,
        "controls_summary": array_summary(controls, control_order),
        "mass_summary": {
            "initial": float(states[0, mass_index]),
            "final": float(states[-1, mass_index]),
            "minimum": float(np.min(states[:, mass_index])),
        },
        "attitude_summary": attitude_summary,
        "iteration_history": {
            "outer_iterations": archived["work"]["outer_iterations"],
            "inner_iterations": archived["work"]["inner_iterations"],
            "accepted_steps": archived["work"]["accepted_steps"],
            "rejected_steps": archived["work"]["rejected_steps"],
        },
        "path_constraint_bounds": constraints,
        "validation": {
            "finite": bool(np.all(np.isfinite(states)) and np.all(np.isfinite(dense_states))),
            "dimensions": {
                "states": list(states.shape),
                "controls": list(controls.shape),
            },
            "initial_inf": 0.0,
            "terminal_inf": archived["quality"]["terminal_residual"],
            "dynamics_inf": archived["quality"]["dynamics_residual"],
            "path_inf": archived["quality"]["path_residual"],
            "continuous_time_inf": archived["quality"]["continuous_time_violation"],
            "transcription_physical_path_inf": helper_payload["validation"][
                "transcription_path_violation"
            ],
            "dense_replay_physical_path_inf": helper_payload["validation"]["replay_path_violation"],
            "dense_replay_terminal_inf": helper_payload["validation"]["replay_terminal_inf"],
            "source_aggregate_reproduced": True,
        },
        "_required_transcription": required,
        "_required_replay": {index * substeps for index in required},
    }
    return metadata, source


def extract_lambert(
    g7_repository: Path,
    helper_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = g7_repository / "cpp/cuda/tests/orbitweaver_gpu_test.cu"
    log_path = g7_repository / "build/g7-cuda-release/Testing/Temporary/LastTest.log"
    log = log_path.read_text(encoding="utf-8")
    if "Testing: orbitweaver_gpu_test" not in log or "Test Passed." not in log:
        raise ValueError("archived one-GPU Lambert parity test is not passing")
    states = np.asarray(helper_payload["states"], dtype=np.float64)
    dense_states = np.asarray(helper_payload["replay_states"], dtype=np.float64)
    times = np.asarray(helper_payload["node_times"], dtype=np.float64)
    dense_times = np.asarray(helper_payload["replay_times"], dtype=np.float64)
    radii = np.linalg.norm(dense_states[:, :3], axis=1)
    required_dense = {
        int(np.argmin(radii)),
        int(np.argmax(radii)),
        int(np.argmax(dense_states[:, 2])),
        int(np.argmin(dense_states[:, 2])),
    }
    source = {
        "times": times,
        "states": states,
        "controls": np.empty((0, 0), dtype=np.float64),
        "dense_times": dense_times,
        "dense_states": dense_states,
    }
    metadata = {
        "trajectory_id": "p2_lambert_g7_gpu_request_17_cpu_replay",
        "family": "P2",
        "physical_family": "zero-revolution Lambert transfer",
        "source": {
            "commit": subprocess.run(
                ["git", "-C", str(g7_repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "campaign": "OrbitWeaver G7 actual one-GPU parity",
            "run_id": "orbitweaver_gpu_test/request-17",
            "coordinate": {
                "departure_position_m": helper_payload["departure_position"],
                "arrival_position_m": helper_payload["arrival_position"],
                "time_of_flight_s": helper_payload["time_of_flight"],
                "maximum_revolutions": 1,
                "include_short_way": True,
                "include_long_way": False,
            },
            "solver": "GPU fixed-slot Lambert kernel parity against CPU universal-variable solver",
            "policy": (
                "one GPU correctness test; no performance measurement; CPU state-history replay"
            ),
            "status": "component-qualified",
            "source_test_sha256": file_sha256(source_path),
            "source_test_log_sha256": file_sha256(log_path),
        },
        "qualification": {
            "qualified": True,
            "label": "GPU Lambert component parity passed; plotted history is labelled CPU replay",
        },
        "frame": helper_payload["frame"],
        "position_units": helper_payload["position_units"],
        "time_units": helper_payload["time_units"],
        "initial_state": states[0].tolist(),
        "terminal_target": states[-1].tolist(),
        "state_order": ["x", "y", "z", "vx", "vy", "vz"],
        "control_order": [],
        "controls_summary": None,
        "mass_summary": None,
        "attitude_summary": None,
        "iteration_history": {
            "lambert_iterations": helper_payload["iterations"],
            "universal_parameter": helper_payload["universal_parameter"],
            "time_of_flight_residual": helper_payload["time_of_flight_residual"],
        },
        "path_constraint_bounds": {"ballistic_two_body": True},
        "validation": {
            "finite": bool(np.all(np.isfinite(states)) and np.all(np.isfinite(dense_states))),
            "dimensions": {"endpoint_states": list(states.shape), "controls": [0, 0]},
            "initial_inf": 0.0,
            "terminal_position_inf": helper_payload["validation"]["terminal_position_inf"],
            "terminal_velocity_inf": helper_payload["validation"]["terminal_velocity_inf"],
            "time_of_flight_residual": helper_payload["time_of_flight_residual"],
            "gpu_cpu_velocity_parity_bound": 1.0e-7,
            "gpu_test_passed": True,
        },
        "_required_transcription": {0, 1},
        "_required_replay": required_dense,
    }
    return metadata, source


def plot_preview(record: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(record["replay"]["points_txyz"], dtype=np.float64)
    xyz = points[:, 1:4]
    units = record["position_units"]
    label = record["qualification"]["label"]
    caption = (
        f"{record['source']['campaign']} | {record['source']['run_id']} | "
        f"{record['source']['commit'][:12]}"
    )
    figure = plt.figure(figsize=(11.0, 8.5), constrained_layout=True)
    axes = [
        figure.add_subplot(2, 2, 1),
        figure.add_subplot(2, 2, 2),
        figure.add_subplot(2, 2, 3),
        figure.add_subplot(2, 2, 4, projection="3d"),
    ]
    projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    for axis, (left, right, left_name, right_name) in zip(axes[:3], projections, strict=True):
        axis.plot(xyz[:, left], xyz[:, right], color="#2463eb", linewidth=1.5, label="dense replay")
        axis.scatter(xyz[0, left], xyz[0, right], color="#16a34a", marker="o", label="start")
        axis.scatter(xyz[-1, left], xyz[-1, right], color="#dc2626", marker="x", label="end")
        axis.set_xlabel(f"{left_name} [{units}]")
        axis.set_ylabel(f"{right_name} [{units}]")
        axis.set_title(f"{left_name}{right_name} orthographic")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
    perspective = axes[3]
    perspective.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#2463eb", linewidth=1.5)
    perspective.scatter(*xyz[0], color="#16a34a", marker="o", label="start")
    perspective.scatter(*xyz[-1], color="#dc2626", marker="x", label="end")
    perspective.set_xlabel(f"X [{units}]")
    perspective.set_ylabel(f"Y [{units}]")
    perspective.set_zlabel(f"Z [{units}]")
    perspective.set_title("Perspective")
    axes[0].legend(loc="best", fontsize=8)
    perspective.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{record['family']} — {record['physical_family']}\n{label}\n{caption}",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "SpacePDHCG verified trajectory extractor"}
    figure.savefig(output.with_suffix(".png"), dpi=160, metadata=metadata)
    figure.savefig(
        output.with_suffix(".pdf"),
        metadata={
            **metadata,
            "CreationDate": None,
            "ModDate": None,
            "Title": record["trajectory_id"],
        },
    )
    plt.close(figure)


def inventory_sources(
    g4_archive: Path,
    cpu_archive: Path,
    selected_results: dict[str, dict[str, Any]],
    g7_repository: Path,
) -> list[dict[str, Any]]:
    with tarfile.open(g4_archive, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    inventory = [
        {
            "source": str(g4_archive),
            "families": ["P1-A", "P1-C", "P1-D", "P1-E"],
            "record_kind": "scalar/component CQP and replay metrics",
            "contains_actual_state_history": False,
            "classification": (
                "P1-A is explicitly non-trajectory CQP evidence; all G4 logs retain scalar "
                "trajectory/replay differences but no traceable state arrays"
            ),
            "member_count": len(members),
            "sha256": file_sha256(g4_archive),
        }
    ]
    for family, result in selected_results.items():
        inventory.append(
            {
                "source": str(
                    cpu_archive / "runs" / family / result["coordinate_id"] / "result.json"
                ),
                "families": [family],
                "record_kind": "aggregate scalar result",
                "contains_actual_state_history": has_traceable_state_arrays(result),
                "classification": (
                    "authoritative run metrics; path arrays are absent and therefore recreated "
                    "only by deterministic execution of the exact source commit"
                ),
            }
        )
    inventory.extend(
        [
            {
                "source": str(g7_repository / "cpp/cuda/tests/orbitweaver_gpu_test.cu"),
                "families": ["P2 Lambert"],
                "record_kind": "actual one-GPU component correctness test",
                "contains_actual_state_history": False,
                "classification": (
                    "actual GPU Lambert velocity parity evidence; no GPU path array was archived"
                ),
            },
            {
                "source": str(g7_repository / "cpp/tests/high_fidelity_certification_smoke.cpp"),
                "families": ["P2 certified"],
                "record_kind": "synthetic coast certification fixture",
                "contains_actual_state_history": True,
                "classification": (
                    "traceable but synthetic/trivial constant-state fixture; excluded as a "
                    "representative physical trajectory"
                ),
            },
        ]
    )
    return inventory


def data_dictionary() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "point_encoding": {
            "points_txyz": "[time, x, y, z] exact selected source values",
            "selected_indices": "indices into the full generated evidence array",
            "original_sha256": "SHA-256 of canonical JSON containing full times and full states",
            "decimation": (
                "deterministic endpoint/extrema/constraint-near preservation plus evenly spaced "
                "exact samples; no interpolation"
            ),
        },
        "qualification": {
            "qualified": "whether the source evidence qualifies under its own campaign policy",
            "label": "human-readable scope and caveat",
        },
        "trajectory_semantics": {
            "transcription": "source solver/reference nodes",
            "replay": "independently integrated dense dynamics; never visual interpolation",
            "frame": "record-specific coordinate reference; records must not share a common scale",
        },
    }


def write_raw_record(output: Path, metadata: dict[str, Any], source: dict[str, Any]) -> str:
    raw = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": metadata["trajectory_id"],
        "times": source["times"].tolist(),
        "states": source["states"].tolist(),
        "controls": source["controls"].tolist(),
        "dense_replay_times": source["dense_times"].tolist(),
        "dense_replay_states": source["dense_states"].tolist(),
    }
    path = output / "raw" / f"{metadata['trajectory_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(raw))
    return file_sha256(path)


def build_compact_record(metadata: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    required_transcription = metadata.pop("_required_transcription")
    required_replay = metadata.pop("_required_replay")
    metadata["transcription"] = compact_points(
        source["times"],
        source["states"],
        256,
        required_transcription,
    )
    metadata["replay"] = compact_points(
        source["dense_times"],
        source["dense_states"],
        512,
        required_replay,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-source", type=Path, required=True)
    parser.add_argument("--cpu-archive", type=Path, required=True)
    parser.add_argument("--native-emitter", type=Path, required=True)
    parser.add_argument("--g4-archive", type=Path, required=True)
    parser.add_argument("--g7-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.output.exists():
        shutil.rmtree(arguments.output)
    arguments.output.mkdir(parents=True)
    sys.path.insert(0, str(arguments.cpu_source / "src"))

    selected = {
        "P1-B-hcw": "040d020dc7841ff58843c70e",
        "P1-C-pd3": "019e151e7f18063c638f0052",
        "P1-D-pd6": "02863523bd46dd1a7a2ac5d3",
        "P1-E-low-thrust": "04fb7199e489e7259fa31f7d",
    }
    selected_results: dict[str, dict[str, Any]] = {}
    checksum_paths = ["summary.json", "validation-summary.json", "coordinate-index.json"]
    for family, coordinate in selected.items():
        prefix = f"runs/{family}/{coordinate}"
        checksum_paths.extend(
            [
                f"{prefix}/coordinate.json",
                f"{prefix}/result.json",
                f"{prefix}/stdout.log",
                f"{prefix}/stderr.log",
            ]
        )
        selected_results[family] = json_load(arguments.cpu_archive / prefix / "result.json")
    source_checksum_validation = verify_source_checksums(
        arguments.cpu_archive,
        checksum_paths,
    )
    archive_validation = json_load(arguments.cpu_archive / "validation-summary.json")
    if not (
        archive_validation["missing"] == 0
        and archive_validation["duplicates"] == 0
        and archive_validation["schema_errors"] == 0
    ):
        raise ValueError("CPU source archive validation summary is not clean")

    helper_source = Path(__file__).resolve().parents[2] / "cpp/tools/trajectory_history_emitter.cpp"
    cpu_helper = arguments.output / "bin/trajectory_history_cpu"
    g7_helper = arguments.output / "bin/trajectory_history_g7"
    compile_helper(helper_source, [arguments.cpu_source], cpu_helper)
    compile_helper(
        helper_source,
        [arguments.g7_repository, arguments.cpu_source],
        g7_helper,
    )

    extracted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    extracted.append(extract_hcw(arguments.cpu_source, selected_results["P1-B-hcw"]))
    extracted.append(extract_pd3(selected_results["P1-C-pd3"]))
    p1d_payload = run_helper(cpu_helper, ["pd6", "100", "0.2", "0.2", "10"])
    extracted.append(
        extract_native(
            selected_results["P1-D-pd6"],
            p1d_payload,
            arguments.native_emitter,
        )
    )
    p1e_payload = run_helper(cpu_helper, ["low-thrust", "2000", "radius_raise", "2"])
    extracted.append(
        extract_native(
            selected_results["P1-E-low-thrust"],
            p1e_payload,
            arguments.native_emitter,
        )
    )
    lambert_payload = run_helper(g7_helper, ["lambert", "7200"])
    extracted.append(extract_lambert(arguments.g7_repository, lambert_payload))

    compact_records: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for metadata, source in extracted:
        if not (
            np.all(np.isfinite(source["times"]))
            and np.all(np.isfinite(source["states"]))
            and np.all(np.isfinite(source["dense_times"]))
            and np.all(np.isfinite(source["dense_states"]))
        ):
            raise ValueError(f"{metadata['trajectory_id']} contains non-finite path values")
        raw_hashes[metadata["trajectory_id"]] = write_raw_record(
            arguments.output,
            metadata,
            source,
        )
        record = build_compact_record(metadata, source)
        record["raw_evidence_sha256"] = raw_hashes[record["trajectory_id"]]
        compact_records.append(record)

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "title": "Verified SpacePDHCG trajectory visualisation evidence",
        "generated_by_commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "prohibitions": {
            "aggregate_metric_path_fabrication": True,
            "gpu_execution_during_extraction": True,
            "visual_interpolation_included": False,
        },
        "trajectories": compact_records,
    }
    dataset_path = arguments.output / "spacepdhcg_verified_trajectories.compact.json"
    dataset_path.write_bytes(canonical_bytes(dataset))

    inventory = inventory_sources(
        arguments.g4_archive,
        arguments.cpu_archive,
        selected_results,
        arguments.g7_repository,
    )
    exclusions = [
        {
            "family": "P1-A",
            "status": "excluded",
            "reason": "non-trajectory CQP evidence; no physical state history",
        },
        {
            "family": "P1-F",
            "status": "excluded",
            "reason": "robust aggregate scenario metrics contain no traceable state arrays",
        },
        {
            "family": "P2-A/P2-B/P2-C CPU campaign records",
            "status": "excluded",
            "reason": "component-contract aggregates only; no state arrays",
        },
        {
            "family": "P2 coarse/refined/route",
            "status": "missing",
            "reason": (
                "no archived real trajectory state history found; "
                "synthetic echo/pipeline fixtures rejected"
            ),
        },
        {
            "family": "P2 certified coast smoke",
            "status": "excluded",
            "reason": "traceable but synthetic constant-state fixture",
        },
        {
            "family": "P2-D/P2-E",
            "status": "excluded",
            "reason": "CPU campaign marks these coordinates unsupported",
        },
    ]
    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": dataset_path.name,
        "dataset_sha256": file_sha256(dataset_path),
        "source_checksum_validation": source_checksum_validation,
        "source_archive_validation": archive_validation,
        "source_inventory": inventory,
        "trajectory_validation": [
            {
                "trajectory_id": record["trajectory_id"],
                "qualification": record["qualification"],
                "frame": record["frame"],
                "transcription_points": record["transcription"]["original_point_count"],
                "transcription_visual_points": record["transcription"]["point_count"],
                "replay_points": record["replay"]["original_point_count"],
                "replay_visual_points": record["replay"]["point_count"],
                "validation": record["validation"],
                "raw_evidence_sha256": record["raw_evidence_sha256"],
            }
            for record in compact_records
        ],
        "excluded_or_missing": exclusions,
        "checks": {
            "all_source_checksums_verified": True,
            "all_arrays_finite": True,
            "all_dimensions_valid": True,
            "all_endpoints_preserved_by_decimation": all(
                record["transcription"]["selected_indices"][0] == 0
                and record["transcription"]["selected_indices"][-1]
                == record["transcription"]["original_point_count"] - 1
                and record["replay"]["selected_indices"][0] == 0
                and record["replay"]["selected_indices"][-1]
                == record["replay"]["original_point_count"] - 1
                for record in compact_records
            ),
            "gpu_workloads_run": False,
            "visual_interpolation_used": False,
        },
    }
    report_path = arguments.output / "validation_report.json"
    report_path.write_bytes(canonical_bytes(validation_report))
    dictionary_path = arguments.output / "data_dictionary.json"
    dictionary_path.write_bytes(canonical_bytes(data_dictionary()))

    for record in compact_records:
        plot_preview(record, arguments.output / "previews" / record["trajectory_id"])

    checksums = []
    for path in sorted(
        item for item in arguments.output.rglob("*") if item.is_file() and "bin" not in item.parts
    ):
        checksums.append(
            {
                "path": str(path.relative_to(arguments.output)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    (arguments.output / "checksums.json").write_bytes(
        canonical_bytes({"schema_version": SCHEMA_VERSION, "files": checksums})
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "sha256": file_sha256(dataset_path),
                "trajectory_count": len(compact_records),
                "validation_report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
