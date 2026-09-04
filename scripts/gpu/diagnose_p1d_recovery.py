#!/usr/bin/env python3
"""Compare exact dumped trajectory CQPs and diagnose recovery conditioning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.gpu.diagnose_g3_pd3 import cqp_hash, load_dump, load_persistent_solution
from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.backends.qoco_gpu import QOCOGPU, canonical_primal_residual
from spacepdhcg.cqp import CanonicalCQP, ConeKind, CQPSolution

FloatArray = NDArray[np.float64]


def _maximum(values: FloatArray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _project_soc(vector: FloatArray) -> FloatArray:
    spatial = vector[:-1]
    radius = float(vector[-1])
    norm = float(np.linalg.norm(spatial))
    if norm <= radius:
        return vector.copy()
    if norm <= -radius:
        return np.zeros_like(vector)
    return np.concatenate((0.5 * (1.0 + radius / norm) * spatial, [0.5 * (norm + radius)]))


def _project_cones(values: FloatArray, problem: CanonicalCQP) -> FloatArray:
    projected = values.copy()
    for cone in problem.structure.affine_cones:
        if cone.kind is not ConeKind.SECOND_ORDER:
            raise NotImplementedError(f"diagnostic only supports SOC, not {cone.kind}")
        projected[cone.start : cone.stop] = _project_soc(projected[cone.start : cone.stop])
    return projected


def natural_residual(
    problem: CanonicalCQP,
    solution: CQPSolution,
    *,
    clarabel_dual: bool,
) -> dict[str, float]:
    structure = problem.structure
    values = problem.values
    primal = np.asarray(solution.primal, dtype=np.float64)
    dual = np.asarray(solution.dual, dtype=np.float64).copy()
    if clarabel_dual:
        dual[structure.n_constraints :] *= -1.0
    quadratic = structure.quadratic.matrix(values.quadratic)
    scalar = structure.constraint.matrix(values.constraint)
    affine = structure.affine_cone.matrix(values.affine_cone)
    scalar_value = np.asarray(scalar @ primal)
    cone_value = np.asarray(affine @ primal + values.affine_offset)
    stationarity = np.asarray(
        quadratic @ primal
        + values.linear
        + scalar.T @ dual[: structure.n_constraints]
        + affine.T @ dual[structure.n_constraints :]
    )
    stationarity_natural = primal - np.clip(
        primal - stationarity,
        values.variable_lower,
        values.variable_upper,
    )
    scalar_natural = scalar_value - np.clip(
        scalar_value + dual[: structure.n_constraints],
        values.lower,
        values.upper,
    )
    cone_natural = cone_value - _project_cones(
        cone_value + dual[structure.n_constraints :],
        problem,
    )
    pieces = {
        "stationarity": _maximum(stationarity_natural),
        "scalar": _maximum(scalar_natural),
        "cone": _maximum(cone_natural),
    }
    pieces["natural"] = max(pieces.values())
    return pieces


def active_jacobian(problem: CanonicalCQP, primal: FloatArray) -> dict[str, float | int]:
    structure = problem.structure
    values = problem.values
    scalar = structure.constraint.matrix(values.constraint)
    affine = structure.affine_cone.matrix(values.affine_cone)
    scalar_value = np.asarray(scalar @ primal)
    cone_value = np.asarray(affine @ primal + values.affine_offset)
    rows: list[FloatArray] = []
    equality_rows = 0
    scalar_active = 0
    cone_active = 0
    for row, (lower, upper, value) in enumerate(
        zip(values.lower, values.upper, scalar_value, strict=True)
    ):
        equality = np.isfinite(lower) and np.isfinite(upper) and lower == upper
        scale = max(
            1.0,
            abs(float(value)),
            abs(float(lower)) if np.isfinite(lower) else 0.0,
            abs(float(upper)) if np.isfinite(upper) else 0.0,
        )
        active = (
            equality
            or (np.isfinite(lower) and value - lower <= 1.0e-6 * scale)
            or (np.isfinite(upper) and upper - value <= 1.0e-6 * scale)
        )
        if active:
            rows.append(np.asarray(scalar.getrow(row).toarray()).ravel())
            equality_rows += int(equality)
            scalar_active += int(not equality)
    for cone in structure.affine_cones:
        segment = cone_value[cone.start : cone.stop]
        norm = float(np.linalg.norm(segment[:-1]))
        scale = max(1.0, _maximum(segment))
        if norm > 1.0e-12 and segment[-1] - norm <= 1.0e-6 * scale:
            normal = np.concatenate((segment[:-1] / norm, [-1.0]))
            rows.append(np.asarray(normal @ affine[cone.start : cone.stop, :]).ravel())
            cone_active += 1
    matrix = np.vstack(rows)
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = singular[0] * max(matrix.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > threshold))
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "equality_rows": equality_rows,
        "scalar_active": scalar_active,
        "cone_active": cone_active,
        "rank": rank,
        "row_rank_deficiency": int(matrix.shape[0] - rank),
        "sigma_max": float(singular[0]),
        "sigma_min": float(singular[-1]),
        "sigma_min_rank": float(singular[rank - 1]),
        "condition_rank": float(singular[0] / singular[rank - 1]),
    }


def run(
    dump: Path,
    qoco_library: Path | None,
    persistent_output: Path | None,
) -> dict[str, object]:
    problem = load_dump(dump)
    structure = problem.structure
    clarabel = PersistentClarabel(
        problem,
        tolerance=1.0e-10,
        iteration_limit=2_000,
    ).solve()
    result: dict[str, object] = {
        "cqp_sha256": cqp_hash(problem),
        "dimensions": {
            "variables": structure.n_variables,
            "scalar_rows": structure.n_constraints,
            "affine_rows": structure.n_affine_constraints,
            "cones": len(structure.affine_cones),
        },
        "clarabel": {
            "status": clarabel.status,
            "iterations": clarabel.iterations,
            "reported_primal": clarabel.primal_residual,
            "reported_dual": clarabel.dual_residual,
            "objective": clarabel.objective,
            "independent": natural_residual(problem, clarabel, clarabel_dual=True),
        },
        "active_jacobian_at_clarabel": active_jacobian(problem, clarabel.primal),
    }
    if persistent_output is not None:
        persistent = load_persistent_solution(persistent_output)
        handoff_residual = canonical_primal_residual(problem, persistent.primal)
        result["persistent_cuda"] = {
            "independent": natural_residual(
                problem,
                persistent,
                clarabel_dual=False,
            ),
            "active_jacobian": active_jacobian(problem, persistent.primal),
            "qoco_handoff": {
                "canonical_primal_residual": handoff_residual,
                "qualification_tolerance": 1.0e-6,
                "qualified": handoff_residual <= 1.0e-6,
                "disposition": (
                    "qualified_primal_handoff"
                    if handoff_residual <= 1.0e-6
                    else "rejected_use_pure_gpu_ipm_label"
                ),
            },
        }
    if qoco_library is not None:
        with QOCOGPU(
            problem,
            library_path=qoco_library,
            tolerance=1.0e-8,
            iteration_limit=500,
        ) as qoco:
            qoco_solution = qoco.solve()
            if qoco.last_report is None:
                raise AssertionError("QOCO returned no run report")
            result["qoco_gpu"] = {
                "status": qoco_solution.status,
                "iterations": qoco_solution.iterations,
                "objective": qoco_solution.objective,
                "reported_primal": qoco_solution.primal_residual,
                "reported_dual": qoco_solution.dual_residual,
                "native_primal": qoco.last_report.native_primal_residual,
                "native_dual": qoco.last_report.native_dual_residual,
                "canonical": asdict(qoco.last_report.canonical_residuals),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--qoco-library", type=Path)
    parser.add_argument("--persistent-output", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run(
        arguments.dump,
        arguments.qoco_library,
        arguments.persistent_output,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
