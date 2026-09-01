"""Independent G1 correctness expansion for pinned one-shot PDHCG.

The upstream Python API reports duals in its native ``[dual_A, dual_F]`` order.
Those values are the negative normal-cone multipliers used by the canonical
residual in ``docs/INEXACT_SCVX_THEORY.md``.  This module converts that sign
explicitly and recomputes all quality quantities from the returned vectors and
the immutable SpacePDHCG CQP data; no upstream residual buffer is reused.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.backends import PDHCGOneShot, PersistentClarabel
from spacepdhcg.cqp import CanonicalCQP, ConeBlock, ConeKind, CQPSolution
from spacepdhcg.models import (
    CWRendezvousConfig,
    CWRendezvousProblem,
    ThrustConstraint,
)

FloatArray = NDArray[np.float64]


class StartMode(StrEnum):
    """Initial iterate supplied to one-shot PDHCG."""

    COLD = "cold"
    PRIMAL = "primal"
    PRIMAL_DUAL = "primal-dual"


@dataclass(frozen=True, slots=True)
class CanonicalQuality:
    """Independently recomputed canonical and trajectory quality."""

    objective_recomputed: float
    objective_reference: float
    objective_gap_relative: float
    reported_objective_error_relative: float
    scalar_primal_violation_inf: float
    variable_primal_violation_inf: float
    cone_primal_violation_inf: float
    stationarity_residual_inf: float
    scalar_natural_residual_inf: float
    cone_natural_residual_inf: float
    natural_residual_inf: float
    relative_natural_residual_inf: float
    initial_error_inf: float
    terminal_error_inf: float
    dynamics_defect_inf: float
    control_violation_inf: float

    @property
    def maximum_trajectory_violation(self) -> float:
        return max(
            self.initial_error_inf,
            self.terminal_error_inf,
            self.dynamics_defect_inf,
            self.control_violation_inf,
        )


def _maximum_absolute(vector: FloatArray) -> float:
    return 0.0 if vector.size == 0 else float(np.max(np.abs(vector)))


def _project_soc(vector: FloatArray) -> FloatArray:
    """Project native PDHCG SOC coordinates ``[v..., t]`` onto the cone."""

    value = np.asarray(vector, dtype=np.float64)
    spatial = value[:-1]
    radius = float(value[-1])
    norm = float(np.linalg.norm(spatial))
    if norm <= radius:
        return value.copy()
    if norm <= -radius:
        return np.zeros_like(value)
    projected = np.empty_like(value)
    scale = 0.5 * (1.0 + radius / norm)
    projected[:-1] = scale * spatial
    projected[-1] = 0.5 * (norm + radius)
    return projected


def _project_cones(
    vector: FloatArray,
    cones: tuple[ConeBlock, ...],
) -> FloatArray:
    projected = np.asarray(vector, dtype=np.float64).copy()
    for cone in cones:
        segment = projected[cone.start : cone.stop]
        if cone.kind is ConeKind.SECOND_ORDER:
            projected[cone.start : cone.stop] = _project_soc(segment)
        else:
            raise NotImplementedError(
                f"G1 HCW quality currently supports SOC blocks, not {cone.kind}"
            )
    return projected


def _finite_bound_scale(lower: FloatArray, upper: FloatArray) -> float:
    finite = np.concatenate((lower[np.isfinite(lower)], upper[np.isfinite(upper)]))
    return max(1.0, _maximum_absolute(finite))


def evaluate_pdhcg_quality(
    problem: CWRendezvousProblem,
    canonical: CanonicalCQP,
    solution: CQPSolution,
    *,
    initial_state: FloatArray,
    target_state: FloatArray,
    reference_objective: float,
) -> CanonicalQuality:
    """Recompute the canonical natural residual and HCW checks independently."""

    structure = canonical.structure
    values = canonical.values
    primal = np.asarray(solution.primal, dtype=np.float64)
    raw_dual = np.asarray(solution.dual, dtype=np.float64)
    if primal.shape != (structure.n_variables,):
        raise ValueError("solution primal has the wrong shape")
    if raw_dual.shape != (structure.n_duals,):
        raise ValueError("solution dual has the wrong shape")
    if not np.all(np.isfinite(primal)) or not np.all(np.isfinite(raw_dual)):
        raise ValueError("solution vectors must be finite")

    quadratic = structure.quadratic.matrix(values.quadratic)
    scalar = structure.constraint.matrix(values.constraint)
    affine = (
        None if structure.affine_cone is None else structure.affine_cone.matrix(values.affine_cone)
    )

    # PDHCG exposes dual-cone values.  The natural-residual definition uses
    # normal-cone multipliers, which have the opposite sign.
    normal_dual = -raw_dual
    scalar_dual = normal_dual[: structure.n_constraints]
    cone_dual = normal_dual[structure.n_constraints :]

    scalar_value = np.asarray(scalar @ primal, dtype=np.float64)
    scalar_projection = np.clip(scalar_value, values.lower, values.upper)
    scalar_violation = scalar_value - scalar_projection
    scalar_natural = scalar_value - np.clip(
        scalar_value + scalar_dual,
        values.lower,
        values.upper,
    )

    variable_projection = np.clip(
        primal,
        values.variable_lower,
        values.variable_upper,
    )
    variable_violation = primal - variable_projection

    cone_value = np.empty(0, dtype=np.float64)
    cone_violation = np.empty(0, dtype=np.float64)
    cone_natural = np.empty(0, dtype=np.float64)
    cone_adjoint = np.zeros(structure.n_variables, dtype=np.float64)
    if affine is not None:
        cone_value = np.asarray(affine @ primal + values.affine_offset, dtype=np.float64)
        cone_projection = _project_cones(cone_value, structure.affine_cones)
        cone_violation = cone_value - cone_projection
        cone_natural = cone_value - _project_cones(
            cone_value + cone_dual,
            structure.affine_cones,
        )
        cone_adjoint = np.asarray(affine.T @ cone_dual, dtype=np.float64)

    gradient = np.asarray(quadratic @ primal + values.linear, dtype=np.float64)
    scalar_adjoint = np.asarray(scalar.T @ scalar_dual, dtype=np.float64)
    stationarity = gradient + scalar_adjoint + cone_adjoint
    stationarity_projection = primal - np.clip(
        primal - stationarity,
        values.variable_lower,
        values.variable_upper,
    )

    objective = float(0.5 * primal @ (quadratic @ primal) + values.linear @ primal)
    objective_scale = max(1.0, abs(float(reference_objective)))
    objective_gap = abs(objective - float(reference_objective)) / objective_scale
    reported_objective_error = abs(objective - float(solution.objective)) / max(
        1.0,
        abs(objective),
    )

    stationarity_scale = max(
        1.0,
        _maximum_absolute(gradient),
        _maximum_absolute(scalar_adjoint),
        _maximum_absolute(cone_adjoint),
    )
    row_scale = max(
        _finite_bound_scale(values.lower, values.upper),
        _maximum_absolute(scalar_value),
        _maximum_absolute(scalar_dual),
        _maximum_absolute(cone_value),
        _maximum_absolute(cone_dual),
    )
    stationarity_residual = _maximum_absolute(stationarity_projection)
    scalar_natural_residual = _maximum_absolute(scalar_natural)
    cone_natural_residual = _maximum_absolute(cone_natural)
    natural_residual = max(
        stationarity_residual,
        scalar_natural_residual,
        cone_natural_residual,
    )
    relative_natural = max(
        stationarity_residual / stationarity_scale,
        scalar_natural_residual / row_scale,
        cone_natural_residual / row_scale,
    )

    trajectory = problem.diagnostics(primal, initial_state, target_state)
    return CanonicalQuality(
        objective_recomputed=objective,
        objective_reference=float(reference_objective),
        objective_gap_relative=objective_gap,
        reported_objective_error_relative=reported_objective_error,
        scalar_primal_violation_inf=_maximum_absolute(scalar_violation),
        variable_primal_violation_inf=_maximum_absolute(variable_violation),
        cone_primal_violation_inf=_maximum_absolute(cone_violation),
        stationarity_residual_inf=stationarity_residual,
        scalar_natural_residual_inf=scalar_natural_residual,
        cone_natural_residual_inf=cone_natural_residual,
        natural_residual_inf=natural_residual,
        relative_natural_residual_inf=relative_natural,
        initial_error_inf=trajectory.initial_error_inf,
        terminal_error_inf=trajectory.terminal_error_inf,
        dynamics_defect_inf=trajectory.dynamics_defect_inf,
        control_violation_inf=trajectory.control_violation_inf,
    )


def _acceptance_threshold(tolerance: float) -> float:
    return max(1.0e-6, 100.0 * tolerance)


def _qualify(quality: CanonicalQuality, tolerance: float) -> None:
    threshold = _acceptance_threshold(tolerance)
    quantities = {
        "objective_gap_relative": quality.objective_gap_relative,
        "reported_objective_error_relative": quality.reported_objective_error_relative,
        "scalar_primal_violation_inf": quality.scalar_primal_violation_inf,
        "variable_primal_violation_inf": quality.variable_primal_violation_inf,
        "cone_primal_violation_inf": quality.cone_primal_violation_inf,
        "relative_natural_residual_inf": quality.relative_natural_residual_inf,
        "trajectory_violation": quality.maximum_trajectory_violation,
    }
    failed = {name: value for name, value in quantities.items() if value > threshold}
    if failed:
        raise RuntimeError(f"independent G1 checks exceed {threshold:.3e}: {failed}")


def _states() -> tuple[FloatArray, FloatArray]:
    return (
        np.array([10.0, -5.0, 2.0, 0.01, -0.02, 0.005], dtype=np.float64),
        np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def _reference(
    canonical: CanonicalCQP,
) -> CQPSolution:
    reference = PersistentClarabel(
        canonical,
        tolerance=1.0e-10,
        iteration_limit=2_000,
    ).solve()
    if not reference.solved:
        raise RuntimeError(f"Clarabel reference failed with status {reference.status!r}")
    return reference


def _objective(canonical: CanonicalCQP, primal: FloatArray) -> float:
    quadratic = canonical.structure.quadratic.matrix(canonical.values.quadratic)
    return float(0.5 * primal @ (quadratic @ primal) + canonical.values.linear @ primal)


def _solution_record(
    solution: CQPSolution,
    quality: CanonicalQuality,
    *,
    start_mode: StartMode,
    tolerance: float,
    model_build_seconds: float,
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "status": solution.status,
        "qualified": True,
        "start_mode": str(start_mode),
        "requested_tolerance": tolerance,
        "acceptance_threshold": _acceptance_threshold(tolerance),
        "reported_objective": solution.objective,
        "reported_primal_residual": solution.primal_residual,
        "reported_dual_residual": solution.dual_residual,
        "iterations": solution.iterations,
        "solve_seconds": solution.solve_seconds,
        "model_build_seconds": model_build_seconds,
        "total_seconds": total_seconds,
        "quality": asdict(quality),
    }


def run_hcw_case(
    *,
    intervals: int,
    tolerance: float,
    thrust_constraint: ThrustConstraint,
    start_mode: StartMode,
    seed_solution: CQPSolution | None = None,
) -> tuple[dict[str, Any], CQPSolution]:
    """Run one independently qualified HCW box or SOC PDHCG solve."""

    initial, target = _states()
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=intervals,
            max_component_acceleration=2.0e-3,
            thrust_constraint=thrust_constraint,
        )
    )
    canonical = problem.canonical(initial, target)
    reference = _reference(canonical)
    reference_objective = _objective(canonical, reference.primal)

    backend = PDHCGOneShot(canonical, params={"LogLevel": 0})
    if start_mode is not StartMode.COLD:
        if seed_solution is None:
            seed_backend = PDHCGOneShot(canonical, params={"LogLevel": 0})
            seed_solution = seed_backend.solve(tolerance=min(tolerance, 1.0e-8))
        if not seed_solution.solved:
            raise RuntimeError("PDHCG seed solve did not produce a warm start")
        backend.warm_start(
            primal=seed_solution.primal,
            dual=(seed_solution.dual if start_mode is StartMode.PRIMAL_DUAL else None),
        )

    solution = backend.solve(tolerance=tolerance)
    if not solution.solved:
        raise RuntimeError(f"PDHCG failed with status {solution.status!r}")
    quality = evaluate_pdhcg_quality(
        problem,
        canonical,
        solution,
        initial_state=initial,
        target_state=target,
        reference_objective=reference_objective,
    )
    _qualify(quality, tolerance)
    record = _solution_record(
        solution,
        quality,
        start_mode=start_mode,
        tolerance=tolerance,
        model_build_seconds=backend.last_model_build_seconds,
        total_seconds=backend.last_total_seconds,
    )
    record.update(
        {
            "intervals": intervals,
            "thrust_constraint": str(thrust_constraint),
            "variables": problem.structure.n_variables,
            "scalar_constraints": problem.structure.n_constraints,
            "affine_cone_rows": problem.structure.n_affine_constraints,
            "reference_status": reference.status,
            "reference_objective": reference_objective,
        }
    )
    return record, solution


def run_update_cases(
    *,
    intervals: int,
    tolerance: float,
    thrust_constraint: ThrustConstraint,
) -> list[dict[str, Any]]:
    """Run repeated-identical and small-values-update one-shot cases."""

    initial, target = _states()
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=intervals,
            max_component_acceleration=2.0e-3,
            thrust_constraint=thrust_constraint,
        )
    )
    canonical = problem.canonical(initial, target)
    backend = PDHCGOneShot(canonical, params={"LogLevel": 0})
    first = backend.solve(tolerance=tolerance)
    if not first.solved:
        raise RuntimeError("initial repeated-case solve failed")
    backend.warm_start(first.primal, first.dual)

    records: list[dict[str, Any]] = []
    for update_kind, current_initial, current_target in (
        ("repeated-identical", initial, target),
        (
            "small-update",
            initial + np.array([0.01, -0.01, 0.005, 1.0e-5, 0.0, -1.0e-5]),
            target + np.array([0.001, 0.0, -0.001, 0.0, 0.0, 0.0]),
        ),
    ):
        current = problem.canonical(current_initial, current_target)
        backend.update(current.values)
        reference = _reference(current)
        reference_objective = _objective(current, reference.primal)
        solution = backend.solve(tolerance=tolerance)
        if not solution.solved:
            raise RuntimeError(f"{update_kind} solve failed with {solution.status!r}")
        quality = evaluate_pdhcg_quality(
            problem,
            current,
            solution,
            initial_state=current_initial,
            target_state=current_target,
            reference_objective=reference_objective,
        )
        _qualify(quality, tolerance)
        record = _solution_record(
            solution,
            quality,
            start_mode=StartMode.PRIMAL_DUAL,
            tolerance=tolerance,
            model_build_seconds=backend.last_model_build_seconds,
            total_seconds=backend.last_total_seconds,
        )
        record.update(
            {
                "intervals": intervals,
                "thrust_constraint": str(thrust_constraint),
                "update_kind": update_kind,
                "update_count": backend.update_count,
                "warm_start_count": backend.warm_start_count,
                "reference_objective": reference_objective,
            }
        )
        records.append(record)
    return records


def run_suite(
    *,
    intervals: tuple[int, ...],
    tolerances: tuple[float, ...],
    start_modes: tuple[StartMode, ...],
    include_updates: bool,
) -> dict[str, Any]:
    """Run the declared G1 HCW correctness matrix."""

    cases: list[dict[str, Any]] = []
    update_cases: list[dict[str, Any]] = []
    for interval_count in intervals:
        for constraint in (
            ThrustConstraint.BOX,
            ThrustConstraint.SECOND_ORDER_CONE,
        ):
            seeds: dict[float, CQPSolution] = {}
            for tolerance in tolerances:
                for start_mode in start_modes:
                    record, solution = run_hcw_case(
                        intervals=interval_count,
                        tolerance=tolerance,
                        thrust_constraint=constraint,
                        start_mode=start_mode,
                        seed_solution=seeds.get(tolerance),
                    )
                    cases.append(record)
                    if start_mode is StartMode.COLD:
                        seeds[tolerance] = solution
            if include_updates:
                update_cases.extend(
                    run_update_cases(
                        intervals=interval_count,
                        tolerance=min(tolerances),
                        thrust_constraint=constraint,
                    )
                )
    return {
        "schema_version": "1.0.0",
        "suite": "G1 HCW one-shot CUDA correctness expansion",
        "pdhcg_version": PDHCGOneShot(
            CWRendezvousProblem(CWRendezvousConfig(intervals=min(intervals))).canonical(*_states()),
            params={"LogLevel": 0},
        ).upstream_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "intervals": list(intervals),
        "tolerances": list(tolerances),
        "start_modes": [str(mode) for mode in start_modes],
        "dual_convention": "canonical normal multiplier = -pdhcg Pi",
        "cases": cases,
        "update_cases": update_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, nargs="+", default=[20, 50, 100, 500])
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8],
    )
    parser.add_argument(
        "--start-modes",
        type=StartMode,
        nargs="+",
        default=list(StartMode),
    )
    parser.add_argument("--include-updates", action="store_true")
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    result = run_suite(
        intervals=tuple(arguments.intervals),
        tolerances=tuple(arguments.tolerances),
        start_modes=tuple(arguments.start_modes),
        include_updates=arguments.include_updates,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
