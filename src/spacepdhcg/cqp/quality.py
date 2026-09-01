"""Solver-status qualification shared by correctness and benchmark code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .problem import CanonicalCQP, ConeBlock, ConeKind

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CanonicalResidualAudit:
    """Unscaled KKT audit in the repository's canonical coordinates."""

    primal: float
    dual: float
    natural: float
    cone: float
    complementarity: float


def _maximum(values: FloatArray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _cone_transform(cone: ConeBlock) -> FloatArray:
    size = cone.slot_count
    transform = np.zeros((size, size), dtype=np.float64)
    dimension = cone.vector_dimension
    if cone.kind is ConeKind.SECOND_ORDER:
        transform[0, size - 1] = 1.0
        transform[1 : dimension + 1, :dimension] = np.eye(dimension)
        transform[-1, dimension] = 1.0
        return transform
    if cone.kind is ConeKind.ROTATED_SECOND_ORDER:
        transform[0, dimension] = 1.0
        transform[0, dimension + 1] = 1.0
        transform[1 : dimension + 1, :dimension] = np.sqrt(2.0) * np.eye(dimension)
        transform[-1, dimension] = 1.0
        transform[-1, dimension + 1] = -1.0
        return transform
    raise NotImplementedError(f"canonical residual audit does not support {cone.kind}")


def _soc_violation(value: FloatArray) -> float:
    return max(0.0, float(np.linalg.norm(value[1:]) - value[0]))


def independent_canonical_residuals(
    problem: CanonicalCQP,
    primal: FloatArray,
    dual: FloatArray,
) -> CanonicalResidualAudit:
    """Recompute primal, stationarity, cone-dual, and complementarity residuals.

    Canonical dual vectors intentionally omit duals introduced by backend conversion of
    variable bounds and variable cones. The audit therefore fails closed with an infinite
    dual residual when either feature is present.
    """

    structure = problem.structure
    values = problem.values
    x = np.asarray(primal, dtype=np.float64)
    y = np.asarray(dual, dtype=np.float64)
    if x.shape != (structure.n_variables,) or y.shape != (structure.n_duals,):
        raise ValueError("primal or canonical dual has the wrong shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return CanonicalResidualAudit(*(float("inf"),) * 5)

    scalar_matrix = structure.constraint.matrix(values.constraint)
    scalar_value = np.asarray(scalar_matrix @ x, dtype=np.float64)
    scalar_violation = _maximum(
        np.maximum(
            np.maximum(values.lower - scalar_value, 0.0),
            np.maximum(scalar_value - values.upper, 0.0),
        )
    )
    variable_violation = _maximum(
        np.maximum(
            np.maximum(values.variable_lower - x, 0.0),
            np.maximum(x - values.variable_upper, 0.0),
        )
    )
    cone_violation = 0.0
    complementarity = 0.0
    stationarity = np.asarray(values.linear, dtype=np.float64).copy()
    quadratic = structure.quadratic.matrix(values.quadratic)
    full_quadratic = quadratic + sp.triu(quadratic, k=1).T
    stationarity += np.asarray(full_quadratic @ x, dtype=np.float64)
    scalar_dual = y[: structure.n_constraints]
    stationarity += np.asarray(scalar_matrix.T @ scalar_dual, dtype=np.float64)

    for row, multiplier in enumerate(scalar_dual):
        lower = values.lower[row]
        upper = values.upper[row]
        if np.isfinite(lower) and np.isfinite(upper) and lower == upper:
            continue
        if np.isfinite(lower) and np.isfinite(upper):
            if multiplier >= 0.0:
                complementarity = max(
                    complementarity,
                    abs(multiplier * (upper - scalar_value[row])),
                )
            else:
                complementarity = max(
                    complementarity,
                    abs((-multiplier) * (scalar_value[row] - lower)),
                )
        elif np.isfinite(upper):
            complementarity = max(
                complementarity,
                abs(multiplier * (upper - scalar_value[row])),
                max(0.0, -multiplier),
            )
        elif np.isfinite(lower):
            complementarity = max(
                complementarity,
                abs((-multiplier) * (scalar_value[row] - lower)),
                max(0.0, multiplier),
            )

    if structure.affine_cone is not None:
        affine_matrix = structure.affine_cone.matrix(values.affine_cone)
        affine_value = np.asarray(
            affine_matrix @ x + values.affine_offset,
            dtype=np.float64,
        )
        affine_dual = y[structure.n_constraints :]
        stationarity += np.asarray(affine_matrix.T @ affine_dual, dtype=np.float64)
        for cone in structure.affine_cones:
            transform = _cone_transform(cone)
            primal_segment = affine_value[cone.start : cone.stop]
            dual_segment = affine_dual[cone.start : cone.stop]
            transformed_primal = transform @ primal_segment
            transformed_dual = np.linalg.solve(transform.T, dual_segment)
            cone_violation = max(
                cone_violation,
                _soc_violation(transformed_primal),
                _soc_violation(transformed_dual),
            )
            complementarity = max(
                complementarity,
                abs(float(primal_segment @ dual_segment)),
            )

    has_backend_only_duals = (
        np.any(np.isfinite(values.variable_lower))
        or np.any(np.isfinite(values.variable_upper))
        or bool(structure.variable_cones)
    )
    dual_residual = float("inf") if has_backend_only_duals else _maximum(stationarity)
    primal_residual = max(scalar_violation, variable_violation, cone_violation)
    natural = max(primal_residual, dual_residual, complementarity)
    return CanonicalResidualAudit(
        primal=primal_residual,
        dual=dual_residual,
        natural=natural,
        cone=cone_violation,
        complementarity=complementarity,
    )


class SolutionQualityView(Protocol):
    """Minimal result surface required for independent solution qualification."""

    status: str
    objective: float
    primal_residual: float
    dual_residual: float

    @property
    def solved(self) -> bool: ...


_RESIDUAL_QUALIFIABLE_STATUSES = {
    "almostsolved",
    "solvedinaccurate",
}


def residual_qualified(
    solution: SolutionQualityView,
    *,
    tolerance: float,
    objective_upper_bound: float | None = None,
    objective_tolerance: float = 0.0,
) -> bool:
    """Independently qualify solver termination, residuals, and an optional incumbent.

    A ``Solved`` status is not accepted on trust: both reported residuals must be finite
    and no larger than ``tolerance``. Conservative statuses such as ``AlmostSolved`` are
    handled by the same numerical test. When a known feasible incumbent objective is
    supplied, a minimisation result may not exceed it beyond ``objective_tolerance``.
    This catches status/residual false positives on degenerate conic formulations.
    """

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(objective_tolerance) or objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and non-negative")
    if objective_upper_bound is not None and not np.isfinite(objective_upper_bound):
        raise ValueError("objective_upper_bound must be finite when supplied")

    status = solution.status.lower().replace("_", "").replace(" ", "")
    if not solution.solved and status not in _RESIDUAL_QUALIFIABLE_STATUSES:
        return False
    residuals = np.asarray(
        [solution.primal_residual, solution.dual_residual],
        dtype=np.float64,
    )
    if not (np.all(np.isfinite(residuals)) and np.max(np.abs(residuals)) <= tolerance):
        return False
    if objective_upper_bound is None:
        return True
    return bool(
        np.isfinite(solution.objective)
        and solution.objective <= objective_upper_bound + objective_tolerance
    )
