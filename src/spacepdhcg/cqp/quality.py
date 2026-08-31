"""Solver-status qualification shared by correctness and benchmark code."""

from __future__ import annotations

from typing import Protocol

import numpy as np


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
    if not (
        np.all(np.isfinite(residuals))
        and np.max(np.abs(residuals)) <= tolerance
    ):
        return False
    if objective_upper_bound is None:
        return True
    return bool(
        np.isfinite(solution.objective)
        and solution.objective <= objective_upper_bound + objective_tolerance
    )
