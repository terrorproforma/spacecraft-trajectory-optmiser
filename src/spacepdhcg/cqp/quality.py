"""Solver-status qualification shared by correctness and benchmark code."""

from __future__ import annotations

import numpy as np

from spacepdhcg.cqp.problem import CQPSolution


_RESIDUAL_QUALIFIABLE_STATUSES = {
    "almostsolved",
    "solvedinaccurate",
}


def residual_qualified(
    solution: CQPSolution,
    *,
    tolerance: float,
) -> bool:
    """Accept a conservative solver status only after residual verification.

    Strictly solved results are always accepted. Conservative statuses such as
    Clarabel's ``AlmostSolved`` are accepted only when both reported residuals are
    finite and no larger than the independently declared tolerance. Other termination
    reasons remain failures regardless of their final iterate.
    """

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if solution.solved:
        return True
    status = solution.status.lower().replace("_", "").replace(" ", "")
    if status not in _RESIDUAL_QUALIFIABLE_STATUSES:
        return False
    residuals = np.asarray(
        [solution.primal_residual, solution.dual_residual],
        dtype=np.float64,
    )
    return bool(
        np.all(np.isfinite(residuals))
        and np.max(np.abs(residuals)) <= tolerance
    )
