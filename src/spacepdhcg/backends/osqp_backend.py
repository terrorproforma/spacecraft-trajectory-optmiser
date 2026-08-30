"""Persistent OSQP reference backend.

OSQP is not the target accelerator. It provides an executable lifecycle and a trusted QP
reference while the PDHCG-CQP bridge is developed.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import osqp
from numpy.typing import NDArray

from spacepdhcg.cqp import CanonicalCQP, CQPSolution, CQPStructure, CQPValues


class PersistentOSQP:
    """One OSQP workspace with fixed matrix structure and repeated value updates."""

    def __init__(
        self,
        problem: CanonicalCQP,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.structure: CQPStructure = problem.structure
        values = problem.values.validated(self.structure)
        self._require_supported(values)
        quadratic = self.structure.quadratic.matrix(values.quadratic)
        constraint = self.structure.constraint.matrix(values.constraint)

        solver_settings: dict[str, Any] = {
            "verbose": False,
            "polishing": True,
            "warm_starting": True,
            "eps_abs": 1.0e-9,
            "eps_rel": 1.0e-9,
            "max_iter": 100_000,
        }
        if settings is not None:
            solver_settings.update(settings)

        self._solver = osqp.OSQP()
        start = perf_counter()
        self._solver.setup(
            P=quadratic,
            q=values.linear,
            A=constraint,
            l=values.lower,
            u=values.upper,
            **solver_settings,
        )
        self.setup_seconds = perf_counter() - start
        self._current = values
        self.update_count = 0
        self.warm_start_count = 0

    @property
    def current_values(self) -> CQPValues:
        return self._current.copy()

    def update(self, values: CQPValues) -> None:
        """Update numerical coefficients while retaining the OSQP workspace."""

        updated = values.validated(self.structure)
        self._require_supported(updated)
        arguments: dict[str, NDArray[np.float64]] = {
            "q": updated.linear,
            "l": updated.lower,
            "u": updated.upper,
        }
        if not np.array_equal(updated.quadratic, self._current.quadratic):
            arguments["Px"] = updated.quadratic
        if not np.array_equal(updated.constraint, self._current.constraint):
            arguments["Ax"] = updated.constraint

        self._solver.update(**arguments)
        self._current = updated
        self.update_count += 1

    def warm_start(
        self,
        primal: NDArray | None = None,
        dual: NDArray | None = None,
    ) -> None:
        if primal is None and dual is None:
            raise ValueError("at least one warm-start vector is required")

        arguments: dict[str, NDArray[np.float64]] = {}
        if primal is not None:
            candidate = np.asarray(primal, dtype=np.float64)
            if candidate.shape != (self.structure.n_variables,):
                raise ValueError("primal warm start has the wrong shape")
            arguments["x"] = candidate
        if dual is not None:
            candidate = np.asarray(dual, dtype=np.float64)
            if candidate.shape != (self.structure.n_constraints,):
                raise ValueError("dual warm start has the wrong shape")
            arguments["y"] = candidate

        self._solver.warm_start(**arguments)
        self.warm_start_count += 1

    def solve(
        self,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> CQPSolution:
        if tolerance is not None and tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if iteration_limit is not None and iteration_limit <= 0:
            raise ValueError("iteration_limit must be positive")

        settings: dict[str, Any] = {}
        if tolerance is not None:
            settings["eps_abs"] = tolerance
            settings["eps_rel"] = tolerance
        if iteration_limit is not None:
            settings["max_iter"] = iteration_limit
        if settings:
            self._solver.update_settings(**settings)

        result = self._solver.solve(raise_error=False)
        info = result.info
        primal = self._result_vector(result.x, self.structure.n_variables)
        dual = self._result_vector(result.y, self.structure.n_constraints)

        return CQPSolution(
            status=str(info.status),
            primal=primal,
            dual=dual,
            objective=float(info.obj_val),
            primal_residual=self._info_value(info, "prim_res", "pri_res"),
            dual_residual=self._info_value(info, "dual_res", "dua_res"),
            iterations=int(info.iter),
            solve_seconds=float(info.run_time),
        )

    def _require_supported(self, values: CQPValues) -> None:
        if self.structure.affine_cone is not None or self.structure.variable_cones:
            raise ValueError("OSQP reference backend supports scalar QPs only")
        if np.any(np.isfinite(values.variable_lower)) or np.any(np.isfinite(values.variable_upper)):
            raise ValueError(
                "OSQP reference backend currently requires variable bounds "
                "to be encoded as scalar rows"
            )

    @staticmethod
    def _result_vector(vector: NDArray | None, size: int) -> NDArray[np.float64]:
        if vector is None:
            return np.full(size, np.nan, dtype=np.float64)
        return np.asarray(vector, dtype=np.float64).copy()

    @staticmethod
    def _info_value(info: Any, primary: str, legacy: str) -> float:
        value = getattr(info, primary, getattr(info, legacy, np.nan))
        return float(value)
