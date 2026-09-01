"""One-shot adapter from SpacePDHCG's canonical form to upstream PDHCG.

This adapter proves exact problem-data compatibility and enables CUDA integration tests. It is
intentionally *not* the persistent backend required by contribution B: upstream's public
Python API currently rebuilds and preprocesses solver state inside every ``optimize()`` call.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.cqp import (
    CanonicalCQP,
    ConeBlock,
    ConeKind,
    CQPSolution,
    CQPStructure,
    CQPValues,
)

_CONE_NAMES = {
    ConeKind.SECOND_ORDER: "soc",
    ConeKind.ROTATED_SECOND_ORDER: "rsoc",
    ConeKind.EXPONENTIAL: "exp",
    ConeKind.POWER: "power",
    ConeKind.POSITIVE_SEMIDEFINITE: "psd",
}


class PDHCGUnavailableError(RuntimeError):
    """Raised when the optional CUDA-enabled upstream package cannot be imported."""


class PDHCGOneShot:
    """Map a canonical CQP to upstream ``pdhcg.Model`` for one-shot solves.

    The class follows the persistent backend lifecycle so callers can already exercise
    ``update`` and ``warm_start``. Nevertheless, each call to :meth:`solve` constructs a new
    upstream model. ``is_persistent`` is therefore a deliberate, testable ``False``.
    """

    is_persistent = False

    def __init__(
        self,
        problem: CanonicalCQP,
        *,
        params: Mapping[str, Any] | None = None,
        pdhcg_module: Any | None = None,
    ) -> None:
        self.structure: CQPStructure = problem.structure
        self._current = problem.values.validated(self.structure)
        self._module = pdhcg_module if pdhcg_module is not None else self._import_upstream()
        self._params = dict(params or {})
        self._primal_start: NDArray[np.float64] | None = None
        self._dual_start: NDArray[np.float64] | None = None
        self._last_model: Any | None = None
        self.update_count = 0
        self.warm_start_count = 0
        self.solve_count = 0
        self.last_model_build_seconds = 0.0
        self.last_total_seconds = 0.0

    @property
    def current_values(self) -> CQPValues:
        return self._current.copy()

    @property
    def last_model(self) -> Any | None:
        """Most recently constructed upstream model, primarily for integration diagnostics."""

        return self._last_model

    @property
    def upstream_version(self) -> str:
        return str(getattr(self._module, "__version__", "unknown"))

    def update(self, values: CQPValues) -> None:
        """Replace numerical values while retaining the immutable SpacePDHCG structure."""

        self._current = values.validated(self.structure)
        self.update_count += 1

    def warm_start(
        self,
        primal: NDArray | None = None,
        dual: NDArray | None = None,
    ) -> None:
        if primal is None and dual is None:
            raise ValueError("at least one warm-start vector is required")
        if primal is not None:
            candidate = np.asarray(primal, dtype=np.float64)
            if candidate.shape != (self.structure.n_variables,):
                raise ValueError("primal warm start has the wrong shape")
            self._primal_start = candidate.copy()
        if dual is not None:
            candidate = np.asarray(dual, dtype=np.float64)
            if candidate.shape != (self.structure.n_duals,):
                raise ValueError("dual warm start has the wrong shape")
            self._dual_start = candidate.copy()
        self.warm_start_count += 1

    def clear_warm_start(self) -> None:
        self._primal_start = None
        self._dual_start = None

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

        total_start = perf_counter()
        build_start = perf_counter()
        model = self._module.Model(**self._model_arguments())
        self.last_model_build_seconds = perf_counter() - build_start

        parameters = dict(self._params)
        if tolerance is not None:
            parameters["OptimalityTol"] = float(tolerance)
            parameters["FeasibilityTol"] = float(tolerance)
        if iteration_limit is not None:
            parameters["IterationLimit"] = int(iteration_limit)
        if parameters:
            model.setParams(**parameters)
        if self._primal_start is not None or self._dual_start is not None:
            model.setWarmStart(
                primal=self._primal_start,
                dual=self._dual_start,
            )

        model.optimize()
        self.last_total_seconds = perf_counter() - total_start
        self._last_model = model
        self.solve_count += 1

        raw_status = str(getattr(model, "Status", None) or "Unknown")
        status = self._normalise_status(raw_status)
        primal = self._vector(getattr(model, "X", None), self.structure.n_variables)
        dual = self._vector(getattr(model, "Pi", None), self.structure.n_duals)
        runtime = self._finite_float(getattr(model, "Runtime", None), self.last_total_seconds)

        return CQPSolution(
            status=status,
            primal=primal,
            dual=dual,
            objective=self._finite_float(getattr(model, "ObjVal", None), np.nan),
            primal_residual=self._finite_float(
                getattr(model, "RelPrimalResidual", None),
                np.nan,
            ),
            dual_residual=self._finite_float(
                getattr(model, "RelDualResidual", None),
                np.nan,
            ),
            iterations=int(getattr(model, "IterCount", None) or 0),
            solve_seconds=runtime,
        )

    def _model_arguments(self) -> dict[str, Any]:
        values = self._current
        affine_matrix = (
            None
            if self.structure.affine_cone is None
            else self.structure.affine_cone.matrix(values.affine_cone)
        )
        return {
            "objective_vector": values.linear.copy(),
            "constraint_matrix": self.structure.constraint.matrix(values.constraint),
            "constraint_lower_bound": values.lower.copy(),
            "constraint_upper_bound": values.upper.copy(),
            "objective_matrix": self.structure.quadratic.matrix(values.quadratic),
            "variable_lower_bound": values.variable_lower.copy(),
            "variable_upper_bound": values.variable_upper.copy(),
            "objective_constant": 0.0,
            "affine_cone_matrix": affine_matrix,
            "affine_cone_offset": (None if affine_matrix is None else values.affine_offset.copy()),
            "affine_cones": self._cone_spec(self.structure.affine_cones),
            "variable_cones": self._cone_spec(self.structure.variable_cones),
        }

    def _cone_spec(self, cones: tuple[ConeBlock, ...]) -> Any | None:
        if not cones:
            return None
        return self._module.ConeSpec(
            types=[_CONE_NAMES[cone.kind] for cone in cones],
            starts=np.asarray([cone.start for cone in cones], dtype=np.int32),
            v_dims=np.asarray([cone.vector_dimension for cone in cones], dtype=np.int32),
            power_alphas=np.asarray([cone.power_alpha for cone in cones], dtype=np.float64),
        )

    @staticmethod
    def _normalise_status(raw_status: str) -> str:
        lower = raw_status.lower()
        failed_tokens = ("infeasible", "unbounded", "error", "failed", "limit")
        solved_tokens = ("optimal", "solved", "converged")
        if not any(token in lower for token in failed_tokens) and any(
            token in lower for token in solved_tokens
        ):
            return f"Solved ({raw_status})"
        return raw_status

    @staticmethod
    def _vector(value: Any, size: int) -> NDArray[np.float64]:
        if value is None:
            return np.full(size, np.nan, dtype=np.float64)
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise RuntimeError(f"PDHCG returned vector shape {vector.shape}; expected ({size},)")
        return vector.copy()

    @staticmethod
    def _finite_float(value: Any, fallback: float) -> float:
        if value is None:
            return float(fallback)
        candidate = float(value)
        return candidate if np.isfinite(candidate) else float(fallback)

    @staticmethod
    def _import_upstream() -> Any:
        try:
            return import_module("pdhcg")
        except (ImportError, OSError) as error:
            raise PDHCGUnavailableError(
                "The optional upstream 'pdhcg' package and its CUDA runtime are required "
                "for PDHCGOneShot. CPU reference backends remain available without them."
            ) from error
