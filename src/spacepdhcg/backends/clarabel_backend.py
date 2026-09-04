"""Persistent Clarabel reference backend for the native SpacePDHCG CQP form."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import clarabel
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from spacepdhcg.cqp import (
    CanonicalCQP,
    CanonicalResidualAudit,
    ConeBlock,
    ConeKind,
    CQPSolution,
    CQPStructure,
    CQPValues,
    CSCStructure,
)

BoundSource = Literal["constraint", "variable"]
BoundSide = Literal["equality", "upper", "lower"]


@dataclass(frozen=True, slots=True)
class _BoundRow:
    source: BoundSource
    index: int
    side: BoundSide

    @property
    def sign(self) -> float:
        return -1.0 if self.side == "lower" else 1.0


class PersistentClarabel:
    """One Clarabel workspace updated without changing its symbolic cone programme.

    Clarabel is the CPU conic correctness reference. It is not the intended high-volume
    backend. The converter accepts PDHCG's native SOC slot ordering and transforms it to
    Clarabel's conventional cone coordinates.
    """

    supports_dynamic_solve_settings = False

    def __init__(
        self,
        problem: CanonicalCQP,
        *,
        tolerance: float = 1.0e-9,
        iteration_limit: int = 500,
        verbose: bool = False,
    ) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if iteration_limit <= 0:
            raise ValueError("iteration_limit must be positive")

        self.structure: CQPStructure = problem.structure
        values = problem.values.validated(self.structure)
        self._tolerance = float(tolerance)
        self._iteration_limit = int(iteration_limit)
        self._constraint_signature = self._bound_signature(values.lower, values.upper)
        self._variable_signature = self._bound_signature(
            values.variable_lower,
            values.variable_upper,
        )
        self._zero_rows, self._nonnegative_rows = self._make_bound_rows(values)
        self._native_cones = self._make_native_cones()
        self._cones = self._make_clarabel_cones()

        quadratic, constraint, right_hand_side = self._assemble(values)
        self._quadratic_structure = CSCStructure.from_matrix(quadratic)
        self._clarabel_constraint_structure = CSCStructure.from_matrix(constraint)

        settings = clarabel.DefaultSettings()
        settings.verbose = verbose
        settings.max_iter = iteration_limit
        settings.tol_gap_abs = tolerance
        settings.tol_gap_rel = tolerance
        settings.tol_feas = tolerance
        settings.presolve_enable = False

        start = perf_counter()
        self._solver = clarabel.DefaultSolver(
            quadratic,
            values.linear,
            constraint,
            right_hand_side,
            self._cones,
            settings,
        )
        self.setup_seconds = perf_counter() - start
        self._current = values
        self.update_count = 0
        self._last_raw_dual: NDArray[np.float64] | None = None

    @property
    def current_values(self) -> CQPValues:
        return self._current.copy()

    def update(self, values: CQPValues) -> None:
        """Update numerical values while retaining the Clarabel solver workspace."""

        updated = values.validated(self.structure)
        if self._bound_signature(updated.lower, updated.upper) != self._constraint_signature:
            raise ValueError("constraint bound type changed; persistent cone structure is invalid")
        if (
            self._bound_signature(updated.variable_lower, updated.variable_upper)
            != self._variable_signature
        ):
            raise ValueError("variable bound type changed; persistent cone structure is invalid")

        quadratic, constraint, right_hand_side = self._assemble(updated)
        self._quadratic_structure.values_from(quadratic)
        constraint_values = self._clarabel_constraint_structure.values_from(constraint)
        self._solver.update(
            P=quadratic,
            q=updated.linear,
            A=constraint_values,
            b=right_hand_side,
        )
        self._current = updated
        self.update_count += 1

    def warm_start(
        self,
        primal: NDArray | None = None,
        dual: NDArray | None = None,
    ) -> None:
        del primal, dual
        raise NotImplementedError(
            "Clarabel's public Python interface does not expose explicit primal-dual warm starts"
        )

    def solve(
        self,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> CQPSolution:
        if tolerance is not None and not np.isclose(tolerance, self._tolerance):
            raise ValueError("Clarabel tolerance is fixed when the persistent workspace is created")
        if iteration_limit is not None and iteration_limit != self._iteration_limit:
            raise ValueError(
                "Clarabel iteration_limit is fixed when the persistent workspace is created"
            )

        start = perf_counter()
        result = self._solver.solve()
        wall_seconds = perf_counter() - start
        primal = self._vector(getattr(result, "x", None), self.structure.n_variables)
        raw_dual = self._vector(
            getattr(result, "z", None),
            self._clarabel_constraint_structure.shape[0],
        )
        dual = self._canonical_dual(raw_dual)
        self._last_raw_dual = raw_dual.copy()

        return CQPSolution(
            status=str(getattr(result, "status", "Unknown")),
            primal=primal,
            dual=dual,
            objective=float(getattr(result, "obj_val", np.nan)),
            primal_residual=self._float_attribute(result, "r_prim", "res_primal"),
            dual_residual=self._float_attribute(result, "r_dual", "res_dual"),
            iterations=int(getattr(result, "iterations", 0)),
            solve_seconds=float(getattr(result, "solve_time", wall_seconds)),
        )

    def independent_residuals(
        self,
        primal: NDArray[np.float64],
    ) -> CanonicalResidualAudit:
        """Recompute unscaled KKT conditions in the expanded Clarabel formulation."""

        if self._last_raw_dual is None:
            raise RuntimeError("independent residuals require a completed solve")
        x = np.asarray(primal, dtype=np.float64)
        if x.shape != (self.structure.n_variables,) or not np.all(np.isfinite(x)):
            return CanonicalResidualAudit(*(float("inf"),) * 5)
        quadratic, constraint, right_hand_side = self._assemble(self._current)
        full_quadratic = quadratic + sp.triu(quadratic, k=1).T
        z = self._last_raw_dual
        slack = right_hand_side - np.asarray(constraint @ x, dtype=np.float64)
        stationarity = np.asarray(
            full_quadratic @ x + self._current.linear + constraint.T @ z,
            dtype=np.float64,
        )
        dual_residual = float(np.max(np.abs(stationarity), initial=0.0))
        primal_residual = 0.0
        cone_residual = 0.0
        complementarity = 0.0
        cursor = 0
        if self._zero_rows:
            count = len(self._zero_rows)
            primal_residual = max(
                primal_residual,
                float(np.max(np.abs(slack[cursor : cursor + count]), initial=0.0)),
            )
            cursor += count
        if self._nonnegative_rows:
            count = len(self._nonnegative_rows)
            primal_segment = slack[cursor : cursor + count]
            dual_segment = z[cursor : cursor + count]
            cone_residual = max(
                cone_residual,
                float(np.max(np.maximum(-primal_segment, 0.0), initial=0.0)),
                float(np.max(np.maximum(-dual_segment, 0.0), initial=0.0)),
            )
            complementarity = max(
                complementarity,
                float(np.max(np.abs(primal_segment * dual_segment), initial=0.0)),
            )
            cursor += count
        for _, cone in self._native_cones:
            count = cone.slot_count
            primal_segment = slack[cursor : cursor + count]
            dual_segment = z[cursor : cursor + count]
            if cone.kind in {ConeKind.SECOND_ORDER, ConeKind.ROTATED_SECOND_ORDER}:
                cone_residual = max(
                    cone_residual,
                    self._soc_violation(primal_segment),
                    self._soc_violation(dual_segment),
                )
            else:
                raise NotImplementedError(
                    f"independent Clarabel audit does not support {cone.kind}"
                )
            complementarity = max(
                complementarity,
                abs(float(primal_segment @ dual_segment)),
            )
            cursor += count
        if cursor != slack.size:
            raise AssertionError("Clarabel residual cone cursor mismatch")
        primal_residual = max(primal_residual, cone_residual)
        return CanonicalResidualAudit(
            primal=primal_residual,
            dual=dual_residual,
            natural=max(primal_residual, dual_residual, complementarity),
            cone=cone_residual,
            complementarity=complementarity,
        )

    def relative_kkt_residuals(
        self,
        primal: NDArray[np.float64],
    ) -> CanonicalResidualAudit:
        """Independent KKT audit normalised by the problem data magnitudes.

        The absolute audit from :meth:`independent_residuals` is divided by the standard
        relative scales ``1 + |b| + |A x|`` (primal), ``1 + |q| + |Q x| + |A^T z|`` (dual)
        and ``1 + |objective| + |b^T z|`` (complementarity) so badly scaled transcriptions
        (thrust in newtons next to 1e-8 tracking weights) are judged like PDHCG/QOCO judge
        their own canonical residuals.
        """

        absolute = self.independent_residuals(primal)
        x = np.asarray(primal, dtype=np.float64)
        if self._last_raw_dual is None or not np.all(np.isfinite(absolute.natural)):
            return absolute
        quadratic, constraint, right_hand_side = self._assemble(self._current)
        full_quadratic = quadratic + sp.triu(quadratic, k=1).T
        z = self._last_raw_dual
        ax = np.asarray(constraint @ x, dtype=np.float64)
        qx = np.asarray(full_quadratic @ x, dtype=np.float64)
        atz = np.asarray(constraint.T @ z, dtype=np.float64)
        primal_scale = (
            1.0
            + float(np.max(np.abs(right_hand_side), initial=0.0))
            + float(np.max(np.abs(ax), initial=0.0))
        )
        dual_scale = (
            1.0
            + float(np.max(np.abs(self._current.linear), initial=0.0))
            + float(np.max(np.abs(qx), initial=0.0))
            + float(np.max(np.abs(atz), initial=0.0))
        )
        objective = 0.5 * float(x @ qx) + float(self._current.linear @ x)
        gap_scale = 1.0 + abs(objective) + abs(float(right_hand_side @ z))
        primal_relative = absolute.primal / primal_scale
        dual_relative = absolute.dual / dual_scale
        complementarity_relative = absolute.complementarity / gap_scale
        return CanonicalResidualAudit(
            primal=primal_relative,
            dual=dual_relative,
            natural=max(primal_relative, dual_relative, complementarity_relative),
            cone=absolute.cone / primal_scale,
            complementarity=complementarity_relative,
        )

    @staticmethod
    def _soc_violation(value: NDArray[np.float64]) -> float:
        return max(0.0, float(np.linalg.norm(value[1:]) - value[0]))

    def _make_bound_rows(self, values: CQPValues) -> tuple[list[_BoundRow], list[_BoundRow]]:
        zero: list[_BoundRow] = []
        nonnegative: list[_BoundRow] = []
        for source, lower, upper in (
            ("constraint", values.lower, values.upper),
            ("variable", values.variable_lower, values.variable_upper),
        ):
            for index, kind in enumerate(self._bound_signature(lower, upper)):
                if kind == "equality":
                    zero.append(_BoundRow(source, index, "equality"))
                elif kind == "box":
                    nonnegative.append(_BoundRow(source, index, "upper"))
                    nonnegative.append(_BoundRow(source, index, "lower"))
                elif kind == "upper":
                    nonnegative.append(_BoundRow(source, index, "upper"))
                elif kind == "lower":
                    nonnegative.append(_BoundRow(source, index, "lower"))
        return zero, nonnegative

    @staticmethod
    def _bound_signature(lower: NDArray, upper: NDArray) -> tuple[str, ...]:
        signature: list[str] = []
        for lower_value, upper_value in zip(lower, upper, strict=True):
            lower_finite = np.isfinite(lower_value)
            upper_finite = np.isfinite(upper_value)
            if lower_finite and upper_finite:
                signature.append("equality" if lower_value == upper_value else "box")
            elif upper_finite:
                signature.append("upper")
            elif lower_finite:
                signature.append("lower")
            else:
                signature.append("free")
        return tuple(signature)

    def _make_native_cones(self) -> tuple[tuple[Literal["affine", "variable"], ConeBlock], ...]:
        native: list[tuple[Literal["affine", "variable"], ConeBlock]] = []
        native.extend(("affine", cone) for cone in self.structure.affine_cones)
        native.extend(("variable", cone) for cone in self.structure.variable_cones)
        return tuple(native)

    def _make_clarabel_cones(self) -> list[Any]:
        cones: list[Any] = []
        if self._zero_rows:
            cones.append(clarabel.ZeroConeT(len(self._zero_rows)))
        if self._nonnegative_rows:
            cones.append(clarabel.NonnegativeConeT(len(self._nonnegative_rows)))
        for _, cone in self._native_cones:
            if cone.kind in {ConeKind.SECOND_ORDER, ConeKind.ROTATED_SECOND_ORDER}:
                cones.append(clarabel.SecondOrderConeT(cone.slot_count))
            elif cone.kind is ConeKind.EXPONENTIAL:
                cones.append(clarabel.ExponentialConeT())
            elif cone.kind is ConeKind.POWER:
                cones.append(clarabel.PowerConeT(cone.power_alpha))
            elif cone.kind is ConeKind.POSITIVE_SEMIDEFINITE:
                raise NotImplementedError(
                    "PSD svec ordering must be verified before enabling the Clarabel adapter"
                )
            else:
                raise AssertionError(f"unhandled cone kind {cone.kind}")
        return cones

    def _assemble(
        self,
        values: CQPValues,
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, NDArray[np.float64]]:
        quadratic = sp.triu(
            self.structure.quadratic.matrix(values.quadratic),
            format="csc",
        )
        scalar = self.structure.constraint.matrix(values.constraint)
        affine = (
            None
            if self.structure.affine_cone is None
            else self.structure.affine_cone.matrix(values.affine_cone)
        )

        row_blocks: list[sp.spmatrix] = []
        right_hand_side: list[NDArray[np.float64]] = []
        for rows in (self._zero_rows, self._nonnegative_rows):
            if rows:
                matrix, vector = self._assemble_bound_rows(rows, scalar, values)
                row_blocks.append(matrix)
                right_hand_side.append(vector)

        for source, cone in self._native_cones:
            transform = self._cone_transform(cone)
            if source == "affine":
                if affine is None:
                    raise AssertionError("affine cone metadata without affine matrix")
                native_matrix = affine[cone.start : cone.stop, :]
                native_offset = values.affine_offset[cone.start : cone.stop]
            else:
                native_matrix = sp.eye(
                    self.structure.n_variables,
                    format="csc",
                )[cone.start : cone.stop, :]
                native_offset = np.zeros(cone.slot_count, dtype=np.float64)
            transformed_matrix = sp.csr_matrix(transform) @ native_matrix
            row_blocks.append(-transformed_matrix)
            right_hand_side.append(transform @ native_offset)

        if row_blocks:
            constraint = sp.vstack(row_blocks, format="csc")
            vector = np.concatenate(right_hand_side)
        else:
            constraint = sp.csc_matrix((0, self.structure.n_variables), dtype=np.float64)
            vector = np.empty(0, dtype=np.float64)
        constraint.sum_duplicates()
        constraint.sort_indices()
        return quadratic, constraint, vector

    def _assemble_bound_rows(
        self,
        rows: list[_BoundRow],
        scalar: sp.csc_matrix,
        values: CQPValues,
    ) -> tuple[sp.csc_matrix, NDArray[np.float64]]:
        matrices: list[sp.spmatrix] = []
        bounds: list[float] = []
        for descriptor in rows:
            if descriptor.source == "constraint":
                row = scalar.getrow(descriptor.index)
                lower = values.lower[descriptor.index]
                upper = values.upper[descriptor.index]
            else:
                row = sp.csr_matrix(
                    (
                        np.array([1.0]),
                        (np.array([0]), np.array([descriptor.index])),
                    ),
                    shape=(1, self.structure.n_variables),
                )
                lower = values.variable_lower[descriptor.index]
                upper = values.variable_upper[descriptor.index]
            matrices.append(descriptor.sign * row)
            if descriptor.side == "lower":
                bounds.append(float(-lower))
            elif descriptor.side == "upper":
                bounds.append(float(upper))
            else:
                bounds.append(float(lower))
        return sp.vstack(matrices, format="csc"), np.asarray(bounds, dtype=np.float64)

    @staticmethod
    def _cone_transform(cone: ConeBlock) -> NDArray[np.float64]:
        size = cone.slot_count
        transform = np.zeros((size, size), dtype=np.float64)
        if cone.kind is ConeKind.SECOND_ORDER:
            dimension = cone.vector_dimension
            transform[0, size - 1] = 1.0
            transform[1 : dimension + 1, :dimension] = np.eye(dimension)
            transform[-1, dimension] = 1.0
            return transform
        if cone.kind is ConeKind.ROTATED_SECOND_ORDER:
            dimension = cone.vector_dimension
            transform[0, dimension] = 1.0
            transform[0, dimension + 1] = 1.0
            transform[1 : dimension + 1, :dimension] = np.sqrt(2.0) * np.eye(dimension)
            transform[-1, dimension] = 1.0
            transform[-1, dimension + 1] = -1.0
            return transform
        if cone.kind in {ConeKind.EXPONENTIAL, ConeKind.POWER}:
            return np.eye(size)
        if cone.kind is ConeKind.POSITIVE_SEMIDEFINITE:
            raise NotImplementedError("PSD cone transformation is not yet verified")
        raise AssertionError(f"unhandled cone kind {cone.kind}")

    def _canonical_dual(self, raw_dual: NDArray[np.float64]) -> NDArray[np.float64]:
        canonical = np.zeros(self.structure.n_duals, dtype=np.float64)
        cursor = 0
        for descriptor in (*self._zero_rows, *self._nonnegative_rows):
            value = raw_dual[cursor]
            cursor += 1
            if descriptor.source == "constraint":
                canonical[descriptor.index] += descriptor.sign * value
        affine_offset = self.structure.n_constraints
        for source, cone in self._native_cones:
            segment = raw_dual[cursor : cursor + cone.slot_count]
            cursor += cone.slot_count
            if source == "affine":
                transform = self._cone_transform(cone)
                canonical[affine_offset + cone.start : affine_offset + cone.stop] = (
                    transform.T @ segment
                )
        return canonical

    @staticmethod
    def _vector(value: Any, size: int) -> NDArray[np.float64]:
        if value is None:
            return np.full(size, np.nan, dtype=np.float64)
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise RuntimeError(f"solver returned vector shape {vector.shape}, expected ({size},)")
        return vector.copy()

    @staticmethod
    def _float_attribute(result: Any, primary: str, fallback: str) -> float:
        value = getattr(result, primary, getattr(result, fallback, np.nan))
        return float(value)
