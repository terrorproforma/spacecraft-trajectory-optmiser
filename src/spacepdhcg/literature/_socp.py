"""Minimal independent SOCP builder on top of Clarabel.

This helper is deliberately separate from the SpacePDHCG canonical CQP path.  It exists so
the literature reproductions can be checked against an *independent* transcription
(``measured-local`` evidence produced without the persistent backend, trust regions, or
virtual controls).  Problems are assembled in Clarabel's native form

    minimise 0.5 x'Px + q'x   subject to   A x + s = b,  s in K,

where ``K`` is an ordered product of zero, non-negative, and second-order cones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SOLVED_STATUSES = {"Solved", "AlmostSolved"}


class SOCPInfeasibleError(RuntimeError):
    """Raised when Clarabel does not return a solved status."""


@dataclass(slots=True)
class LinearExpression:
    """Sparse affine expression ``sum(coefficients[i] * x[index[i]]) + constant``."""

    indices: list[int] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    constant: float = 0.0

    @classmethod
    def variable(cls, index: int, coefficient: float = 1.0) -> LinearExpression:
        return cls([int(index)], [float(coefficient)], 0.0)

    @classmethod
    def const(cls, value: float) -> LinearExpression:
        return cls([], [], float(value))

    def scaled(self, factor: float) -> LinearExpression:
        return LinearExpression(
            list(self.indices),
            [float(factor) * value for value in self.coefficients],
            float(factor) * self.constant,
        )

    def plus(self, other: LinearExpression | float) -> LinearExpression:
        if isinstance(other, LinearExpression):
            return LinearExpression(
                self.indices + other.indices,
                self.coefficients + other.coefficients,
                self.constant + other.constant,
            )
        return LinearExpression(list(self.indices), list(self.coefficients), self.constant + other)

    def minus(self, other: LinearExpression | float) -> LinearExpression:
        if isinstance(other, LinearExpression):
            return self.plus(other.scaled(-1.0))
        return self.plus(-float(other))

    def evaluate(self, x: FloatArray) -> float:
        total = self.constant
        for index, coefficient in zip(self.indices, self.coefficients, strict=True):
            total += coefficient * float(x[index])
        return float(total)


def lin(index: int, coefficient: float = 1.0) -> LinearExpression:
    return LinearExpression.variable(index, coefficient)


def affine(terms: dict[int, float], constant: float = 0.0) -> LinearExpression:
    return LinearExpression(list(terms.keys()), [float(v) for v in terms.values()], constant)


@dataclass(slots=True)
class SOCPSolution:
    x: FloatArray
    objective: float
    status: str
    iterations: int
    solve_seconds: float
    primal_residual: float
    dual_residual: float

    @property
    def solved(self) -> bool:
        return self.status in _SOLVED_STATUSES


class SOCPBuilder:
    """Accumulate variables, cones, and objective terms; then solve with Clarabel."""

    def __init__(self) -> None:
        self.n_variables = 0
        self._names: dict[str, slice] = {}
        self._linear: dict[int, float] = {}
        self._quadratic: dict[tuple[int, int], float] = {}
        # Each row is stored as (expression, rhs) meaning expression·x + s = rhs.
        self._zero_rows: list[LinearExpression] = []
        self._nonneg_rows: list[LinearExpression] = []
        self._soc_blocks: list[list[LinearExpression]] = []

    # ------------------------------------------------------------------ variables
    def add_variables(self, count: int, name: str | None = None) -> slice:
        if count <= 0:
            raise ValueError("count must be positive")
        block = slice(self.n_variables, self.n_variables + int(count))
        self.n_variables += int(count)
        if name is not None:
            if name in self._names:
                raise ValueError(f"duplicate variable block name {name!r}")
            self._names[name] = block
        return block

    def block(self, name: str) -> slice:
        return self._names[name]

    # ------------------------------------------------------------------ objective
    def add_linear_cost(self, index: int, coefficient: float) -> None:
        self._linear[int(index)] = self._linear.get(int(index), 0.0) + float(coefficient)

    def add_quadratic_cost(self, index: int, coefficient: float, other: int | None = None) -> None:
        """Add ``0.5 * coefficient * x_i * x_j`` (``j = i`` when ``other`` is None)."""

        i = int(index)
        j = i if other is None else int(other)
        key = (min(i, j), max(i, j))
        self._quadratic[key] = self._quadratic.get(key, 0.0) + float(coefficient)

    # ------------------------------------------------------------------ constraints
    def add_equality(self, expression: LinearExpression, rhs: float = 0.0) -> None:
        """Impose ``expression == rhs``."""

        self._zero_rows.append(expression.minus(rhs))

    def add_leq(self, expression: LinearExpression, rhs: float = 0.0) -> None:
        """Impose ``expression <= rhs``."""

        self._nonneg_rows.append(expression.minus(rhs))

    def add_geq(self, expression: LinearExpression, rhs: float = 0.0) -> None:
        """Impose ``expression >= rhs``."""

        self._nonneg_rows.append(expression.scaled(-1.0).plus(rhs))

    def add_bounds(self, block: slice, lower: float | None, upper: float | None) -> None:
        for index in range(block.start, block.stop):
            if lower is not None and np.isfinite(lower):
                self.add_geq(lin(index), float(lower))
            if upper is not None and np.isfinite(upper):
                self.add_leq(lin(index), float(upper))

    def add_soc(self, scalar: LinearExpression, vector: list[LinearExpression]) -> None:
        """Impose ``||vector||_2 <= scalar``."""

        if not vector:
            raise ValueError("second-order cone needs at least one vector component")
        self._soc_blocks.append([scalar, *vector])

    def add_norm_bound(self, block: slice, bound: LinearExpression | float) -> None:
        scalar = bound if isinstance(bound, LinearExpression) else LinearExpression.const(bound)
        self.add_soc(scalar, [lin(index) for index in range(block.start, block.stop)])

    # ------------------------------------------------------------------ assembly
    def _rows_to_matrix(
        self,
        rows: list[LinearExpression],
        *,
        negate: bool,
    ) -> tuple[sp.csc_matrix, FloatArray]:
        """Return (A_block, b_block) such that ``A x + s = b`` encodes each row.

        For a zero/non-negative row ``e·x + c (<=|=) 0`` we need ``s = -(e·x + c) >= 0``,
        i.e. ``A = e``, ``b = -c``.  For SOC entries we need ``s = e·x + c`` so ``A = -e``,
        ``b = c``.
        """

        data: list[float] = []
        row_indices: list[int] = []
        column_indices: list[int] = []
        b = np.zeros(len(rows), dtype=np.float64)
        for row, expression in enumerate(rows):
            sign = -1.0 if negate else 1.0
            for index, coefficient in zip(expression.indices, expression.coefficients, strict=True):
                data.append(sign * coefficient)
                row_indices.append(row)
                column_indices.append(index)
            b[row] = expression.constant if negate else -expression.constant
        matrix = sp.csc_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(rows), self.n_variables),
            dtype=np.float64,
        )
        matrix.sum_duplicates()
        return matrix, b

    def assemble(self) -> tuple[sp.csc_matrix, FloatArray, sp.csc_matrix, FloatArray, list]:
        import clarabel

        blocks = []
        rhs = []
        cones = []
        if self._zero_rows:
            matrix, b = self._rows_to_matrix(self._zero_rows, negate=False)
            blocks.append(matrix)
            rhs.append(b)
            cones.append(clarabel.ZeroConeT(len(self._zero_rows)))
        if self._nonneg_rows:
            matrix, b = self._rows_to_matrix(self._nonneg_rows, negate=False)
            blocks.append(matrix)
            rhs.append(b)
            cones.append(clarabel.NonnegativeConeT(len(self._nonneg_rows)))
        for block in self._soc_blocks:
            matrix, b = self._rows_to_matrix(block, negate=True)
            blocks.append(matrix)
            rhs.append(b)
            cones.append(clarabel.SecondOrderConeT(len(block)))
        A = sp.vstack(blocks, format="csc") if blocks else sp.csc_matrix((0, self.n_variables))
        b_full = np.concatenate(rhs) if rhs else np.zeros(0)
        q = np.zeros(self.n_variables, dtype=np.float64)
        for index, coefficient in self._linear.items():
            q[index] = coefficient
        if self._quadratic:
            rows_q = [key[0] for key in self._quadratic]
            cols_q = [key[1] for key in self._quadratic]
            vals_q = list(self._quadratic.values())
            P = sp.csc_matrix(
                (vals_q, (rows_q, cols_q)),
                shape=(self.n_variables, self.n_variables),
                dtype=np.float64,
            )
        else:
            P = sp.csc_matrix((self.n_variables, self.n_variables), dtype=np.float64)
        return P, q, A, b_full, cones

    def solve(
        self,
        *,
        tolerance: float = 1.0e-9,
        max_iterations: int = 500,
        verbose: bool = False,
        raise_on_failure: bool = True,
    ) -> SOCPSolution:
        import clarabel

        P, q, A, b, cones = self.assemble()
        settings = clarabel.DefaultSettings()
        settings.verbose = verbose
        settings.max_iter = int(max_iterations)
        settings.tol_gap_abs = float(tolerance)
        settings.tol_gap_rel = float(tolerance)
        settings.tol_feas = float(tolerance)
        settings.tol_ktratio = 1.0e-6
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        result = solver.solve()
        status = str(result.status).split(".")[-1]
        x = np.asarray(result.x, dtype=np.float64)
        primal = float(getattr(result, "r_prim", np.nan))
        dual = float(getattr(result, "r_dual", np.nan))
        solution = SOCPSolution(
            x=x,
            objective=float(result.obj_val),
            status=status,
            iterations=int(result.iterations),
            solve_seconds=float(result.solve_time),
            primal_residual=primal,
            dual_residual=dual,
        )
        if raise_on_failure and not solution.solved:
            raise SOCPInfeasibleError(f"Clarabel returned status {status}")
        return solution
