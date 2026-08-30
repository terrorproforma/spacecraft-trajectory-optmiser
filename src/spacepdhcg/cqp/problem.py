"""Fixed-pattern conic quadratic problem representation.

The immutable :class:`CQPStructure` is allocated once. Successive solves replace only
:class:`CQPValues`. This is the CPU reference contract for the future device-resident
``PersistentCQP`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ConeKind(StrEnum):
    """Cone families required by the core spacecraft programme."""

    ZERO = "zero"
    NONNEGATIVE = "nonnegative"
    SECOND_ORDER = "second_order"
    ROTATED_SECOND_ORDER = "rotated_second_order"
    EXPONENTIAL = "exponential"
    POWER = "power"
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"


@dataclass(frozen=True, slots=True)
class ConeBlock:
    """A fixed conic row block in the affine constraint operator."""

    kind: ConeKind
    start: int
    size: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("cone start must be non-negative")
        if self.size <= 0:
            raise ValueError("cone size must be positive")


@dataclass(frozen=True, slots=True)
class CSCStructure:
    """Immutable compressed-sparse-column index structure."""

    shape: tuple[int, int]
    indptr: IntArray
    indices: IntArray

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 0:
            raise ValueError("shape must contain two non-negative dimensions")

        indptr = np.asarray(self.indptr, dtype=np.int64).copy()
        indices = np.asarray(self.indices, dtype=np.int64).copy()
        rows, columns = self.shape

        if indptr.ndim != 1 or indices.ndim != 1:
            raise ValueError("CSC index arrays must be one-dimensional")
        if indptr.size != columns + 1:
            raise ValueError("indptr length must equal number of columns plus one")
        if indptr.size == 0 or indptr[0] != 0:
            raise ValueError("indptr must begin at zero")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must be non-decreasing")
        if indptr[-1] != indices.size:
            raise ValueError("indptr[-1] must equal the number of stored entries")
        if np.any(indices < 0) or np.any(indices >= rows):
            raise ValueError("row index outside matrix shape")

        for column in range(columns):
            start = indptr[column]
            stop = indptr[column + 1]
            column_indices = indices[start:stop]
            if column_indices.size > 1 and np.any(np.diff(column_indices) <= 0):
                raise ValueError("row indices must be strictly increasing within each column")

        indptr.flags.writeable = False
        indices.flags.writeable = False
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)

    @property
    def nnz(self) -> int:
        """Number of stored numerical entries."""

        return int(self.indices.size)

    @classmethod
    def from_matrix(cls, matrix: sp.spmatrix) -> CSCStructure:
        """Extract a canonical, sorted CSC structure from ``matrix``."""

        csc = sp.csc_matrix(matrix, dtype=np.float64, copy=True)
        csc.sum_duplicates()
        csc.sort_indices()
        return cls(shape=csc.shape, indptr=csc.indptr, indices=csc.indices)

    def matrix(self, values: FloatArray) -> sp.csc_matrix:
        """Construct a CSC matrix with this structure and supplied values."""

        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 1 or data.size != self.nnz:
            raise ValueError(f"expected {self.nnz} values, received shape {data.shape}")
        return sp.csc_matrix(
            (data.copy(), self.indices.copy(), self.indptr.copy()),
            shape=self.shape,
        )

    def values_from(self, matrix: sp.spmatrix) -> FloatArray:
        """Return matrix values after asserting an identical CSC pattern."""

        csc = sp.csc_matrix(matrix, dtype=np.float64, copy=True)
        csc.sum_duplicates()
        csc.sort_indices()
        if csc.shape != self.shape:
            raise ValueError(f"matrix shape changed from {self.shape} to {csc.shape}")
        if not np.array_equal(csc.indptr, self.indptr) or not np.array_equal(
            csc.indices, self.indices
        ):
            raise ValueError("matrix sparsity pattern changed")
        return np.asarray(csc.data, dtype=np.float64).copy()


@dataclass(frozen=True, slots=True)
class CQPStructure:
    """Immutable symbolic structure of a conic quadratic problem."""

    quadratic: CSCStructure
    constraint: CSCStructure
    cones: tuple[ConeBlock, ...] = ()

    def __post_init__(self) -> None:
        rows, columns = self.quadratic.shape
        if rows != columns:
            raise ValueError("quadratic matrix must be square")
        if self.constraint.shape[1] != columns:
            raise ValueError("constraint matrix column count must equal variable count")

        previous_stop = 0
        for cone in sorted(self.cones, key=lambda item: item.start):
            if cone.start < previous_stop:
                raise ValueError("cone blocks must not overlap")
            if cone.start + cone.size > self.n_constraints:
                raise ValueError("cone block exceeds constraint row count")
            previous_stop = cone.start + cone.size

    @property
    def n_variables(self) -> int:
        return self.quadratic.shape[0]

    @property
    def n_constraints(self) -> int:
        return self.constraint.shape[0]


@dataclass(slots=True)
class CQPValues:
    """Mutable numerical values associated with a fixed :class:`CQPStructure`."""

    quadratic: FloatArray
    constraint: FloatArray
    linear: FloatArray
    lower: FloatArray
    upper: FloatArray

    def validated(self, structure: CQPStructure) -> CQPValues:
        """Return an owned, validated copy compatible with ``structure``."""

        quadratic = np.asarray(self.quadratic, dtype=np.float64).copy()
        constraint = np.asarray(self.constraint, dtype=np.float64).copy()
        linear = np.asarray(self.linear, dtype=np.float64).copy()
        lower = np.asarray(self.lower, dtype=np.float64).copy()
        upper = np.asarray(self.upper, dtype=np.float64).copy()

        expected = {
            "quadratic": (quadratic, structure.quadratic.nnz),
            "constraint": (constraint, structure.constraint.nnz),
            "linear": (linear, structure.n_variables),
            "lower": (lower, structure.n_constraints),
            "upper": (upper, structure.n_constraints),
        }
        for name, (array, size) in expected.items():
            if array.ndim != 1 or array.size != size:
                raise ValueError(f"{name} must have shape ({size},), received {array.shape}")

        if not np.all(np.isfinite(quadratic)):
            raise ValueError("quadratic values must be finite")
        if not np.all(np.isfinite(constraint)):
            raise ValueError("constraint values must be finite")
        if not np.all(np.isfinite(linear)):
            raise ValueError("linear objective must be finite")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("constraint bounds may be infinite but not NaN")
        if np.any(lower > upper):
            raise ValueError("lower constraint bound exceeds upper bound")

        return CQPValues(
            quadratic=quadratic,
            constraint=constraint,
            linear=linear,
            lower=lower,
            upper=upper,
        )

    def copy(self) -> CQPValues:
        return CQPValues(
            quadratic=self.quadratic.copy(),
            constraint=self.constraint.copy(),
            linear=self.linear.copy(),
            lower=self.lower.copy(),
            upper=self.upper.copy(),
        )


@dataclass(frozen=True, slots=True)
class CanonicalCQP:
    """A complete numerical CQP instance."""

    structure: CQPStructure
    values: CQPValues

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", self.values.validated(self.structure))


@dataclass(frozen=True, slots=True)
class CQPSolution:
    """Backend-independent solver result and residual summary."""

    status: str
    primal: FloatArray
    dual: FloatArray
    objective: float
    primal_residual: float
    dual_residual: float
    iterations: int
    solve_seconds: float

    @property
    def solved(self) -> bool:
        return self.status.lower().startswith("solved")
