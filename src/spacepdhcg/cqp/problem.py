"""Fixed-pattern conic quadratic problem representation.

The immutable :class:`CQPStructure` is allocated once. Successive solves replace only
:class:`CQPValues`. The canonical form mirrors PDHCG's native split between scalar
bounds, affine cone rows, variable bounds, and variable cone blocks.
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
    """Native nonpolyhedral cone families supported by the programme."""

    SECOND_ORDER = "second_order"
    ROTATED_SECOND_ORDER = "rotated_second_order"
    EXPONENTIAL = "exponential"
    POWER = "power"
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"


@dataclass(frozen=True, slots=True)
class ConeBlock:
    """A fixed native cone block.

    ``vector_dimension`` follows PDHCG's ``v_dim`` convention. SOC and rotated-SOC
    blocks therefore contain ``vector_dimension + 2`` scalar slots. Exponential and
    power cones contain three slots. PSD blocks contain an ``svec`` representation of
    a matrix whose order is ``vector_dimension``.
    """

    kind: ConeKind
    start: int
    vector_dimension: int
    power_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("cone start must be non-negative")
        if self.vector_dimension <= 0:
            raise ValueError("cone vector_dimension must be positive")
        if self.kind in {ConeKind.EXPONENTIAL, ConeKind.POWER} and self.vector_dimension != 1:
            raise ValueError("exponential and power cones require vector_dimension == 1")
        if self.kind is ConeKind.POWER:
            if not np.isfinite(self.power_alpha) or not 0.0 < self.power_alpha < 1.0:
                raise ValueError("power cone alpha must lie strictly between zero and one")
        elif self.power_alpha != 0.0:
            raise ValueError("power_alpha is only valid for power cones")

    @property
    def slot_count(self) -> int:
        if self.kind in {ConeKind.SECOND_ORDER, ConeKind.ROTATED_SECOND_ORDER}:
            return self.vector_dimension + 2
        if self.kind in {ConeKind.EXPONENTIAL, ConeKind.POWER}:
            return 3
        if self.kind is ConeKind.POSITIVE_SEMIDEFINITE:
            order = self.vector_dimension
            return order * (order + 1) // 2
        raise AssertionError(f"unhandled cone kind {self.kind}")

    @property
    def stop(self) -> int:
        return self.start + self.slot_count


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
    """Immutable symbolic structure of a native PDHCG-compatible CQP."""

    quadratic: CSCStructure
    constraint: CSCStructure
    affine_cone: CSCStructure | None = None
    affine_cones: tuple[ConeBlock, ...] = ()
    variable_cones: tuple[ConeBlock, ...] = ()

    def __post_init__(self) -> None:
        rows, columns = self.quadratic.shape
        if rows != columns:
            raise ValueError("quadratic matrix must be square")
        if self.constraint.shape[1] != columns:
            raise ValueError("constraint matrix column count must equal variable count")
        if self.affine_cone is None:
            if self.affine_cones:
                raise ValueError("affine cone blocks require an affine cone matrix")
        else:
            if self.affine_cone.shape[1] != columns:
                raise ValueError("affine cone matrix column count must equal variable count")
            self._validate_cones(
                self.affine_cones,
                self.affine_cone.shape[0],
                require_cover=True,
            )
        self._validate_cones(self.variable_cones, columns, require_cover=False)

    @staticmethod
    def _validate_cones(
        cones: tuple[ConeBlock, ...],
        ambient_dimension: int,
        *,
        require_cover: bool,
    ) -> None:
        previous_stop = 0
        for index, cone in enumerate(cones):
            if index and cone.start < previous_stop:
                raise ValueError("cone blocks must be sorted and non-overlapping")
            if require_cover and cone.start != previous_stop:
                raise ValueError("affine cone blocks must contiguously cover every affine row")
            if cone.stop > ambient_dimension:
                raise ValueError("cone block exceeds its ambient dimension")
            previous_stop = cone.stop
        if require_cover and previous_stop != ambient_dimension:
            raise ValueError("affine cone blocks must cover every affine row")

    @property
    def n_variables(self) -> int:
        return self.quadratic.shape[0]

    @property
    def n_constraints(self) -> int:
        return self.constraint.shape[0]

    @property
    def n_affine_constraints(self) -> int:
        return 0 if self.affine_cone is None else self.affine_cone.shape[0]

    @property
    def n_duals(self) -> int:
        """PDHCG dual size, ordered ``[dual_A, dual_F]``."""

        return self.n_constraints + self.n_affine_constraints


@dataclass(slots=True)
class CQPValues:
    """Mutable numerical values associated with a fixed :class:`CQPStructure`."""

    quadratic: FloatArray
    constraint: FloatArray
    linear: FloatArray
    lower: FloatArray
    upper: FloatArray
    affine_cone: FloatArray
    affine_offset: FloatArray
    variable_lower: FloatArray
    variable_upper: FloatArray

    def validated(self, structure: CQPStructure) -> CQPValues:
        """Return an owned, validated copy compatible with ``structure``."""

        arrays = {
            "quadratic": np.asarray(self.quadratic, dtype=np.float64).copy(),
            "constraint": np.asarray(self.constraint, dtype=np.float64).copy(),
            "linear": np.asarray(self.linear, dtype=np.float64).copy(),
            "lower": np.asarray(self.lower, dtype=np.float64).copy(),
            "upper": np.asarray(self.upper, dtype=np.float64).copy(),
            "affine_cone": np.asarray(self.affine_cone, dtype=np.float64).copy(),
            "affine_offset": np.asarray(self.affine_offset, dtype=np.float64).copy(),
            "variable_lower": np.asarray(self.variable_lower, dtype=np.float64).copy(),
            "variable_upper": np.asarray(self.variable_upper, dtype=np.float64).copy(),
        }
        expected_sizes = {
            "quadratic": structure.quadratic.nnz,
            "constraint": structure.constraint.nnz,
            "linear": structure.n_variables,
            "lower": structure.n_constraints,
            "upper": structure.n_constraints,
            "affine_cone": 0 if structure.affine_cone is None else structure.affine_cone.nnz,
            "affine_offset": structure.n_affine_constraints,
            "variable_lower": structure.n_variables,
            "variable_upper": structure.n_variables,
        }
        for name, size in expected_sizes.items():
            array = arrays[name]
            if array.ndim != 1 or array.size != size:
                raise ValueError(f"{name} must have shape ({size},), received {array.shape}")

        for name in ("quadratic", "constraint", "linear", "affine_cone", "affine_offset"):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{name} values must be finite")
        for name in ("lower", "upper", "variable_lower", "variable_upper"):
            if np.any(np.isnan(arrays[name])):
                raise ValueError(f"{name} may be infinite but not NaN")
        if np.any(arrays["lower"] > arrays["upper"]):
            raise ValueError("lower constraint bound exceeds upper bound")
        if np.any(arrays["variable_lower"] > arrays["variable_upper"]):
            raise ValueError("lower variable bound exceeds upper variable bound")

        return CQPValues(**arrays)

    def copy(self) -> CQPValues:
        return CQPValues(
            quadratic=self.quadratic.copy(),
            constraint=self.constraint.copy(),
            linear=self.linear.copy(),
            lower=self.lower.copy(),
            upper=self.upper.copy(),
            affine_cone=self.affine_cone.copy(),
            affine_offset=self.affine_offset.copy(),
            variable_lower=self.variable_lower.copy(),
            variable_upper=self.variable_upper.copy(),
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
