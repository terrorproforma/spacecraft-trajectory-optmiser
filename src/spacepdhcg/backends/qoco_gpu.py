"""Production adapter for the pinned QOCO GPU interior-point solver.

The adapter keeps QOCO's symbolic workspace alive across same-pattern updates.
Conversion and residual code is deliberately independent of QOCO so it can be
qualified with CPU-only tests before scarce GPU execution is scheduled.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TypeVar

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from spacepdhcg.cqp import (
    CanonicalCQP,
    ConeBlock,
    ConeKind,
    CQPSolution,
    CQPStructure,
    CQPValues,
)

FloatArray = NDArray[np.float64]
Int32Array = NDArray[np.int32]
Handback = TypeVar("Handback")

PINNED_QOCO_COMMIT = "09f049597deef2a7ead15b3da19a9456ff7d4e53"


class UnsupportedQOCOClass(StrEnum):
    """Stable, machine-readable reasons that a CQP cannot enter QOCO."""

    UNSUPPORTED_CONE = "unsupported_cone"
    NONSYMMETRIC_QUADRATIC = "nonsymmetric_quadratic"
    NONCONVEX_QUADRATIC = "nonconvex_quadratic"
    CHANGING_BOUND_STRUCTURE = "changing_bound_structure"


class QOCOAdapterError(RuntimeError):
    """Base class for adapter, conversion, setup, and solve failures."""


class QOCOUnavailableError(QOCOAdapterError):
    """Raised when the pinned QOCO shared library cannot be loaded."""


class QOCOUnsupportedError(QOCOAdapterError):
    """Raised with an explicit unsupported-formulation classification."""

    def __init__(self, classification: UnsupportedQOCOClass, detail: str) -> None:
        self.classification = classification
        self.detail = detail
        super().__init__(f"{classification.value}: {detail}")


class QOCOSetupError(QOCOAdapterError):
    """Raised when QOCO rejects or cannot allocate a workspace."""

    def __init__(self, code: int) -> None:
        self.code = int(code)
        names = {
            1: "data_validation",
            2: "settings_validation",
            3: "setup",
            4: "amd_ordering",
            5: "out_of_memory",
        }
        self.failure_class = names.get(self.code, "unknown_setup_failure")
        super().__init__(f"QOCO setup failed ({self.failure_class}, code={self.code})")


class QOCOSolveError(QOCOAdapterError):
    """Raised when the QOCO API fails before returning a solver status."""


class QOCOHybridIneligibleError(QOCOAdapterError):
    """Raised before QOCO when the PDHCG handoff fails the frozen gate."""

    def __init__(self, report: HybridRunReport) -> None:
        self.report = report
        super().__init__(report.reason)


@dataclass(frozen=True, slots=True)
class QOCOSettings:
    """Supported subset of pinned QOCO 0.3.2 settings."""

    max_iters: int = 200
    ruiz_iters: int = 0
    max_ir_iters: int = 5
    ir_tol: float = 1.0e-6
    kkt_static_reg_p: float = 1.0e-13
    kkt_static_reg_a: float = 1.0e-8
    kkt_static_reg_g: float = 1.0e-13
    kkt_dynamic_reg: float = 1.0e-11
    abstol: float = 1.0e-7
    reltol: float = 1.0e-7
    abstol_inaccurate: float = 1.0e-5
    reltol_inaccurate: float = 1.0e-5
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_iters <= 0 or self.max_ir_iters < 0 or self.ruiz_iters < 0:
            raise ValueError("QOCO iteration counts are invalid")
        positive = (
            self.ir_tol,
            self.kkt_static_reg_p,
            self.kkt_static_reg_a,
            self.kkt_static_reg_g,
            self.kkt_dynamic_reg,
            self.abstol,
            self.abstol_inaccurate,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("QOCO positive settings must be finite and positive")
        if self.reltol < 0.0 or self.reltol_inaccurate < 0.0:
            raise ValueError("QOCO relative tolerances must be non-negative")

    def with_solve_overrides(
        self,
        tolerance: float | None,
        iteration_limit: int | None,
    ) -> QOCOSettings:
        values = {field: getattr(self, field) for field in self.__dataclass_fields__}
        if tolerance is not None:
            if tolerance <= 0.0:
                raise ValueError("tolerance must be positive")
            values["abstol"] = float(tolerance)
            values["reltol"] = float(tolerance)
        if iteration_limit is not None:
            if iteration_limit <= 0:
                raise ValueError("iteration_limit must be positive")
            values["max_iters"] = int(iteration_limit)
        return QOCOSettings(**values)


@dataclass(frozen=True, slots=True)
class _MappedRow:
    source: str
    index: int
    side: str
    cone: ConeBlock | None = None
    transform: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class QOCOFormulation:
    """Owned QOCO arrays in its required equality/nonnegative/SOC ordering."""

    p: sp.csc_matrix
    c: FloatArray
    a: sp.csc_matrix
    b: FloatArray
    g: sp.csc_matrix
    h: FloatArray
    nonnegative_dimension: int
    soc_dimensions: Int32Array
    equality_rows: tuple[_MappedRow, ...]
    conic_rows: tuple[_MappedRow, ...]

    @property
    def n(self) -> int:
        return int(self.c.size)

    @property
    def m(self) -> int:
        return int(self.h.size)

    @property
    def equality_dimension(self) -> int:
        return int(self.b.size)


@dataclass(frozen=True, slots=True)
class CanonicalResiduals:
    """Residuals recomputed from original canonical data and QOCO multipliers."""

    primal: float
    dual: float
    equality: float
    scalar_bounds: float
    variable_bounds: float
    cones: float


@dataclass(frozen=True, slots=True)
class WarmStartReport:
    """Qualification and acceptance evidence for one requested warm start."""

    requested: bool
    primal_qualified: bool
    primal_accepted: bool
    primal_residual: float | None
    dual_requested: bool
    dual_discarded: bool
    reason: str


@dataclass(frozen=True, slots=True)
class QOCORunReport:
    """Adapter diagnostics kept outside the frozen :class:`CQPSolution` schema."""

    qoco_commit: str
    native_status: int
    native_primal_residual: float
    native_dual_residual: float
    native_gap: float
    setup_seconds: float
    conversion_seconds: float
    update_seconds: float
    solve_seconds: float
    analysis_seconds: float
    canonical_residuals: CanonicalResiduals
    warm_start: WarmStartReport
    failure_class: str | None


@dataclass(frozen=True, slots=True)
class QOCORawSolution:
    x: FloatArray
    y: FloatArray
    z: FloatArray
    objective: float
    primal_residual: float
    dual_residual: float
    gap: float
    iterations: int
    setup_seconds: float
    solve_seconds: float
    analysis_seconds: float
    status: int


class _QOCOAPI(Protocol):
    def setup(self, formulation: QOCOFormulation, settings: QOCOSettings) -> Any: ...

    def update(
        self,
        handle: Any,
        formulation: QOCOFormulation,
        settings: QOCOSettings,
    ) -> None: ...

    def set_primal_start(self, handle: Any, primal: FloatArray | None) -> None: ...

    def solve(self, handle: Any, formulation: QOCOFormulation) -> QOCORawSolution: ...

    def cleanup(self, handle: Any) -> None: ...


def _maximum(values: NDArray[np.floating[Any]]) -> float:
    return 0.0 if values.size == 0 else float(np.max(values))


def canonical_numeric_fingerprint(values: CQPValues) -> str:
    """Hash every canonical numeric array in fixed field order."""

    digest = hashlib.sha256()
    for name in (
        "quadratic",
        "constraint",
        "linear",
        "lower",
        "upper",
        "affine_cone",
        "affine_offset",
        "variable_lower",
        "variable_upper",
    ):
        array = np.ascontiguousarray(getattr(values, name), dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(array.size.to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bound_signature(lower: FloatArray, upper: FloatArray) -> tuple[str, ...]:
    signature: list[str] = []
    for lower_value, upper_value in zip(lower, upper, strict=True):
        if np.isfinite(lower_value) and np.isfinite(upper_value):
            signature.append("equality" if lower_value == upper_value else "box")
        elif np.isfinite(upper_value):
            signature.append("upper")
        elif np.isfinite(lower_value):
            signature.append("lower")
        else:
            signature.append("free")
    return tuple(signature)


def _cone_transform(cone: ConeBlock) -> FloatArray:
    """Map native ``[..., radius]`` coordinates to QOCO ``[radius, ...]``."""

    size = cone.slot_count
    transform = np.zeros((size, size), dtype=np.float64)
    if cone.kind is ConeKind.SECOND_ORDER:
        transform[0, size - 1] = 1.0
        transform[1:, :-1] = np.eye(size - 1)
        return transform
    if cone.kind is ConeKind.ROTATED_SECOND_ORDER:
        dimension = cone.vector_dimension
        scale = 1.0 / np.sqrt(2.0)
        transform[0, dimension] = scale
        transform[0, dimension + 1] = scale
        transform[1 : dimension + 1, :dimension] = np.eye(dimension)
        transform[-1, dimension] = scale
        transform[-1, dimension + 1] = -scale
        return transform
    raise QOCOUnsupportedError(
        UnsupportedQOCOClass.UNSUPPORTED_CONE,
        f"QOCO supports SOC and mapped rotated-SOC blocks, not {cone.kind.value}",
    )


def _quadratic_for_qoco(structure: CQPStructure, values: CQPValues) -> sp.csc_matrix:
    quadratic = structure.quadratic.matrix(values.quadratic)
    difference = quadratic - quadratic.T
    if difference.nnz and _maximum(np.abs(difference.data)) > 1.0e-12:
        raise QOCOUnsupportedError(
            UnsupportedQOCOClass.NONSYMMETRIC_QUADRATIC,
            "canonical Hessian must be symmetric for an exact upper-CSC QOCO mapping",
        )
    if quadratic.shape[0]:
        try:
            if quadratic.shape[0] <= 256:
                minimum_eigenvalue = float(np.linalg.eigvalsh(quadratic.toarray())[0])
            else:
                minimum_eigenvalue = float(
                    sp.linalg.eigsh(quadratic, k=1, which="SA", return_eigenvectors=False)[0]
                )
        except sp.linalg.ArpackNoConvergence as error:
            raise QOCOUnsupportedError(
                UnsupportedQOCOClass.NONCONVEX_QUADRATIC,
                "could not qualify the Hessian as positive semidefinite",
            ) from error
        scale = max(1.0, _maximum(np.abs(quadratic.data)))
        if minimum_eigenvalue < -1.0e-10 * scale:
            raise QOCOUnsupportedError(
                UnsupportedQOCOClass.NONCONVEX_QUADRATIC,
                f"minimum Hessian eigenvalue is {minimum_eigenvalue:.6e}",
            )
    upper = sp.triu(quadratic, format="csc")
    upper.sum_duplicates()
    upper.sort_indices()
    return upper


def convert_to_qoco(problem: CanonicalCQP) -> QOCOFormulation:
    """Convert canonical CQP data exactly into QOCO's product-cone form."""

    structure = problem.structure
    values = problem.values.validated(structure)
    n = structure.n_variables
    scalar = structure.constraint.matrix(values.constraint)
    identity = sp.eye(n, format="csc")

    equality_matrices: list[sp.spmatrix] = []
    equality_rhs: list[float] = []
    equality_rows: list[_MappedRow] = []
    nonnegative_matrices: list[sp.spmatrix] = []
    nonnegative_rhs: list[float] = []
    nonnegative_rows: list[_MappedRow] = []

    for source, matrix, lower, upper in (
        ("constraint", scalar, values.lower, values.upper),
        ("variable", identity, values.variable_lower, values.variable_upper),
    ):
        for index, kind in enumerate(_bound_signature(lower, upper)):
            row = matrix.getrow(index)
            if kind == "equality":
                equality_matrices.append(row)
                equality_rhs.append(float(lower[index]))
                equality_rows.append(_MappedRow(source, index, "equality"))
            elif kind in {"box", "upper"}:
                nonnegative_matrices.append(row)
                nonnegative_rhs.append(float(upper[index]))
                nonnegative_rows.append(_MappedRow(source, index, "upper"))
            if kind in {"box", "lower"}:
                nonnegative_matrices.append(-row)
                nonnegative_rhs.append(float(-lower[index]))
                nonnegative_rows.append(_MappedRow(source, index, "lower"))

    affine = (
        None if structure.affine_cone is None else structure.affine_cone.matrix(values.affine_cone)
    )
    soc_matrices: list[sp.spmatrix] = []
    soc_rhs: list[FloatArray] = []
    soc_rows: list[_MappedRow] = []
    soc_dimensions: list[int] = []
    for source, cones in (
        ("affine", structure.affine_cones),
        ("variable_cone", structure.variable_cones),
    ):
        for cone in cones:
            transform = _cone_transform(cone)
            if source == "affine":
                if affine is None:
                    raise AssertionError("affine cone metadata has no matrix")
                native_matrix = affine[cone.start : cone.stop, :]
                native_offset = values.affine_offset[cone.start : cone.stop]
            else:
                native_matrix = identity[cone.start : cone.stop, :]
                native_offset = np.zeros(cone.slot_count, dtype=np.float64)
            soc_matrices.append(-sp.csr_matrix(transform) @ native_matrix)
            soc_rhs.append(np.asarray(transform @ native_offset, dtype=np.float64))
            soc_dimensions.append(cone.slot_count)
            soc_rows.append(_MappedRow(source, cone.start, "cone", cone, transform))

    a = (
        sp.vstack(equality_matrices, format="csc")
        if equality_matrices
        else sp.csc_matrix((0, n), dtype=np.float64)
    )
    conic_matrices = [*nonnegative_matrices, *soc_matrices]
    g = (
        sp.vstack(conic_matrices, format="csc")
        if conic_matrices
        else sp.csc_matrix((0, n), dtype=np.float64)
    )
    for matrix in (a, g):
        matrix.sum_duplicates()
        matrix.sort_indices()
    h = np.concatenate(
        (
            np.asarray(nonnegative_rhs, dtype=np.float64),
            *(np.asarray(item, dtype=np.float64) for item in soc_rhs),
        )
    )
    return QOCOFormulation(
        p=_quadratic_for_qoco(structure, values),
        c=values.linear.copy(),
        a=a,
        b=np.asarray(equality_rhs, dtype=np.float64),
        g=g,
        h=h,
        nonnegative_dimension=len(nonnegative_rows),
        soc_dimensions=np.asarray(soc_dimensions, dtype=np.int32),
        equality_rows=tuple(equality_rows),
        conic_rows=tuple((*nonnegative_rows, *soc_rows)),
    )


def _soc_violation(value: FloatArray) -> float:
    return max(0.0, float(np.linalg.norm(value[1:]) - value[0]))


def canonical_primal_residual(problem: CanonicalCQP, primal: NDArray) -> float:
    """Maximum unscaled feasibility violation in original canonical coordinates."""

    structure = problem.structure
    values = problem.values
    x = np.asarray(primal, dtype=np.float64)
    if x.shape != (structure.n_variables,) or not np.all(np.isfinite(x)):
        return float("inf")
    scalar_value = np.asarray(
        structure.constraint.matrix(values.constraint) @ x,
        dtype=np.float64,
    )
    scalar = _maximum(
        np.maximum(
            np.maximum(values.lower - scalar_value, 0.0),
            np.maximum(scalar_value - values.upper, 0.0),
        )
    )
    variable = _maximum(
        np.maximum(
            np.maximum(values.variable_lower - x, 0.0),
            np.maximum(x - values.variable_upper, 0.0),
        )
    )
    cones = 0.0
    if structure.affine_cone is not None:
        affine_value = np.asarray(
            structure.affine_cone.matrix(values.affine_cone) @ x + values.affine_offset,
            dtype=np.float64,
        )
        for cone in structure.affine_cones:
            transformed = _cone_transform(cone) @ affine_value[cone.start : cone.stop]
            cones = max(cones, _soc_violation(transformed))
    for cone in structure.variable_cones:
        cones = max(cones, _soc_violation(_cone_transform(cone) @ x[cone.start : cone.stop]))
    return max(scalar, variable, cones)


def _canonical_dual(
    structure: CQPStructure,
    formulation: QOCOFormulation,
    y: FloatArray,
    z: FloatArray,
) -> FloatArray:
    dual = np.zeros(structure.n_duals, dtype=np.float64)
    for value, row in zip(y, formulation.equality_rows, strict=True):
        if row.source == "constraint":
            dual[row.index] += value
    cursor = 0
    for row in formulation.conic_rows:
        if row.cone is None:
            if row.source == "constraint":
                dual[row.index] += value_sign(row.side) * z[cursor]
            cursor += 1
        else:
            size = row.cone.slot_count
            segment = z[cursor : cursor + size]
            cursor += size
            if row.source == "affine":
                if row.transform is None:
                    raise AssertionError("cone dual is missing its transform")
                start = structure.n_constraints + row.cone.start
                dual[start : start + size] = row.transform.T @ segment
    return dual


def value_sign(side: str) -> float:
    return -1.0 if side == "lower" else 1.0


def independent_residuals(
    problem: CanonicalCQP,
    formulation: QOCOFormulation,
    raw: QOCORawSolution,
) -> CanonicalResiduals:
    """Recompute unscaled residuals without using QOCO's reported residuals."""

    x = raw.x
    values = problem.values
    scalar_value = np.asarray(
        problem.structure.constraint.matrix(values.constraint) @ x,
        dtype=np.float64,
    )
    equality_mask = np.isfinite(values.lower) & np.isfinite(values.upper)
    equality_mask &= values.lower == values.upper
    equality = _maximum(np.abs(scalar_value[equality_mask] - values.lower[equality_mask]))
    scalar_bounds = _maximum(
        np.maximum(
            np.maximum(values.lower - scalar_value, 0.0),
            np.maximum(scalar_value - values.upper, 0.0),
        )
    )
    variable_bounds = _maximum(
        np.maximum(
            np.maximum(values.variable_lower - x, 0.0),
            np.maximum(x - values.variable_upper, 0.0),
        )
    )
    slack = formulation.h - np.asarray(formulation.g @ x, dtype=np.float64)
    cone_violation = _maximum(np.maximum(-slack[: formulation.nonnegative_dimension], 0.0))
    cursor = formulation.nonnegative_dimension
    for dimension in formulation.soc_dimensions:
        segment = slack[cursor : cursor + dimension]
        cone_violation = max(cone_violation, _soc_violation(segment))
        cursor += int(dimension)
    full_p = formulation.p + sp.triu(formulation.p, k=1).T
    stationarity = np.asarray(
        full_p @ x + formulation.c + formulation.a.T @ raw.y + formulation.g.T @ raw.z,
        dtype=np.float64,
    )
    dual = _maximum(np.abs(stationarity))
    primal = max(equality, scalar_bounds, variable_bounds, cone_violation)
    return CanonicalResiduals(
        primal=primal,
        dual=dual,
        equality=equality,
        scalar_bounds=scalar_bounds,
        variable_bounds=variable_bounds,
        cones=cone_violation,
    )


class _CCSC(ct.Structure):
    _fields_ = [
        ("m", ct.c_int),
        ("n", ct.c_int),
        ("nnz", ct.c_int),
        ("i", ct.POINTER(ct.c_int)),
        ("p", ct.POINTER(ct.c_int)),
        ("x", ct.POINTER(ct.c_double)),
    ]


class _CSettings(ct.Structure):
    _fields_ = [
        ("max_iters", ct.c_int),
        ("ruiz_iters", ct.c_int),
        ("max_ir_iters", ct.c_int),
        ("ir_tol", ct.c_double),
        ("kkt_static_reg_p", ct.c_double),
        ("kkt_static_reg_a", ct.c_double),
        ("kkt_static_reg_g", ct.c_double),
        ("kkt_dynamic_reg", ct.c_double),
        ("abstol", ct.c_double),
        ("reltol", ct.c_double),
        ("abstol_inaccurate", ct.c_double),
        ("reltol_inaccurate", ct.c_double),
        ("verbose", ct.c_ubyte),
    ]


class _CSolution(ct.Structure):
    _fields_ = [
        ("x", ct.POINTER(ct.c_double)),
        ("s", ct.POINTER(ct.c_double)),
        ("y", ct.POINTER(ct.c_double)),
        ("z", ct.POINTER(ct.c_double)),
        ("iters", ct.c_int),
        ("ir_iters", ct.c_int),
        ("setup_time_sec", ct.c_double),
        ("solve_time_sec", ct.c_double),
        ("analysis_time_sec", ct.c_double),
        ("obj", ct.c_double),
        ("pres", ct.c_double),
        ("dres", ct.c_double),
        ("gap", ct.c_double),
        ("status", ct.c_int),
    ]


class _CSolver(ct.Structure):
    _fields_ = [
        ("settings", ct.c_void_p),
        ("work", ct.c_void_p),
        ("linsys", ct.c_void_p),
        ("linsys_data", ct.c_void_p),
        ("sol", ct.POINTER(_CSolution)),
    ]


@dataclass(slots=True)
class _CHandle:
    solver: ct.POINTER(_CSolver)


class CtypesQOCOAPI:
    """Thin owner of the pinned QOCO C ABI."""

    def __init__(self, library_path: str | Path) -> None:
        path = Path(library_path).expanduser()
        if not path.is_file():
            raise QOCOUnavailableError(f"QOCO shared library does not exist: {path}")
        try:
            self._library = ct.CDLL(str(path))
        except OSError as error:
            raise QOCOUnavailableError(f"could not load QOCO shared library: {path}") from error
        self._libc = ct.CDLL(None)
        self._libc.calloc.argtypes = [ct.c_size_t, ct.c_size_t]
        self._libc.calloc.restype = ct.c_void_p
        self._libc.free.argtypes = [ct.c_void_p]
        self._configure()

    def _configure(self) -> None:
        library = self._library
        matrix_pointer = ct.POINTER(_CCSC)
        float_pointer = ct.POINTER(ct.c_double)
        library.qoco_setup.argtypes = [
            ct.POINTER(_CSolver),
            ct.c_int,
            ct.c_int,
            ct.c_int,
            matrix_pointer,
            float_pointer,
            matrix_pointer,
            float_pointer,
            matrix_pointer,
            float_pointer,
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(_CSettings),
        ]
        library.qoco_setup.restype = ct.c_int
        library.qoco_update_settings.argtypes = [ct.POINTER(_CSolver), ct.POINTER(_CSettings)]
        library.qoco_update_settings.restype = ct.c_int
        library.qoco_update_vector_data.argtypes = [
            ct.POINTER(_CSolver),
            float_pointer,
            float_pointer,
            float_pointer,
        ]
        library.qoco_update_matrix_data.argtypes = [
            ct.POINTER(_CSolver),
            float_pointer,
            float_pointer,
            float_pointer,
        ]
        library.qoco_set_x0.argtypes = [ct.POINTER(_CSolver), float_pointer]
        library.qoco_solve.argtypes = [ct.POINTER(_CSolver)]
        library.qoco_solve.restype = ct.c_int
        library.qoco_cleanup.argtypes = [ct.POINTER(_CSolver)]
        library.qoco_cleanup.restype = ct.c_int

    @staticmethod
    def _settings(settings: QOCOSettings) -> _CSettings:
        return _CSettings(
            settings.max_iters,
            settings.ruiz_iters,
            settings.max_ir_iters,
            settings.ir_tol,
            settings.kkt_static_reg_p,
            settings.kkt_static_reg_a,
            settings.kkt_static_reg_g,
            settings.kkt_dynamic_reg,
            settings.abstol,
            settings.reltol,
            settings.abstol_inaccurate,
            settings.reltol_inaccurate,
            int(settings.verbose),
        )

    @staticmethod
    def _array(values: NDArray, dtype: np.dtype[Any]) -> NDArray:
        return np.ascontiguousarray(values, dtype=dtype)

    def _matrix(
        self,
        matrix: sp.csc_matrix,
    ) -> tuple[_CCSC, tuple[FloatArray, Int32Array, Int32Array]]:
        data = self._array(matrix.data, np.dtype(np.float64))
        indices = self._array(matrix.indices, np.dtype(np.int32))
        indptr = self._array(matrix.indptr, np.dtype(np.int32))
        csc = _CCSC(
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
            indices.ctypes.data_as(ct.POINTER(ct.c_int)),
            indptr.ctypes.data_as(ct.POINTER(ct.c_int)),
            data.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        return csc, (data, indices, indptr)

    def setup(self, formulation: QOCOFormulation, settings: QOCOSettings) -> _CHandle:
        storage = self._libc.calloc(1, ct.sizeof(_CSolver))
        if not storage:
            raise QOCOSetupError(5)
        solver = ct.cast(storage, ct.POINTER(_CSolver))
        p, p_owner = self._matrix(formulation.p)
        a, a_owner = self._matrix(formulation.a)
        g, g_owner = self._matrix(formulation.g)
        owners = (p_owner, a_owner, g_owner)
        del owners
        c = self._array(formulation.c, np.dtype(np.float64))
        b = self._array(formulation.b, np.dtype(np.float64))
        h = self._array(formulation.h, np.dtype(np.float64))
        q = self._array(formulation.soc_dimensions, np.dtype(np.int32))
        configured = self._settings(settings)
        code = self._library.qoco_setup(
            solver,
            formulation.n,
            formulation.m,
            formulation.equality_dimension,
            ct.byref(p),
            c.ctypes.data_as(ct.POINTER(ct.c_double)),
            None if formulation.equality_dimension == 0 else ct.byref(a),
            (
                None
                if formulation.equality_dimension == 0
                else b.ctypes.data_as(ct.POINTER(ct.c_double))
            ),
            None if formulation.m == 0 else ct.byref(g),
            None if formulation.m == 0 else h.ctypes.data_as(ct.POINTER(ct.c_double)),
            formulation.nonnegative_dimension,
            int(q.size),
            None if q.size == 0 else q.ctypes.data_as(ct.POINTER(ct.c_int)),
            ct.byref(configured),
        )
        if code != 0:
            # Pinned QOCO only owns the solver allocation after successful setup.
            self._libc.free(storage)
            raise QOCOSetupError(code)
        return _CHandle(solver)

    def update(
        self,
        handle: _CHandle,
        formulation: QOCOFormulation,
        settings: QOCOSettings,
    ) -> None:
        configured = self._settings(settings)
        code = self._library.qoco_update_settings(handle.solver, ct.byref(configured))
        if code != 0:
            raise QOCOSetupError(code)
        self._library.qoco_update_matrix_data(
            handle.solver,
            formulation.p.data.ctypes.data_as(ct.POINTER(ct.c_double)),
            None
            if formulation.a.nnz == 0
            else formulation.a.data.ctypes.data_as(ct.POINTER(ct.c_double)),
            None
            if formulation.g.nnz == 0
            else formulation.g.data.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        self._library.qoco_update_vector_data(
            handle.solver,
            formulation.c.ctypes.data_as(ct.POINTER(ct.c_double)),
            None
            if formulation.b.size == 0
            else formulation.b.ctypes.data_as(ct.POINTER(ct.c_double)),
            None
            if formulation.h.size == 0
            else formulation.h.ctypes.data_as(ct.POINTER(ct.c_double)),
        )

    def set_primal_start(self, handle: _CHandle, primal: FloatArray | None) -> None:
        pointer = None if primal is None else primal.ctypes.data_as(ct.POINTER(ct.c_double))
        self._library.qoco_set_x0(handle.solver, pointer)

    def solve(self, handle: _CHandle, formulation: QOCOFormulation) -> QOCORawSolution:
        returned_status = int(self._library.qoco_solve(handle.solver))
        if not handle.solver.contents.sol:
            raise QOCOSolveError("QOCO returned without a solution structure")
        solution = handle.solver.contents.sol.contents
        if returned_status != int(solution.status):
            raise QOCOSolveError(
                f"QOCO status disagreement: return={returned_status}, solution={solution.status}"
            )

        def vector(pointer: ct.POINTER(ct.c_double), size: int) -> FloatArray:
            if size == 0:
                return np.empty(0, dtype=np.float64)
            if not pointer:
                raise QOCOSolveError("QOCO returned a null solution vector")
            return np.ctypeslib.as_array(pointer, shape=(size,)).copy()

        return QOCORawSolution(
            x=vector(solution.x, formulation.n),
            y=vector(solution.y, formulation.equality_dimension),
            z=vector(solution.z, formulation.m),
            objective=float(solution.obj),
            primal_residual=float(solution.pres),
            dual_residual=float(solution.dres),
            gap=float(solution.gap),
            iterations=int(solution.iters),
            setup_seconds=float(solution.setup_time_sec),
            solve_seconds=float(solution.solve_time_sec),
            analysis_seconds=float(solution.analysis_time_sec),
            status=int(solution.status),
        )

    def cleanup(self, handle: _CHandle) -> None:
        # This pinned revision returns 1 after successful cleanup despite the
        # public header documenting zero, and frees the QOCOSolver allocation.
        code = int(self._library.qoco_cleanup(handle.solver))
        if code not in {0, 1}:
            raise QOCOAdapterError(f"QOCO cleanup failed (code={code})")


class QOCOGPU:
    """Persistent exact QP/SOCP adapter for QOCO's CUDA or builtin library."""

    is_persistent = True
    supports_dynamic_solve_settings = True

    def __init__(
        self,
        problem: CanonicalCQP,
        *,
        library_path: str | Path | None = None,
        settings: QOCOSettings | None = None,
        qoco_api: _QOCOAPI | None = None,
        warm_start_primal_tolerance: float = np.inf,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> None:
        if warm_start_primal_tolerance <= 0.0:
            raise ValueError("warm_start_primal_tolerance must be positive")
        self.structure = problem.structure
        self._current = CanonicalCQP(self.structure, problem.values)
        self._settings = (settings or QOCOSettings()).with_solve_overrides(
            tolerance,
            iteration_limit,
        )
        self._warm_start_primal_tolerance = float(warm_start_primal_tolerance)
        self._api = qoco_api or CtypesQOCOAPI(library_path or Path("build/qoco-g4/libqoco.so"))
        conversion_start = perf_counter()
        self._formulation = convert_to_qoco(self._current)
        self.conversion_seconds = perf_counter() - conversion_start
        setup_start = perf_counter()
        self._handle = self._api.setup(self._formulation, self._settings)
        self.setup_seconds = perf_counter() - setup_start
        self.update_count = 0
        self.warm_start_count = 0
        self.solve_count = 0
        self.last_update_seconds = 0.0
        self.last_report: QOCORunReport | None = None
        self._closed = False
        self._warm_report = WarmStartReport(False, False, False, None, False, False, "none")

    @property
    def current_values(self) -> CQPValues:
        return self._current.values.copy()

    def _ensure_open(self) -> None:
        if self._closed:
            raise QOCOAdapterError("QOCO workspace is closed")

    @staticmethod
    def _same_pattern(left: sp.csc_matrix, right: sp.csc_matrix) -> bool:
        return (
            left.shape == right.shape
            and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices)
        )

    def update(self, values: CQPValues) -> None:
        self._ensure_open()
        candidate = CanonicalCQP(self.structure, values)
        start = perf_counter()
        converted = convert_to_qoco(candidate)
        for old, new in (
            (self._formulation.p, converted.p),
            (self._formulation.a, converted.a),
            (self._formulation.g, converted.g),
        ):
            if not self._same_pattern(old, new):
                raise QOCOUnsupportedError(
                    UnsupportedQOCOClass.CHANGING_BOUND_STRUCTURE,
                    "numeric update changed QOCO row or CSC sparsity structure",
                )
        if (
            self._formulation.nonnegative_dimension != converted.nonnegative_dimension
            or not np.array_equal(
                self._formulation.soc_dimensions,
                converted.soc_dimensions,
            )
        ):
            raise QOCOUnsupportedError(
                UnsupportedQOCOClass.CHANGING_BOUND_STRUCTURE,
                "numeric update changed QOCO cone structure",
            )
        self._api.update(self._handle, converted, self._settings)
        self._current = candidate
        self._formulation = converted
        self.last_update_seconds = perf_counter() - start
        self.update_count += 1

    def warm_start(
        self,
        primal: NDArray | None = None,
        dual: NDArray | None = None,
    ) -> None:
        self._ensure_open()
        if primal is None and dual is None:
            raise ValueError("at least one warm-start vector is required")
        dual_requested = dual is not None
        if dual is not None:
            candidate_dual = np.asarray(dual, dtype=np.float64)
            if candidate_dual.shape != (self.structure.n_duals,):
                raise ValueError("dual warm start has the wrong shape")
        residual: float | None = None
        accepted = False
        qualified = False
        reason = "QOCO has no dual warm-start API"
        if primal is not None:
            candidate = np.asarray(primal, dtype=np.float64)
            if candidate.shape != (self.structure.n_variables,):
                raise ValueError("primal warm start has the wrong shape")
            residual = canonical_primal_residual(self._current, candidate)
            qualified = bool(
                np.all(np.isfinite(candidate)) and residual <= self._warm_start_primal_tolerance
            )
            if qualified:
                owned = np.ascontiguousarray(candidate, dtype=np.float64)
                self._api.set_primal_start(self._handle, owned)
                accepted = True
                reason = "qualified primal accepted"
                if dual_requested:
                    reason += "; dual discarded because QOCO is primal-only"
            else:
                self._api.set_primal_start(self._handle, None)
                reason = "primal rejected by finite/feasibility qualification"
                if dual_requested:
                    reason += "; dual discarded because QOCO is primal-only"
        else:
            self._api.set_primal_start(self._handle, None)
        self._warm_report = WarmStartReport(
            requested=True,
            primal_qualified=qualified,
            primal_accepted=accepted,
            primal_residual=residual,
            dual_requested=dual_requested,
            dual_discarded=dual_requested,
            reason=reason,
        )
        self.warm_start_count += 1

    def clear_warm_start(self) -> None:
        self._ensure_open()
        self._api.set_primal_start(self._handle, None)
        self._warm_report = WarmStartReport(False, False, False, None, False, False, "cleared")

    def solve(
        self,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> CQPSolution:
        self._ensure_open()
        requested = self._settings.with_solve_overrides(tolerance, iteration_limit)
        if requested != self._settings:
            self._api.update(self._handle, self._formulation, requested)
            self._settings = requested
        raw = self._api.solve(self._handle, self._formulation)
        residuals = independent_residuals(self._current, self._formulation, raw)
        dual = _canonical_dual(self.structure, self._formulation, raw.y, raw.z)
        statuses = {
            0: ("Unsolved", "unsolved"),
            1: ("Solved (QOCO_SOLVED)", None),
            2: ("Solved inaccurate (QOCO_SOLVED_INACCURATE)", None),
            3: ("Numerical failure (QOCO_NUMERICAL_ERROR)", "numerical_failure"),
            4: ("Iteration limit (QOCO_MAX_ITER)", "iteration_limit"),
        }
        status, failure = statuses.get(raw.status, (f"QOCO status {raw.status}", "unknown"))
        self.solve_count += 1
        self.last_report = QOCORunReport(
            qoco_commit=PINNED_QOCO_COMMIT,
            native_status=raw.status,
            native_primal_residual=raw.primal_residual,
            native_dual_residual=raw.dual_residual,
            native_gap=raw.gap,
            setup_seconds=raw.setup_seconds or self.setup_seconds,
            conversion_seconds=self.conversion_seconds,
            update_seconds=self.last_update_seconds,
            solve_seconds=raw.solve_seconds,
            analysis_seconds=raw.analysis_seconds,
            canonical_residuals=residuals,
            warm_start=self._warm_report,
            failure_class=failure,
        )
        return CQPSolution(
            status=status,
            primal=raw.x.copy(),
            dual=dual,
            objective=float(
                0.5 * raw.x @ ((self._formulation.p + sp.triu(self._formulation.p, k=1).T) @ raw.x)
                + self._formulation.c @ raw.x
            ),
            primal_residual=residuals.primal,
            dual_residual=residuals.dual,
            iterations=raw.iterations,
            solve_seconds=raw.solve_seconds,
        )

    def solve_and_handback(
        self,
        handback: Callable[[CQPSolution], Handback],
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> tuple[CQPSolution, Handback]:
        """Solve and pass the unscaled canonical result to a nonlinear owner."""

        solution = self.solve(tolerance=tolerance, iteration_limit=iteration_limit)
        if not solution.solved:
            failure = self.last_report.failure_class if self.last_report else "unknown"
            raise QOCOSolveError(f"QOCO result cannot be handed back ({failure})")
        return solution, handback(solution)

    def solve_outer_candidate(
        self,
        owner: Any,
        context: Any,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> tuple[CQPSolution, Any]:
        """Solve and execute the production pure-GPU-IPM nonlinear handback."""

        from spacepdhcg.backends.qoco_handback import handback_qoco_candidate

        solution = self.solve(tolerance=tolerance, iteration_limit=iteration_limit)
        return solution, handback_qoco_candidate(self, solution, owner, context)

    def close(self) -> None:
        if not self._closed:
            self._api.cleanup(self._handle)
            self._closed = True

    def __enter__(self) -> QOCOGPU:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class HybridRunReport:
    pdhcg_status: str
    pdhcg_seconds: float
    polish_seconds: float
    warm_start: WarmStartReport
    qoco: QOCORunReport | None
    eligible: bool
    disposition: str
    reason: str
    independent_primal_residual: float
    reported_primal_residual: float
    reported_dual_residual: float
    dual_disposition: str
    cqp_numeric_fingerprint: str
    fingerprint_match: bool


class PDHCGQOCOHybrid:
    """Run PDHCG first, then conditionally hand its primal to QOCO for polish."""

    def __init__(
        self,
        pdhcg: Any,
        qoco: QOCOGPU,
        *,
        handoff_tolerance: float = 1.0e-6,
    ) -> None:
        if handoff_tolerance <= 0.0:
            raise ValueError("handoff_tolerance must be positive")
        if pdhcg.structure is not qoco.structure and pdhcg.structure != qoco.structure:
            raise ValueError("PDHCG and QOCO must own the same canonical structure")
        self.pdhcg = pdhcg
        self.qoco = qoco
        self.structure = qoco.structure
        self.setup_seconds = qoco.setup_seconds + float(getattr(pdhcg, "setup_seconds", 0.0))
        self.update_count = 0
        self.warm_start_count = 0
        self.solve_count = 0
        self.handoff_tolerance = float(handoff_tolerance)
        self.last_report: HybridRunReport | None = None

    def update(self, values: CQPValues) -> None:
        self.pdhcg.update(values)
        self.qoco.update(values)
        self.update_count += 1

    def warm_start(
        self,
        primal: NDArray | None = None,
        dual: NDArray | None = None,
    ) -> None:
        """Seed the PDHCG predictor; its qualified primal will seed QOCO."""

        self.pdhcg.warm_start(primal, dual)
        self.warm_start_count += 1

    def solve(
        self,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
        handback: Callable[[CQPSolution], Handback] | None = None,
    ) -> CQPSolution | tuple[CQPSolution, Handback]:
        pdhcg_result = self.pdhcg.solve(
            tolerance=tolerance,
            iteration_limit=iteration_limit,
        )
        independent_primal = canonical_primal_residual(
            self.qoco._current,
            pdhcg_result.primal,
        )
        qoco_fingerprint = canonical_numeric_fingerprint(self.qoco.current_values)
        predictor_values = getattr(
            self.pdhcg,
            "current_values",
            getattr(self.pdhcg, "values", None),
        )
        predictor_fingerprint = (
            canonical_numeric_fingerprint(predictor_values)
            if isinstance(predictor_values, CQPValues)
            else None
        )
        fingerprint_match = predictor_fingerprint == qoco_fingerprint
        finite_dual = bool(
            pdhcg_result.dual.shape == (self.structure.n_duals,)
            and np.all(np.isfinite(pdhcg_result.dual))
        )
        eligible = bool(
            pdhcg_result.solved
            and finite_dual
            and fingerprint_match
            and independent_primal <= self.handoff_tolerance
            and pdhcg_result.primal_residual <= self.handoff_tolerance
            and pdhcg_result.dual_residual <= self.handoff_tolerance
        )
        if not eligible:
            self.qoco.clear_warm_start()
            failures: list[str] = []
            if not pdhcg_result.solved:
                failures.append("PDHCG did not solve")
            if independent_primal > self.handoff_tolerance:
                failures.append("independent primal residual exceeds gate")
            if pdhcg_result.primal_residual > self.handoff_tolerance:
                failures.append("reported primal residual exceeds gate")
            if pdhcg_result.dual_residual > self.handoff_tolerance:
                failures.append("reported dual residual exceeds gate")
            if not finite_dual:
                failures.append("dual is non-finite or has the wrong ordering")
            if not fingerprint_match:
                failures.append("PDHCG and QOCO CQP fingerprints differ")
            warm = WarmStartReport(
                requested=True,
                primal_qualified=False,
                primal_accepted=False,
                primal_residual=independent_primal,
                dual_requested=True,
                dual_discarded=True,
                reason="hybrid ineligible before QOCO: "
                + "; ".join(failures)
                + "; dual discarded because QOCO is primal-only",
            )
            self.last_report = HybridRunReport(
                pdhcg_status=pdhcg_result.status,
                pdhcg_seconds=pdhcg_result.solve_seconds,
                polish_seconds=0.0,
                warm_start=warm,
                qoco=None,
                eligible=False,
                disposition="ineligible",
                reason=warm.reason,
                independent_primal_residual=independent_primal,
                reported_primal_residual=pdhcg_result.primal_residual,
                reported_dual_residual=pdhcg_result.dual_residual,
                dual_disposition="discarded-unsupported-by-pinned-qoco",
                cqp_numeric_fingerprint=qoco_fingerprint,
                fingerprint_match=fingerprint_match,
            )
            raise QOCOHybridIneligibleError(self.last_report)
        self.qoco.warm_start(pdhcg_result.primal, pdhcg_result.dual)
        polish_start = perf_counter()
        qoco_result = self.qoco.solve(
            tolerance=tolerance,
            iteration_limit=iteration_limit,
        )
        polish_seconds = perf_counter() - polish_start
        self.solve_count += 1
        if self.qoco.last_report is None:
            raise AssertionError("QOCO solve completed without a report")
        self.last_report = HybridRunReport(
            pdhcg_status=pdhcg_result.status,
            pdhcg_seconds=pdhcg_result.solve_seconds,
            polish_seconds=polish_seconds,
            warm_start=self.qoco.last_report.warm_start,
            qoco=self.qoco.last_report,
            eligible=True,
            disposition="eligible-primal-only-polish",
            reason="PDHCG handoff passed the frozen quality gate; dual discarded",
            independent_primal_residual=independent_primal,
            reported_primal_residual=pdhcg_result.primal_residual,
            reported_dual_residual=pdhcg_result.dual_residual,
            dual_disposition="discarded-unsupported-by-pinned-qoco",
            cqp_numeric_fingerprint=qoco_fingerprint,
            fingerprint_match=True,
        )
        if handback is None:
            return qoco_result
        if not qoco_result.solved:
            raise QOCOSolveError(
                f"hybrid polish cannot be handed back ({self.qoco.last_report.failure_class})"
            )
        return qoco_result, handback(qoco_result)

    def close(self) -> None:
        self.qoco.close()

    def solve_outer_candidate(
        self,
        owner: Any,
        context: Any,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> tuple[CQPSolution, Any]:
        """Run a qualified predictor/polish and its distinct hybrid handback."""

        from spacepdhcg.backends.qoco_handback import (
            QOCOSolverMode,
            handback_qoco_candidate,
        )

        solution = self.solve(tolerance=tolerance, iteration_limit=iteration_limit)
        if not isinstance(solution, CQPSolution):
            raise AssertionError("hybrid solve unexpectedly returned a callback result")
        if self.last_report is None:
            raise AssertionError("hybrid solve completed without a report")
        record = handback_qoco_candidate(
            self.qoco,
            solution,
            owner,
            context,
            mode=QOCOSolverMode.HYBRID_PDHCG_IPM,
            predictor_seconds=self.last_report.pdhcg_seconds,
        )
        return solution, record

    def __enter__(self) -> PDHCGQOCOHybrid:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
