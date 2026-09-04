"""ctypes bindings for the planner transcription C ABI in ``libspacepdhcg``.

The CPU reference solver uses these bindings so that every family shares the exact C++
transcription, dynamics, independent RK4 replay, and quality metrics used by the CUDA
executable.  No solver runs here; the bindings only expose topology, coefficients,
references, replays, and evaluations.
"""

from __future__ import annotations

import ctypes
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.cqp import ConeBlock, ConeKind, CQPStructure, CQPValues, CSCStructure
from spacepdhcg.native import NativeLibraryError, load_native_library

FloatArray = NDArray[np.float64]

_MAX_PATH_COMPONENTS = 8
_CONE_KINDS = (
    ConeKind.SECOND_ORDER,
    ConeKind.ROTATED_SECOND_ORDER,
    ConeKind.EXPONENTIAL,
    ConeKind.POWER,
    ConeKind.POSITIVE_SEMIDEFINITE,
)


class PlannerNativeError(RuntimeError):
    """Raised when the native planner ABI is unavailable or rejects a request."""


class _Dimensions(ctypes.Structure):
    _fields_ = [
        ("state_dimension", ctypes.c_uint64),
        ("control_dimension", ctypes.c_uint64),
        ("intervals", ctypes.c_uint64),
        ("terminal_dimension", ctypes.c_uint64),
        ("variables", ctypes.c_uint64),
        ("scalar_rows", ctypes.c_uint64),
        ("affine_rows", ctypes.c_uint64),
        ("quadratic_nonzeros", ctypes.c_uint64),
        ("scalar_nonzeros", ctypes.c_uint64),
        ("affine_nonzeros", ctypes.c_uint64),
        ("affine_cone_count", ctypes.c_uint64),
        ("variable_cone_count", ctypes.c_uint64),
        ("virtual_variable_count", ctypes.c_uint64),
        ("step_seconds", ctypes.c_double),
        ("initial_trust_radius", ctypes.c_double),
    ]


class _Cone(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("start", ctypes.c_int32),
        ("vector_dimension", ctypes.c_int32),
        ("power_alpha", ctypes.c_double),
    ]


class _Evaluation(ctypes.Structure):
    _fields_ = [
        ("objective", ctypes.c_double),
        ("path_violation", ctypes.c_double),
        ("terminal_residual", ctypes.c_double),
        ("terminal_position_error", ctypes.c_double),
        ("terminal_velocity_error", ctypes.c_double),
        ("propellant_used", ctypes.c_double),
        ("final_mass", ctypes.c_double),
        ("path_component_count", ctypes.c_uint64),
        ("path_normalised", ctypes.c_double * _MAX_PATH_COMPONENTS),
        ("path_physical", ctypes.c_double * _MAX_PATH_COMPONENTS),
        ("path_names", (ctypes.c_char * 32) * _MAX_PATH_COMPONENTS),
    ]


@dataclass(frozen=True, slots=True)
class Dimensions:
    state_dimension: int
    control_dimension: int
    intervals: int
    terminal_dimension: int
    variables: int
    scalar_rows: int
    affine_rows: int
    virtual_variable_count: int
    step_seconds: float
    initial_trust_radius: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Device-equivalent nonlinear quality of a node trajectory."""

    objective: float
    path_violation: float
    path_components: dict[str, float]
    path_components_physical: dict[str, float]
    terminal_residual: float
    terminal_position_error: float
    terminal_velocity_error: float
    propellant_used: float
    final_mass: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "path_violation": self.path_violation,
            "path_components": dict(self.path_components),
            "path_components_physical": dict(self.path_components_physical),
            "terminal_residual": self.terminal_residual,
            "terminal_position_error": self.terminal_position_error,
            "terminal_velocity_error": self.terminal_velocity_error,
            "propellant_used": self.propellant_used,
            "final_mass": self.final_mass,
        }


_DoubleP = ctypes.POINTER(ctypes.c_double)
_Int32P = ctypes.POINTER(ctypes.c_int32)
_ConeP = ctypes.POINTER(_Cone)


def _configure(library: ctypes.CDLL) -> ctypes.CDLL:
    try:
        library.spacepdhcg_planner_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.spacepdhcg_planner_create.restype = ctypes.c_int
        library.spacepdhcg_planner_destroy.argtypes = [ctypes.c_void_p]
        library.spacepdhcg_planner_destroy.restype = None
        library.spacepdhcg_planner_get_dimensions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Dimensions),
        ]
        library.spacepdhcg_planner_get_dimensions.restype = ctypes.c_int
        library.spacepdhcg_planner_structure.argtypes = [
            ctypes.c_void_p,
            _Int32P,
            _Int32P,
            _Int32P,
            _Int32P,
            _Int32P,
            _Int32P,
            _ConeP,
            _ConeP,
            _Int32P,
            _Int32P,
            _Int32P,
        ]
        library.spacepdhcg_planner_structure.restype = ctypes.c_int
        library.spacepdhcg_planner_values.argtypes = [
            ctypes.c_void_p,
            _DoubleP,
            _DoubleP,
            ctypes.c_double,
        ] + [_DoubleP] * 9
        library.spacepdhcg_planner_values.restype = ctypes.c_int
        library.spacepdhcg_planner_initial_reference.argtypes = [
            ctypes.c_void_p,
            _DoubleP,
            _DoubleP,
        ]
        library.spacepdhcg_planner_initial_reference.restype = ctypes.c_int
        library.spacepdhcg_planner_rollout.argtypes = [
            ctypes.c_void_p,
            _DoubleP,
            _DoubleP,
            ctypes.c_uint64,
            ctypes.c_uint64,
            _DoubleP,
        ]
        library.spacepdhcg_planner_rollout.restype = ctypes.c_int
        library.spacepdhcg_planner_evaluate.argtypes = [
            ctypes.c_void_p,
            _DoubleP,
            _DoubleP,
            ctypes.POINTER(_Evaluation),
        ]
        library.spacepdhcg_planner_evaluate.restype = ctypes.c_int
        library.spacepdhcg_planner_path_components.argtypes = [
            ctypes.c_void_p,
            _DoubleP,
            _DoubleP,
            ctypes.c_uint64,
            ctypes.POINTER(_Evaluation),
        ]
        library.spacepdhcg_planner_path_components.restype = ctypes.c_int
        library.spacepdhcg_planner_describe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.spacepdhcg_planner_describe.restype = ctypes.c_int
        library.spacepdhcg_planner_default_document.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.spacepdhcg_planner_default_document.restype = ctypes.c_int
    except AttributeError as error:
        raise PlannerNativeError(
            "the loaded libspacepdhcg does not export the planner transcription ABI; rebuild the "
            "native library (cpp/src/c_api.cpp) or point SPACEPDHCG_NATIVE_LIBRARY at a build that "
            f"includes it ({error})"
        ) from error
    return library


_LIBRARY: ctypes.CDLL | None = None


def load_planner_library() -> ctypes.CDLL:
    """Load the packaged (or ``SPACEPDHCG_NATIVE_LIBRARY``) library with planner bindings."""

    global _LIBRARY
    if _LIBRARY is None:
        try:
            _LIBRARY = _configure(load_native_library())
        except NativeLibraryError as error:
            raise PlannerNativeError(str(error)) from error
    return _LIBRARY


def planner_library_available() -> bool:
    try:
        load_planner_library()
    except PlannerNativeError:
        return False
    return True


def _last_error(library: ctypes.CDLL) -> str:
    raw = library.spacepdhcg_last_error()
    return raw.decode("utf-8", "replace") if raw else "unknown native planner error"


def _check(library: ctypes.CDLL, status: int, operation: str) -> None:
    if status != 0:
        raise PlannerNativeError(f"{operation}: {_last_error(library)}")


def _pointer(array: FloatArray) -> Any:
    return array.ctypes.data_as(_DoubleP)


def _int_pointer(array: NDArray[np.int32]) -> Any:
    return array.ctypes.data_as(_Int32P)


def _string_result(library: ctypes.CDLL, call: Any, *arguments: Any) -> str:
    required = ctypes.c_size_t(0)
    _check(library, call(*arguments, None, 0, ctypes.byref(required)), "planner string size")
    buffer = ctypes.create_string_buffer(int(required.value))
    _check(
        library,
        call(*arguments, buffer, int(required.value), ctypes.byref(required)),
        "planner string",
    )
    return buffer.value.decode("utf-8")


def native_default_document(family: str) -> dict[str, Any]:
    """Native family defaults (vehicle, environment, constraints, weights, orders, units)."""

    library = load_planner_library()
    text = _string_result(
        library, library.spacepdhcg_planner_default_document, family.encode("utf-8")
    )
    return json.loads(text)


def _evaluation_from(raw: _Evaluation) -> Evaluation:
    count = int(raw.path_component_count)
    names = [
        bytes(raw.path_names[index]).split(b"\0", 1)[0].decode("utf-8") for index in range(count)
    ]
    return Evaluation(
        objective=float(raw.objective),
        path_violation=float(raw.path_violation),
        path_components={
            name: float(raw.path_normalised[index]) for index, name in enumerate(names)
        },
        path_components_physical={
            name: float(raw.path_physical[index]) for index, name in enumerate(names)
        },
        terminal_residual=float(raw.terminal_residual),
        terminal_position_error=float(raw.terminal_position_error),
        terminal_velocity_error=float(raw.terminal_velocity_error),
        propellant_used=float(raw.propellant_used),
        final_mass=float(raw.final_mass),
    )


class PlannerTranscription:
    """One frozen transcription built by the native code from a canonical document."""

    def __init__(self, canonical_document: Mapping[str, Any]) -> None:
        self._library = load_planner_library()
        self._handle = ctypes.c_void_p()
        text = json.dumps(dict(canonical_document), allow_nan=False).encode("utf-8")
        _check(
            self._library,
            self._library.spacepdhcg_planner_create(text, ctypes.byref(self._handle)),
            "planner transcription create",
        )
        raw = _Dimensions()
        _check(
            self._library,
            self._library.spacepdhcg_planner_get_dimensions(self._handle, ctypes.byref(raw)),
            "planner dimensions",
        )
        self.dimensions = Dimensions(
            state_dimension=int(raw.state_dimension),
            control_dimension=int(raw.control_dimension),
            intervals=int(raw.intervals),
            terminal_dimension=int(raw.terminal_dimension),
            variables=int(raw.variables),
            scalar_rows=int(raw.scalar_rows),
            affine_rows=int(raw.affine_rows),
            virtual_variable_count=int(raw.virtual_variable_count),
            step_seconds=float(raw.step_seconds),
            initial_trust_radius=float(raw.initial_trust_radius),
        )
        self._raw = raw
        self.structure, self.state_variables, self.control_variables, self.virtual_variables = (
            self._load_structure()
        )
        self.description = json.loads(
            _string_result(self._library, self._library.spacepdhcg_planner_describe, self._handle)
        )

    def close(self) -> None:
        if self._handle:
            self._library.spacepdhcg_planner_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> PlannerTranscription:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # -- topology ---------------------------------------------------------

    def _load_structure(
        self,
    ) -> tuple[CQPStructure, NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        raw = self._raw
        variables = int(raw.variables)
        q_offsets = np.zeros(variables + 1, dtype=np.int32)
        q_indices = np.zeros(max(int(raw.quadratic_nonzeros), 1), dtype=np.int32)
        a_offsets = np.zeros(variables + 1, dtype=np.int32)
        a_indices = np.zeros(max(int(raw.scalar_nonzeros), 1), dtype=np.int32)
        f_offsets = np.zeros(variables + 1, dtype=np.int32)
        f_indices = np.zeros(max(int(raw.affine_nonzeros), 1), dtype=np.int32)
        affine_cones = (_Cone * max(int(raw.affine_cone_count), 1))()
        variable_cones = (_Cone * max(int(raw.variable_cone_count), 1))()
        state_variables = np.zeros(
            (int(raw.intervals) + 1) * int(raw.state_dimension), dtype=np.int32
        )
        control_variables = np.zeros(
            int(raw.intervals) * int(raw.control_dimension), dtype=np.int32
        )
        virtual_variables = np.zeros(max(int(raw.virtual_variable_count), 1), dtype=np.int32)
        _check(
            self._library,
            self._library.spacepdhcg_planner_structure(
                self._handle,
                _int_pointer(q_offsets),
                _int_pointer(q_indices),
                _int_pointer(a_offsets),
                _int_pointer(a_indices),
                _int_pointer(f_offsets),
                _int_pointer(f_indices),
                affine_cones,
                variable_cones,
                _int_pointer(state_variables),
                _int_pointer(control_variables),
                _int_pointer(virtual_variables),
            ),
            "planner structure",
        )
        quadratic = CSCStructure(
            shape=(variables, variables),
            indptr=q_offsets.astype(np.int64),
            indices=q_indices[: int(raw.quadratic_nonzeros)].astype(np.int64),
        )
        constraint = CSCStructure(
            shape=(int(raw.scalar_rows), variables),
            indptr=a_offsets.astype(np.int64),
            indices=a_indices[: int(raw.scalar_nonzeros)].astype(np.int64),
        )
        affine = None
        cones: tuple[ConeBlock, ...] = ()
        if int(raw.affine_rows) > 0:
            affine = CSCStructure(
                shape=(int(raw.affine_rows), variables),
                indptr=f_offsets.astype(np.int64),
                indices=f_indices[: int(raw.affine_nonzeros)].astype(np.int64),
            )
            cones = tuple(
                ConeBlock(
                    kind=_CONE_KINDS[int(cone.kind)],
                    start=int(cone.start),
                    vector_dimension=int(cone.vector_dimension),
                    power_alpha=float(cone.power_alpha),
                )
                for cone in affine_cones[: int(raw.affine_cone_count)]
            )
        variable_cone_blocks = tuple(
            ConeBlock(
                kind=_CONE_KINDS[int(cone.kind)],
                start=int(cone.start),
                vector_dimension=int(cone.vector_dimension),
                power_alpha=float(cone.power_alpha),
            )
            for cone in variable_cones[: int(raw.variable_cone_count)]
        )
        structure = CQPStructure(
            quadratic=quadratic,
            constraint=constraint,
            affine_cone=affine,
            affine_cones=cones,
            variable_cones=variable_cone_blocks,
        )
        return (
            structure,
            state_variables,
            control_variables,
            virtual_variables[: int(raw.virtual_variable_count)],
        )

    # -- coefficients ---------------------------------------------------

    def values(
        self, reference_states: FloatArray, reference_controls: FloatArray, trust_radius: float
    ) -> CQPValues:
        dims = self.dimensions
        states = np.ascontiguousarray(reference_states, dtype=np.float64).reshape(-1)
        controls = np.ascontiguousarray(reference_controls, dtype=np.float64).reshape(-1)
        if states.size != (dims.intervals + 1) * dims.state_dimension:
            raise ValueError("reference states have the wrong shape")
        if controls.size != dims.intervals * dims.control_dimension:
            raise ValueError("reference controls have the wrong shape")
        raw = self._raw
        quadratic = np.zeros(int(raw.quadratic_nonzeros), dtype=np.float64)
        scalar = np.zeros(int(raw.scalar_nonzeros), dtype=np.float64)
        affine = np.zeros(max(int(raw.affine_nonzeros), 1), dtype=np.float64)
        linear = np.zeros(dims.variables, dtype=np.float64)
        lower = np.zeros(dims.scalar_rows, dtype=np.float64)
        upper = np.zeros(dims.scalar_rows, dtype=np.float64)
        offset = np.zeros(max(dims.affine_rows, 1), dtype=np.float64)
        variable_lower = np.zeros(dims.variables, dtype=np.float64)
        variable_upper = np.zeros(dims.variables, dtype=np.float64)
        _check(
            self._library,
            self._library.spacepdhcg_planner_values(
                self._handle,
                _pointer(states),
                _pointer(controls),
                float(trust_radius),
                _pointer(quadratic),
                _pointer(scalar),
                _pointer(affine),
                _pointer(linear),
                _pointer(lower),
                _pointer(upper),
                _pointer(offset),
                _pointer(variable_lower),
                _pointer(variable_upper),
            ),
            "planner values",
        )
        return CQPValues(
            quadratic=quadratic,
            constraint=scalar,
            linear=linear,
            lower=lower,
            upper=upper,
            affine_cone=affine[: int(raw.affine_nonzeros)],
            affine_offset=offset[: dims.affine_rows],
            variable_lower=variable_lower,
            variable_upper=variable_upper,
        )

    # -- references, replays, evaluations --------------------------------

    def initial_reference(self) -> tuple[FloatArray, FloatArray]:
        dims = self.dimensions
        states = np.zeros((dims.intervals + 1, dims.state_dimension), dtype=np.float64)
        controls = np.zeros((dims.intervals, dims.control_dimension), dtype=np.float64)
        _check(
            self._library,
            self._library.spacepdhcg_planner_initial_reference(
                self._handle, _pointer(states), _pointer(controls)
            ),
            "planner initial reference",
        )
        return states, controls

    def rollout(
        self, initial_state: FloatArray, controls: FloatArray, substeps: int = 1
    ) -> FloatArray:
        dims = self.dimensions
        initial = np.ascontiguousarray(initial_state, dtype=np.float64).reshape(-1)
        control_matrix = np.ascontiguousarray(controls, dtype=np.float64).reshape(
            -1, dims.control_dimension
        )
        intervals = control_matrix.shape[0]
        if initial.size != dims.state_dimension:
            raise ValueError("initial state has the wrong dimension")
        if substeps < 1:
            raise ValueError("substeps must be positive")
        states = np.zeros((intervals * substeps + 1, dims.state_dimension), dtype=np.float64)
        _check(
            self._library,
            self._library.spacepdhcg_planner_rollout(
                self._handle,
                _pointer(initial),
                _pointer(np.ascontiguousarray(control_matrix.reshape(-1))),
                intervals,
                substeps,
                _pointer(states),
            ),
            "planner rollout",
        )
        return states

    def evaluate(self, states: FloatArray, controls: FloatArray) -> Evaluation:
        dims = self.dimensions
        state_matrix = np.ascontiguousarray(states, dtype=np.float64).reshape(-1)
        control_matrix = np.ascontiguousarray(controls, dtype=np.float64).reshape(-1)
        if state_matrix.size != (dims.intervals + 1) * dims.state_dimension:
            raise ValueError("states have the wrong shape for evaluation")
        if control_matrix.size != dims.intervals * dims.control_dimension:
            raise ValueError("controls have the wrong shape for evaluation")
        raw = _Evaluation()
        _check(
            self._library,
            self._library.spacepdhcg_planner_evaluate(
                self._handle, _pointer(state_matrix), _pointer(control_matrix), ctypes.byref(raw)
            ),
            "planner evaluate",
        )
        return _evaluation_from(raw)

    def path_components(self, states: FloatArray, controls: FloatArray) -> Evaluation:
        dims = self.dimensions
        control_matrix = np.ascontiguousarray(controls, dtype=np.float64).reshape(
            -1, dims.control_dimension
        )
        intervals = control_matrix.shape[0]
        state_matrix = np.ascontiguousarray(states, dtype=np.float64).reshape(-1)
        if state_matrix.size != (intervals + 1) * dims.state_dimension:
            raise ValueError("states must have intervals + 1 rows for path evaluation")
        raw = _Evaluation()
        _check(
            self._library,
            self._library.spacepdhcg_planner_path_components(
                self._handle,
                _pointer(state_matrix),
                _pointer(np.ascontiguousarray(control_matrix.reshape(-1))),
                intervals,
                ctypes.byref(raw),
            ),
            "planner path components",
        )
        return _evaluation_from(raw)

    def decode(self, primal: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Split a canonical primal into (states, controls, virtual controls)."""

        dims = self.dimensions
        vector = np.asarray(primal, dtype=np.float64).reshape(-1)
        if vector.size != dims.variables:
            raise ValueError("primal vector has the wrong length")
        states = vector[self.state_variables].reshape(dims.intervals + 1, dims.state_dimension)
        controls = vector[self.control_variables].reshape(dims.intervals, dims.control_dimension)
        virtual = (
            vector[self.virtual_variables].reshape(dims.intervals, dims.state_dimension)
            if self.virtual_variables.size
            else np.zeros((0, dims.state_dimension))
        )
        return states, controls, virtual
