"""ctypes bridge to the native free-final-time transcriptions (``pd3_fft`` / ``pd6_fft``).

The native C++ headers ``transcription/powered_descent_{3,6}dof_free_time.hpp`` build the
time-dilated SCvx subproblem (sigma column, variational sensitivities, quaternion tangent rule).
The C API exports the frozen CSC structure and the numeric values about a reference so a host
conic solver (Clarabel here) can drive the outer loop against the *exact native linearisation*.
This is the CPU reference path for the native free-final-time formulation; the CUDA kernel in
``device_scvx.cu`` (``spacepdhcg_cuda_time_dilated_variational_rk4_async``) produces the same
``A, B, S, z`` blocks on the device.
"""

from __future__ import annotations

import ctypes as ct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.cqp.problem import (
    CanonicalCQP,
    ConeBlock,
    ConeKind,
    CQPStructure,
    CQPValues,
    CSCStructure,
)
from spacepdhcg.native._library import NativeLibraryError, load_native_library

FloatArray = NDArray[np.float64]

_STATUS_OK = 0


class _Pd3ModelConfig(ct.Structure):
    _fields_ = [
        ("gravity", ct.c_double * 3),
        ("mass_flow_coefficient", ct.c_double),
        ("minimum_mass", ct.c_double),
        ("maximum_thrust", ct.c_double),
        ("minimum_sigma", ct.c_double),
        ("maximum_tilt_radians", ct.c_double),
        ("glide_slope_radians", ct.c_double),
    ]


class _Pd3FftConfig(ct.Structure):
    _fields_ = [
        ("model", _Pd3ModelConfig),
        ("intervals", ct.c_uint64),
        ("substeps", ct.c_uint64),
        ("sigma_minimum", ct.c_double),
        ("sigma_maximum", ct.c_double),
        ("trust_radius", ct.c_double),
        ("sigma_trust_radius", ct.c_double),
        ("virtual_l1_weight", ct.c_double),
        ("virtual_quadratic_weight", ct.c_double),
        ("virtual_epigraph_regularisation", ct.c_double),
        ("fuel_weight", ct.c_double),
        ("time_weight", ct.c_double),
        ("sigma_tracking_weight", ct.c_double),
        ("state_tracking_weights", ct.c_double * 7),
        ("control_tracking_weights", ct.c_double * 4),
        ("state_trust_scales", ct.c_double * 7),
        ("control_trust_scales", ct.c_double * 4),
    ]


class _Pd6FftConfig(ct.Structure):
    _fields_ = [
        ("gravity", ct.c_double * 3),
        ("principal_inertia", ct.c_double * 3),
        ("mass_flow_coefficient", ct.c_double),
        ("minimum_mass", ct.c_double),
        ("maximum_thrust", ct.c_double),
        ("minimum_sigma", ct.c_double),
        ("maximum_torque", ct.c_double),
        ("maximum_angular_rate", ct.c_double),
        ("maximum_tilt_radians", ct.c_double),
        ("glide_slope_radians", ct.c_double),
        ("intervals", ct.c_uint64),
        ("substeps", ct.c_uint64),
        ("sigma_minimum", ct.c_double),
        ("sigma_maximum", ct.c_double),
        ("trust_radius", ct.c_double),
        ("sigma_trust_radius", ct.c_double),
        ("virtual_l1_weight", ct.c_double),
        ("virtual_quadratic_weight", ct.c_double),
        ("virtual_epigraph_regularisation", ct.c_double),
        ("fuel_weight", ct.c_double),
        ("time_weight", ct.c_double),
        ("sigma_tracking_weight", ct.c_double),
        ("maximum_attitude_tilt_radians", ct.c_double),
        ("thrust_norm_mode", ct.c_int32),
        ("torque_mode", ct.c_int32),
        ("terminal_thrust_axial", ct.c_int32),
        ("reserved", ct.c_int32),
        ("thrust_arm", ct.c_double * 3),
        ("initial_fixed", ct.c_uint8 * 14),
        ("terminal_fixed", ct.c_uint8 * 14),
        ("state_tracking_weights", ct.c_double * 14),
        ("control_tracking_weights", ct.c_double * 7),
        ("state_trust_scales", ct.c_double * 14),
        ("control_trust_scales", ct.c_double * 7),
    ]


class _Sizes(ct.Structure):
    _fields_ = [
        (name, ct.c_uint64)
        for name in (
            "state_dimension",
            "control_dimension",
            "intervals",
            "variables",
            "scalar_rows",
            "affine_rows",
            "quadratic_nonzeros",
            "scalar_nonzeros",
            "affine_nonzeros",
            "cone_count",
            "control_offset",
            "sigma_index",
            "virtual_offset",
            "epigraph_offset",
            "dynamics_row_start",
            "topology_fingerprint",
        )
    ]


_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int32)


def _configure(library: ct.CDLL) -> ct.CDLL:
    if getattr(library, "_spacepdhcg_free_time_configured", False):
        return library
    library.spacepdhcg_default_pd3_fft_config.argtypes = [ct.POINTER(_Pd3FftConfig)]
    library.spacepdhcg_default_pd3_fft_config.restype = None
    library.spacepdhcg_default_pd6_fft_config.argtypes = [ct.POINTER(_Pd6FftConfig)]
    library.spacepdhcg_default_pd6_fft_config.restype = None
    library.spacepdhcg_pd3_fft_create.argtypes = [
        ct.POINTER(_Pd3FftConfig),
        ct.POINTER(ct.c_void_p),
    ]
    library.spacepdhcg_pd3_fft_create.restype = ct.c_int
    library.spacepdhcg_pd6_fft_create.argtypes = [
        ct.POINTER(_Pd6FftConfig),
        ct.POINTER(ct.c_void_p),
    ]
    library.spacepdhcg_pd6_fft_create.restype = ct.c_int
    library.spacepdhcg_free_time_destroy.argtypes = [ct.c_void_p]
    library.spacepdhcg_free_time_destroy.restype = None
    library.spacepdhcg_free_time_sizes_of.argtypes = [ct.c_void_p, ct.POINTER(_Sizes)]
    library.spacepdhcg_free_time_sizes_of.restype = ct.c_int
    library.spacepdhcg_free_time_structure.argtypes = [ct.c_void_p] + [_IP] * 8
    library.spacepdhcg_free_time_structure.restype = ct.c_int
    library.spacepdhcg_free_time_values.argtypes = [
        ct.c_void_p,
        _DP,
        _DP,
        ct.c_double,
        _DP,
        _DP,
        ct.c_double,
        ct.c_double,
    ] + [_DP] * 9
    library.spacepdhcg_free_time_values.restype = ct.c_int
    library.spacepdhcg_free_time_replay.argtypes = [ct.c_void_p, _DP, _DP, ct.c_double, _DP]
    library.spacepdhcg_free_time_replay.restype = ct.c_int
    library.spacepdhcg_free_time_project_control.argtypes = [ct.c_void_p, _DP, _DP]
    library.spacepdhcg_free_time_project_control.restype = ct.c_int
    library._spacepdhcg_free_time_configured = True
    return library


def _dp(array: FloatArray) -> _DP:
    return array.ctypes.data_as(_DP)


def _ip(array: NDArray[np.int32]) -> _IP:
    return array.ctypes.data_as(_IP)


def _check(library: ct.CDLL, status: int, what: str) -> None:
    if status != _STATUS_OK:
        message = library.spacepdhcg_last_error()
        text = message.decode("utf-8") if message else "unknown"
        raise NativeLibraryError(f"{what} failed (status {status}): {text}")


@dataclass(slots=True)
class Pd6FreeTimeOptions:
    """Mirror of ``spacepdhcg_pd6_fft_config`` with Python defaults from the native header."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -3.711)
    principal_inertia: tuple[float, float, float] = (2500.0, 2200.0, 1800.0)
    mass_flow_coefficient: float = 4.6e-4
    minimum_mass: float = 1000.0
    maximum_thrust: float = 15000.0
    minimum_sigma: float = 0.0
    maximum_torque: float = 2000.0
    maximum_angular_rate: float = 1.0
    maximum_tilt_radians: float = 0.5235987755982988
    glide_slope_radians: float = 1.0471975511965976
    intervals: int = 20
    substeps: int = 1
    sigma_minimum: float = 1.0e-2
    sigma_maximum: float = 1.0e3
    trust_radius: float = 1.0
    sigma_trust_radius: float = 1.0
    virtual_l1_weight: float = 1.0e3
    virtual_quadratic_weight: float = 1.0e-8
    virtual_epigraph_regularisation: float = 1.0e-10
    fuel_weight: float = 0.0
    time_weight: float = 1.0
    sigma_tracking_weight: float = 1.0e-6
    #: Body-thrust-axis tilt bound from the vertical (Szmuk theta_max); pi = disabled.
    maximum_attitude_tilt_radians: float = 3.141592653589793
    thrust_norm_mode: Literal["epigraph", "linearised"] = "epigraph"
    torque_mode: Literal["direct", "thrust_arm"] = "direct"
    terminal_thrust_axial: bool = False
    thrust_arm: tuple[float, float, float] = (0.0, 0.0, -1.0e-2)
    initial_fixed: tuple[bool, ...] = (True,) * 14
    terminal_fixed: tuple[bool, ...] = (True,) * 13 + (False,)
    state_tracking_weights: tuple[float, ...] = (1.0e-6,) * 14
    control_tracking_weights: tuple[float, ...] = (1.0e-8,) * 7
    state_trust_scales: tuple[float, ...] = (1.0,) * 14
    control_trust_scales: tuple[float, ...] = (1.0,) * 7

    def to_ctypes(self) -> _Pd6FftConfig:
        config = _Pd6FftConfig()
        for name in (
            "mass_flow_coefficient",
            "minimum_mass",
            "maximum_thrust",
            "minimum_sigma",
            "maximum_torque",
            "maximum_angular_rate",
            "maximum_tilt_radians",
            "glide_slope_radians",
            "sigma_minimum",
            "sigma_maximum",
            "trust_radius",
            "sigma_trust_radius",
            "virtual_l1_weight",
            "virtual_quadratic_weight",
            "virtual_epigraph_regularisation",
            "fuel_weight",
            "time_weight",
            "sigma_tracking_weight",
            "maximum_attitude_tilt_radians",
        ):
            setattr(config, name, float(getattr(self, name)))
        config.intervals = int(self.intervals)
        config.substeps = int(self.substeps)
        config.gravity[:] = [float(v) for v in self.gravity]
        config.principal_inertia[:] = [float(v) for v in self.principal_inertia]
        config.thrust_arm[:] = [float(v) for v in self.thrust_arm]
        config.thrust_norm_mode = 0 if self.thrust_norm_mode == "epigraph" else 1
        config.torque_mode = 0 if self.torque_mode == "direct" else 1
        config.terminal_thrust_axial = 1 if self.terminal_thrust_axial else 0
        config.reserved = 0
        config.initial_fixed[:] = [1 if v else 0 for v in self.initial_fixed]
        config.terminal_fixed[:] = [1 if v else 0 for v in self.terminal_fixed]
        config.state_tracking_weights[:] = [float(v) for v in self.state_tracking_weights]
        config.control_tracking_weights[:] = [float(v) for v in self.control_tracking_weights]
        config.state_trust_scales[:] = [float(v) for v in self.state_trust_scales]
        config.control_trust_scales[:] = [float(v) for v in self.control_trust_scales]
        return config


@dataclass(slots=True)
class Pd3FreeTimeOptions:
    """Mirror of ``spacepdhcg_pd3_fft_config``."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -3.711)
    mass_flow_coefficient: float = 4.6e-4
    minimum_mass: float = 1000.0
    maximum_thrust: float = 15000.0
    minimum_sigma: float = 0.0
    maximum_tilt_radians: float = 0.5235987755982988
    glide_slope_radians: float = 1.0471975511965976
    intervals: int = 20
    substeps: int = 1
    sigma_minimum: float = 1.0
    sigma_maximum: float = 1.0e4
    trust_radius: float = 1.0
    sigma_trust_radius: float = 5.0
    virtual_l1_weight: float = 1.0e3
    virtual_quadratic_weight: float = 1.0e-8
    virtual_epigraph_regularisation: float = 1.0e-10
    fuel_weight: float = 1.0e-3
    time_weight: float = 0.0
    sigma_tracking_weight: float = 1.0e-6
    state_tracking_weights: tuple[float, ...] = (1.0e-6,) * 7
    control_tracking_weights: tuple[float, ...] = (1.0e-8,) * 4
    state_trust_scales: tuple[float, ...] = (1.0e-2,) * 3 + (1.0e-1,) * 3 + (1.0e-3,)
    control_trust_scales: tuple[float, ...] = (1.0e-4,) * 4

    def to_ctypes(self) -> _Pd3FftConfig:
        config = _Pd3FftConfig()
        config.model.gravity[:] = [float(v) for v in self.gravity]
        for name in (
            "mass_flow_coefficient",
            "minimum_mass",
            "maximum_thrust",
            "minimum_sigma",
            "maximum_tilt_radians",
            "glide_slope_radians",
        ):
            setattr(config.model, name, float(getattr(self, name)))
        for name in (
            "sigma_minimum",
            "sigma_maximum",
            "trust_radius",
            "sigma_trust_radius",
            "virtual_l1_weight",
            "virtual_quadratic_weight",
            "virtual_epigraph_regularisation",
            "fuel_weight",
            "time_weight",
            "sigma_tracking_weight",
        ):
            setattr(config, name, float(getattr(self, name)))
        config.intervals = int(self.intervals)
        config.substeps = int(self.substeps)
        config.state_tracking_weights[:] = [float(v) for v in self.state_tracking_weights]
        config.control_tracking_weights[:] = [float(v) for v in self.control_tracking_weights]
        config.state_trust_scales[:] = [float(v) for v in self.state_trust_scales]
        config.control_trust_scales[:] = [float(v) for v in self.control_trust_scales]
        return config


@dataclass(slots=True)
class FreeTimeLayout:
    state_dimension: int
    control_dimension: int
    intervals: int
    variables: int
    control_offset: int
    sigma_index: int
    virtual_offset: int
    epigraph_offset: int
    topology_fingerprint: int


class NativeFreeTimeTranscription:
    """Owns one native ``pd3_fft`` or ``pd6_fft`` subproblem and exposes its CQP."""

    def __init__(self, options: Pd3FreeTimeOptions | Pd6FreeTimeOptions) -> None:
        self._library = _configure(load_native_library())
        self._handle = ct.c_void_p()
        config = options.to_ctypes()
        self.family = "pd3_fft" if isinstance(options, Pd3FreeTimeOptions) else "pd6_fft"
        creator = (
            self._library.spacepdhcg_pd3_fft_create
            if self.family == "pd3_fft"
            else self._library.spacepdhcg_pd6_fft_create
        )
        _check(self._library, creator(ct.byref(config), ct.byref(self._handle)), self.family)
        self.options = options
        sizes = _Sizes()
        _check(
            self._library,
            self._library.spacepdhcg_free_time_sizes_of(self._handle, ct.byref(sizes)),
            "sizes",
        )
        self._sizes = sizes
        self.layout = FreeTimeLayout(
            state_dimension=int(sizes.state_dimension),
            control_dimension=int(sizes.control_dimension),
            intervals=int(sizes.intervals),
            variables=int(sizes.variables),
            control_offset=int(sizes.control_offset),
            sigma_index=int(sizes.sigma_index),
            virtual_offset=int(sizes.virtual_offset),
            epigraph_offset=int(sizes.epigraph_offset),
            topology_fingerprint=int(sizes.topology_fingerprint),
        )
        self.structure = self._read_structure()

    def close(self) -> None:
        if self._handle:
            self._library.spacepdhcg_free_time_destroy(self._handle)
            self._handle = ct.c_void_p()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def _read_structure(self) -> CQPStructure:
        s = self._sizes
        n = int(s.variables)
        q_off = np.zeros(n + 1, dtype=np.int32)
        q_idx = np.zeros(int(s.quadratic_nonzeros), dtype=np.int32)
        a_off = np.zeros(n + 1, dtype=np.int32)
        a_idx = np.zeros(int(s.scalar_nonzeros), dtype=np.int32)
        f_off = np.zeros(n + 1, dtype=np.int32)
        f_idx = np.zeros(int(s.affine_nonzeros), dtype=np.int32)
        cone_starts = np.zeros(int(s.cone_count), dtype=np.int32)
        cone_dims = np.zeros(int(s.cone_count), dtype=np.int32)
        _check(
            self._library,
            self._library.spacepdhcg_free_time_structure(
                self._handle,
                _ip(q_off),
                _ip(q_idx),
                _ip(a_off),
                _ip(a_idx),
                _ip(f_off),
                _ip(f_idx),
                _ip(cone_starts),
                _ip(cone_dims),
            ),
            "structure",
        )
        cones = tuple(
            ConeBlock(ConeKind.SECOND_ORDER, int(start), int(dim))
            for start, dim in zip(cone_starts, cone_dims, strict=True)
        )
        return CQPStructure(
            quadratic=CSCStructure((n, n), q_off, q_idx),
            constraint=CSCStructure((int(s.scalar_rows), n), a_off, a_idx),
            affine_cone=CSCStructure((int(s.affine_rows), n), f_off, f_idx),
            affine_cones=cones,
        )

    def _flat(self, states: FloatArray, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        lay = self.layout
        states = np.ascontiguousarray(states, dtype=np.float64)
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        if states.shape != (lay.intervals + 1, lay.state_dimension):
            raise ValueError(f"states must have shape {(lay.intervals + 1, lay.state_dimension)}")
        if controls.shape != (lay.intervals, lay.control_dimension):
            raise ValueError(f"controls must have shape {(lay.intervals, lay.control_dimension)}")
        return states.reshape(-1), controls.reshape(-1)

    def values(
        self,
        states: FloatArray,
        controls: FloatArray,
        sigma: float,
        initial: FloatArray,
        target: FloatArray,
        *,
        trust_radius: float = -1.0,
        sigma_trust_radius: float = -1.0,
    ) -> CQPValues:
        flat_states, flat_controls = self._flat(states, controls)
        initial = np.ascontiguousarray(initial, dtype=np.float64)
        target = np.ascontiguousarray(target, dtype=np.float64)
        expected_target = self.layout.state_dimension if self.family == "pd6_fft" else 6
        if initial.shape != (self.layout.state_dimension,) or target.shape != (expected_target,):
            raise ValueError("initial/target boundary vectors have the wrong length")
        s = self._sizes
        out = {
            "quadratic": np.zeros(int(s.quadratic_nonzeros)),
            "constraint": np.zeros(int(s.scalar_nonzeros)),
            "linear": np.zeros(int(s.variables)),
            "lower": np.zeros(int(s.scalar_rows)),
            "upper": np.zeros(int(s.scalar_rows)),
            "affine_cone": np.zeros(int(s.affine_nonzeros)),
            "affine_offset": np.zeros(int(s.affine_rows)),
            "variable_lower": np.zeros(int(s.variables)),
            "variable_upper": np.zeros(int(s.variables)),
        }
        _check(
            self._library,
            self._library.spacepdhcg_free_time_values(
                self._handle,
                _dp(flat_states),
                _dp(flat_controls),
                float(sigma),
                _dp(initial),
                _dp(target),
                float(trust_radius),
                float(sigma_trust_radius),
                *[_dp(out[name]) for name in out],
            ),
            "values",
        )
        return CQPValues(**out).validated(self.structure)

    def replay(self, states: FloatArray, controls: FloatArray, sigma: float) -> FloatArray:
        """Nonlinear time-dilated propagation of every interval (shape ``(K, n_x)``)."""

        flat_states, flat_controls = self._flat(states, controls)
        lay = self.layout
        out = np.zeros(lay.intervals * lay.state_dimension)
        _check(
            self._library,
            self._library.spacepdhcg_free_time_replay(
                self._handle, _dp(flat_states), _dp(flat_controls), float(sigma), _dp(out)
            ),
            "replay",
        )
        return out.reshape(lay.intervals, lay.state_dimension)

    def project_control(self, control: FloatArray) -> FloatArray:
        control = np.ascontiguousarray(control, dtype=np.float64)
        out = np.zeros_like(control)
        _check(
            self._library,
            self._library.spacepdhcg_free_time_project_control(
                self._handle, _dp(control), _dp(out)
            ),
            "project_control",
        )
        return out

    def decode(self, primal: FloatArray) -> tuple[FloatArray, FloatArray, float, FloatArray]:
        lay = self.layout
        primal = np.asarray(primal, dtype=np.float64)
        states = primal[: lay.control_offset].reshape(lay.intervals + 1, lay.state_dimension)
        controls = primal[lay.control_offset : lay.sigma_index].reshape(
            lay.intervals, lay.control_dimension
        )
        sigma = float(primal[lay.sigma_index])
        virtual = primal[lay.virtual_offset : lay.epigraph_offset].reshape(
            lay.intervals, lay.state_dimension
        )
        return states.copy(), controls.copy(), sigma, virtual.copy()


@dataclass(slots=True)
class FreeTimeIteration:
    iteration: int
    sigma: float
    objective: float
    virtual_inf: float
    replay_defect_inf: float
    trust_radius: float
    sigma_trust_radius: float
    predicted_reduction: float
    actual_reduction: float
    agreement: float
    accepted: bool
    solver_status: str


@dataclass(slots=True)
class FreeTimeOutcome:
    states: FloatArray
    controls: FloatArray
    sigma: float
    converged: bool
    iterations: list[FreeTimeIteration] = field(default_factory=list)
    replay_defect_inf: float = float("nan")
    termination: str = ""
    topology_fingerprint: int = 0


@dataclass(slots=True)
class FreeTimeLoopSettings:
    max_iterations: int = 60
    defect_tolerance: float = 1.0e-6
    sigma_tolerance: float = 1.0e-4
    virtual_tolerance: float = 1.0e-7
    defect_penalty: float = 1.0e3
    trust_radius: float = 1.0
    sigma_trust_radius: float = 0.5
    minimum_trust_radius: float = 1.0e-4
    shrink: float = 0.5
    grow: float = 2.0
    accept_ratio: float = 0.1
    grow_ratio: float = 0.7
    clarabel_tolerance: float = 1.0e-8
    clarabel_iterations: int = 400
    project_controls: bool = True
    # Szmuk 2018 Algorithm 1 semantics: accept every solved step (soft trust through the
    # quadratic tracking weights in the native objective), no ratio test, fixed radii.
    accept_every_step: bool = False


def run_free_time_scvx(
    transcription: NativeFreeTimeTranscription,
    states: FloatArray,
    controls: FloatArray,
    sigma: float,
    initial: FloatArray,
    target: FloatArray,
    settings: FreeTimeLoopSettings | None = None,
    backend_builder: Callable[[CanonicalCQP], Any] | None = None,
) -> FreeTimeOutcome:
    """Hard-trust-region SCvx outer loop over the native free-final-time CQP.

    Merit = (native linear objective at the candidate, i.e. time/fuel with the bilinear fuel
    term linearised) + defect_penalty * sum |nonlinear replay defect|.  The predicted reduction
    uses the linearised dynamics (virtual control), the actual reduction the nonlinear replay;
    standard ratio test with shrink/grow of the state/control trust radius and sigma box.

    ``backend_builder`` receives the first ``CanonicalCQP`` and must return a persistent
    workspace with ``update(values)`` and ``solve()`` (Clarabel by default; the pure-QOCO GPU
    backend for the deferred GPU legs).
    """

    from spacepdhcg.backends.clarabel_backend import PersistentClarabel

    settings = settings or FreeTimeLoopSettings()
    if backend_builder is None:

        def backend_builder(problem: CanonicalCQP) -> Any:
            return PersistentClarabel(
                problem,
                tolerance=settings.clarabel_tolerance,
                iteration_limit=settings.clarabel_iterations,
            )

    lay = transcription.layout
    opt = transcription.options
    states = np.array(states, dtype=np.float64)
    controls = np.array(controls, dtype=np.float64)
    if settings.project_controls:
        controls = np.array([transcription.project_control(u) for u in controls])
    sigma = float(sigma)
    radius = settings.trust_radius
    sigma_radius = settings.sigma_trust_radius

    def true_objective(st: FloatArray, ct_: FloatArray, sg: float) -> float:
        d_tau = 1.0 / lay.intervals
        fuel = float(opt.fuel_weight) * sg * d_tau * float(np.sum(ct_[:, -1]))
        return float(opt.time_weight) * sg + fuel

    def defects(st: FloatArray, ct_: FloatArray, sg: float) -> FloatArray:
        return transcription.replay(st, ct_, sg) - st[1:]

    def defect_penalty(d: FloatArray) -> float:
        # Exact-penalty term of the merit.  Once an iterate is dynamically feasible to the
        # declared tolerance the residual roundoff-level defects (interior-point accuracy) are
        # not charged, otherwise they bias the agreement ratio and stall the trust region.
        if d.size == 0 or float(np.max(np.abs(d))) <= settings.defect_tolerance:
            return 0.0
        return settings.defect_penalty * float(np.sum(np.abs(d)))

    def merit(st: FloatArray, ct_: FloatArray, sg: float) -> float:
        return true_objective(st, ct_, sg) + defect_penalty(defects(st, ct_, sg))

    current_merit = merit(states, controls, sigma)
    backend: Any = None
    records: list[FreeTimeIteration] = []
    converged = False
    termination = "maximum_iterations"
    for iteration in range(settings.max_iterations):
        values = transcription.values(
            states,
            controls,
            sigma,
            initial,
            target,
            trust_radius=radius,
            sigma_trust_radius=sigma_radius,
        )
        if backend is None:
            backend = backend_builder(CanonicalCQP(transcription.structure, values))
        else:
            backend.update(values)
        solution = backend.solve()
        cand_states, cand_controls, cand_sigma, virtual = transcription.decode(solution.primal)
        if settings.project_controls:
            cand_controls = np.array([transcription.project_control(u) for u in cand_controls])
        virtual_inf = float(np.max(np.abs(virtual))) if virtual.size else 0.0
        usable = solution.solved or solution.status.lower().startswith("almost")
        if not usable or not np.all(np.isfinite(cand_states)):
            radius *= settings.shrink
            sigma_radius *= settings.shrink
            records.append(
                FreeTimeIteration(
                    iteration,
                    sigma,
                    true_objective(states, controls, sigma),
                    virtual_inf,
                    float("nan"),
                    radius,
                    sigma_radius,
                    0.0,
                    0.0,
                    float("nan"),
                    False,
                    solution.status,
                )
            )
            if radius < settings.minimum_trust_radius:
                termination = "trust_region_collapsed"
                break
            continue
        # Model merit: linear objective + penalty on the virtual control (the linearised defect).
        model_merit = true_objective(cand_states, cand_controls, cand_sigma) + defect_penalty(
            virtual
        )
        cand_defects = defects(cand_states, cand_controls, cand_sigma)
        cand_merit = true_objective(cand_states, cand_controls, cand_sigma) + defect_penalty(
            cand_defects
        )
        predicted = current_merit - model_merit
        actual = current_merit - cand_merit
        agreement = actual / predicted if predicted > 1.0e-14 else (1.0 if actual >= 0 else -1.0)
        defect_inf = float(np.max(np.abs(cand_defects)))
        accepted = (
            True
            if settings.accept_every_step
            else (
                agreement >= settings.accept_ratio
                or (abs(predicted) <= 1.0e-12 and actual >= -1.0e-12)
            )
        )
        records.append(
            FreeTimeIteration(
                iteration,
                cand_sigma,
                true_objective(cand_states, cand_controls, cand_sigma),
                virtual_inf,
                defect_inf,
                radius,
                sigma_radius,
                predicted,
                actual,
                agreement,
                accepted,
                solution.status,
            )
        )
        if accepted:
            sigma_change = abs(cand_sigma - sigma)
            states, controls, sigma = cand_states, cand_controls, cand_sigma
            current_merit = cand_merit
            if not settings.accept_every_step and agreement >= settings.grow_ratio:
                radius *= settings.grow
                sigma_radius *= settings.grow
            if (
                defect_inf < settings.defect_tolerance
                and virtual_inf < settings.virtual_tolerance
                and sigma_change < settings.sigma_tolerance
                and iteration > 0
            ):
                converged = True
                termination = "converged"
                break
        else:
            radius *= settings.shrink
            sigma_radius *= settings.shrink
            if radius < settings.minimum_trust_radius:
                termination = "trust_region_collapsed"
                break
    final_defect = float(np.max(np.abs(defects(states, controls, sigma))))
    return FreeTimeOutcome(
        states=states,
        controls=controls,
        sigma=sigma,
        converged=converged,
        iterations=records,
        replay_defect_inf=final_defect,
        termination=termination,
        topology_fingerprint=lay.topology_fingerprint,
    )
