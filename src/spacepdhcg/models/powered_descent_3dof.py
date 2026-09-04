"""Nonlinear three-degree-of-freedom powered-descent dynamics.

State ordering is ``[r_x, r_y, r_z, v_x, v_y, v_z, mass]``. Control ordering is
``[T_x, T_y, T_z, sigma]``, where ``sigma`` is the thrust-magnitude epigraph used by
the convex subproblem:

``||T||_2 <= sigma`` and ``mass_dot = -alpha * sigma``.

The model deliberately keeps gravity constant in this first reference implementation. The
solver and transcription APIs do not depend on that simplification; higher-fidelity gravity and
aerodynamics can replace the model after the persistent CT-SCvx loop is established.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STATE_DIMENSION = 7
CONTROL_DIMENSION = 4
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PoweredDescent3DOFConfig:
    """Physical and convex path-constraint parameters."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -3.711)
    mass_flow_coefficient: float = 4.6e-4
    minimum_mass: float = 1_000.0
    maximum_thrust: float = 15_000.0
    minimum_sigma: float = 0.0
    maximum_tilt_radians: float = np.deg2rad(30.0)
    glide_slope_radians: float = np.deg2rad(60.0)

    def __post_init__(self) -> None:
        gravity = np.asarray(self.gravity, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must contain three finite components")
        if not np.isfinite(self.mass_flow_coefficient) or self.mass_flow_coefficient <= 0:
            raise ValueError("mass_flow_coefficient must be finite and positive")
        if not np.isfinite(self.minimum_mass) or self.minimum_mass <= 0:
            raise ValueError("minimum_mass must be finite and positive")
        if not np.isfinite(self.maximum_thrust) or self.maximum_thrust <= 0:
            raise ValueError("maximum_thrust must be finite and positive")
        if not np.isfinite(self.minimum_sigma) or self.minimum_sigma < 0:
            raise ValueError("minimum_sigma must be finite and non-negative")
        if self.minimum_sigma > self.maximum_thrust:
            raise ValueError("minimum_sigma may not exceed maximum_thrust")
        if not 0.0 < self.maximum_tilt_radians < 0.5 * np.pi:
            raise ValueError("maximum_tilt_radians must lie between zero and pi/2")
        if not 0.0 < self.glide_slope_radians < 0.5 * np.pi:
            raise ValueError("glide_slope_radians must lie between zero and pi/2")

    @property
    def gravity_vector(self) -> FloatArray:
        return np.asarray(self.gravity, dtype=np.float64)

    @property
    def tilt_cosine(self) -> float:
        return float(np.cos(self.maximum_tilt_radians))

    @property
    def glide_slope_tangent(self) -> float:
        return float(np.tan(self.glide_slope_radians))


@dataclass(frozen=True, slots=True)
class PoweredDescentPathDiagnostics:
    """Independent nonlinear path-constraint violations."""

    thrust_epigraph: float
    throttle_lower: float
    throttle_upper: float
    tilt: float
    minimum_mass: float
    altitude: float
    glide_slope: float

    @property
    def maximum_violation(self) -> float:
        return max(
            self.thrust_epigraph,
            self.throttle_lower,
            self.throttle_upper,
            self.tilt,
            self.minimum_mass,
            self.altitude,
            self.glide_slope,
        )

    def feasible(self, tolerance: float = 1.0e-8) -> bool:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        return self.maximum_violation <= tolerance


class PoweredDescent3DOFModel:
    """Nonlinear powered-descent dynamics with analytic first derivatives."""

    state_dimension = STATE_DIMENSION
    control_dimension = CONTROL_DIMENSION

    def __init__(self, config: PoweredDescent3DOFConfig | None = None) -> None:
        self.config = config or PoweredDescent3DOFConfig()

    def dynamics(self, state: FloatArray, control: FloatArray) -> FloatArray:
        """Evaluate continuous dynamics ``x_dot = f(x, u)``."""

        state_vector = self._state(state)
        control_vector = self._control(control)
        mass = state_vector[6]
        if mass <= 0:
            raise ValueError("mass must be positive")

        derivative = np.empty(STATE_DIMENSION, dtype=np.float64)
        derivative[:3] = state_vector[3:6]
        derivative[3:6] = control_vector[:3] / mass + self.config.gravity_vector
        derivative[6] = -self.config.mass_flow_coefficient * control_vector[3]
        return derivative

    def jacobians(self, state: FloatArray, control: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return analytic continuous Jacobians ``df/dx`` and ``df/du``."""

        state_vector = self._state(state)
        control_vector = self._control(control)
        mass = state_vector[6]
        if mass <= 0:
            raise ValueError("mass must be positive")

        state_jacobian = np.zeros((STATE_DIMENSION, STATE_DIMENSION), dtype=np.float64)
        control_jacobian = np.zeros((STATE_DIMENSION, CONTROL_DIMENSION), dtype=np.float64)
        state_jacobian[:3, 3:6] = np.eye(3)
        state_jacobian[3:6, 6] = -control_vector[:3] / mass**2
        control_jacobian[3:6, :3] = np.eye(3) / mass
        control_jacobian[6, 3] = -self.config.mass_flow_coefficient
        return state_jacobian, control_jacobian

    def affine_linearisation(
        self,
        state: FloatArray,
        control: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``A, B, c`` such that ``f(x,u) ~= A x + B u + c``."""

        state_vector = self._state(state)
        control_vector = self._control(control)
        state_jacobian, control_jacobian = self.jacobians(state_vector, control_vector)
        offset = (
            self.dynamics(state_vector, control_vector)
            - state_jacobian @ state_vector
            - control_jacobian @ control_vector
        )
        return state_jacobian, control_jacobian, offset

    def linearised_euler_dynamics(
        self,
        state: FloatArray,
        control: FloatArray,
        step_seconds: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return affine forward-Euler matrices ``x+ = A_d x + B_d u + d``."""

        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")
        state_jacobian, control_jacobian, offset = self.affine_linearisation(state, control)
        discrete_state = np.eye(STATE_DIMENSION) + step_seconds * state_jacobian
        discrete_control = step_seconds * control_jacobian
        discrete_offset = step_seconds * offset
        return discrete_state, discrete_control, discrete_offset

    def euler_step(
        self,
        state: FloatArray,
        control: FloatArray,
        step_seconds: float,
    ) -> FloatArray:
        """Advance one reference-path step with explicit Euler integration."""

        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")
        state_vector = self._state(state)
        return state_vector + step_seconds * self.dynamics(state_vector, control)

    def rk4_step(
        self,
        state: FloatArray,
        control: FloatArray,
        step_seconds: float,
        *,
        substeps: int = 1,
    ) -> FloatArray:
        """Advance one zero-order-hold interval with classical RK4 (``substeps`` stages)."""

        _check_step(step_seconds, substeps)
        x = self._state(state)
        u = self._control(control)
        h = step_seconds / substeps
        for _ in range(substeps):
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + 0.5 * h * k1, u)
            k3 = self.dynamics(x + 0.5 * h * k2, u)
            k4 = self.dynamics(x + h * k3, u)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def linearised_rk4_dynamics(
        self,
        state: FloatArray,
        control: FloatArray,
        step_seconds: float,
        *,
        substeps: int = 1,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the exact affine model ``x+ = A_d x + B_d u + d`` of :meth:`rk4_step`.

        The state-transition and control-sensitivity matrices are propagated through the same
        RK4 stages as the state (variational RK4), so ``A_d`` and ``B_d`` are the exact Jacobians
        of the implemented discrete map and the affine model reproduces it at the reference to
        roundoff.  This is the Python twin of the native ``rk4_variational`` discretisation.
        """

        _check_step(step_seconds, substeps)
        x = self._state(state)
        u = self._control(control)
        phi = np.eye(STATE_DIMENSION)
        psi = np.zeros((STATE_DIMENSION, CONTROL_DIMENSION))
        h = step_seconds / substeps
        for _ in range(substeps):
            k1, kphi1, kpsi1 = self._variational_stage(x, u, phi, psi)
            k2, kphi2, kpsi2 = self._variational_stage(
                x + 0.5 * h * k1, u, phi + 0.5 * h * kphi1, psi + 0.5 * h * kpsi1
            )
            k3, kphi3, kpsi3 = self._variational_stage(
                x + 0.5 * h * k2, u, phi + 0.5 * h * kphi2, psi + 0.5 * h * kpsi2
            )
            k4, kphi4, kpsi4 = self._variational_stage(
                x + h * k3, u, phi + h * kphi3, psi + h * kpsi3
            )
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            phi = phi + (h / 6.0) * (kphi1 + 2.0 * kphi2 + 2.0 * kphi3 + kphi4)
            psi = psi + (h / 6.0) * (kpsi1 + 2.0 * kpsi2 + 2.0 * kpsi3 + kpsi4)
        offset = x - phi @ self._state(state) - psi @ u
        return phi, psi, offset

    def _variational_stage(
        self,
        x: FloatArray,
        u: FloatArray,
        phi: FloatArray,
        psi: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        a, b = self.jacobians(x, u)
        return self.dynamics(x, u), a @ phi, a @ psi + b

    def discrete_step(
        self,
        state: FloatArray,
        control: FloatArray,
        step_seconds: float,
        *,
        method: str = "forward_euler",
        substeps: int = 1,
    ) -> FloatArray:
        """Advance one interval with the named discretisation."""

        if method == "forward_euler":
            return self.euler_step(state, control, step_seconds)
        if method == "rk4":
            return self.rk4_step(state, control, step_seconds, substeps=substeps)
        raise ValueError(f"unknown discretisation {method!r}")

    def linearised_discrete_dynamics(
        self,
        state: FloatArray,
        control: FloatArray,
        step_seconds: float,
        *,
        method: str = "forward_euler",
        substeps: int = 1,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the affine one-step model matching :meth:`discrete_step`."""

        if method == "forward_euler":
            return self.linearised_euler_dynamics(state, control, step_seconds)
        if method == "rk4":
            return self.linearised_rk4_dynamics(state, control, step_seconds, substeps=substeps)
        raise ValueError(f"unknown discretisation {method!r}")

    def rollout(
        self,
        initial_state: FloatArray,
        controls: FloatArray,
        step_seconds: float,
        *,
        method: str = "forward_euler",
        substeps: int = 1,
    ) -> FloatArray:
        """Roll out zero-order-held controls with the reference integrator.

        The default forward-Euler integrator is the frozen reference transcription; ``method="rk4"``
        selects the variational RK4 map used by the accurate literature option.
        """

        initial = self._state(initial_state)
        control_array = np.asarray(controls, dtype=np.float64)
        if control_array.ndim != 2 or control_array.shape[1] != CONTROL_DIMENSION:
            raise ValueError(f"controls must have shape (N, {CONTROL_DIMENSION})")
        if not np.all(np.isfinite(control_array)):
            raise ValueError("controls must be finite")
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")

        states = np.empty((control_array.shape[0] + 1, STATE_DIMENSION), dtype=np.float64)
        states[0] = initial
        for interval, control in enumerate(control_array):
            states[interval + 1] = self.discrete_step(
                states[interval],
                control,
                step_seconds,
                method=method,
                substeps=substeps,
            )
            if states[interval + 1, 6] <= 0:
                raise ValueError("rollout produced non-positive mass")
        return states

    def path_diagnostics(
        self,
        states: FloatArray,
        controls: FloatArray,
    ) -> PoweredDescentPathDiagnostics:
        """Evaluate thrust, tilt, mass, altitude and glide-slope violations."""

        state_array = np.asarray(states, dtype=np.float64)
        control_array = np.asarray(controls, dtype=np.float64)
        if state_array.ndim != 2 or state_array.shape[1] != STATE_DIMENSION:
            raise ValueError(f"states must have shape (N+1, {STATE_DIMENSION})")
        if control_array.ndim != 2 or control_array.shape[1] != CONTROL_DIMENSION:
            raise ValueError(f"controls must have shape (N, {CONTROL_DIMENSION})")
        if state_array.shape[0] != control_array.shape[0] + 1:
            raise ValueError("states must contain exactly one more node than controls")
        if not np.all(np.isfinite(state_array)) or not np.all(np.isfinite(control_array)):
            raise ValueError("states and controls must be finite")

        thrust = control_array[:, :3]
        sigma = control_array[:, 3]
        thrust_norm = np.linalg.norm(thrust, axis=1)
        horizontal_position = np.linalg.norm(state_array[:, :2], axis=1)
        altitude = state_array[:, 2]
        return PoweredDescentPathDiagnostics(
            thrust_epigraph=float(np.max(np.maximum(thrust_norm - sigma, 0.0))),
            throttle_lower=float(np.max(np.maximum(self.config.minimum_sigma - sigma, 0.0))),
            throttle_upper=float(np.max(np.maximum(sigma - self.config.maximum_thrust, 0.0))),
            tilt=float(
                np.max(
                    np.maximum(
                        self.config.tilt_cosine * sigma - thrust[:, 2],
                        0.0,
                    )
                )
            ),
            minimum_mass=float(
                np.max(np.maximum(self.config.minimum_mass - state_array[:, 6], 0.0))
            ),
            altitude=float(np.max(np.maximum(-altitude, 0.0))),
            glide_slope=float(
                np.max(
                    np.maximum(
                        horizontal_position - self.config.glide_slope_tangent * altitude,
                        0.0,
                    )
                )
            ),
        )

    @staticmethod
    def _state(state: FloatArray) -> FloatArray:
        vector = np.asarray(state, dtype=np.float64)
        if vector.shape != (STATE_DIMENSION,):
            raise ValueError(f"state must have shape ({STATE_DIMENSION},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("state must be finite")
        return vector.copy()

    @staticmethod
    def _control(control: FloatArray) -> FloatArray:
        vector = np.asarray(control, dtype=np.float64)
        if vector.shape != (CONTROL_DIMENSION,):
            raise ValueError(f"control must have shape ({CONTROL_DIMENSION},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("control must be finite")
        return vector.copy()


DISCRETISATION_METHODS = ("forward_euler", "rk4")


def _check_step(step_seconds: float, substeps: int) -> None:
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    if int(substeps) != substeps or substeps < 1:
        raise ValueError("substeps must be a positive integer")
