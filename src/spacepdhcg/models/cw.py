"""Clohessy-Wiltshire rendezvous QP/SOCP benchmark family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from spacepdhcg.cqp import (
    CanonicalCQP,
    ConeBlock,
    ConeKind,
    CQPStructure,
    CQPValues,
    CSCStructure,
)

NX = 6
NU = 3
SOC_SLOTS = 4
FloatArray = NDArray[np.float64]


class ThrustConstraint(StrEnum):
    """Control constraint used by the CW benchmark."""

    BOX = "box"
    SECOND_ORDER_CONE = "soc"


def cw_continuous_matrices(mean_motion: float) -> tuple[FloatArray, FloatArray]:
    """Return continuous HCW matrices for state ``[x,y,z,vx,vy,vz]``."""

    if not np.isfinite(mean_motion) or mean_motion <= 0:
        raise ValueError("mean_motion must be finite and positive")

    n = float(mean_motion)
    dynamics = np.zeros((NX, NX), dtype=np.float64)
    dynamics[:3, 3:] = np.eye(3)
    dynamics[3, 0] = 3.0 * n**2
    dynamics[3, 4] = 2.0 * n
    dynamics[4, 3] = -2.0 * n
    dynamics[5, 2] = -(n**2)

    control = np.zeros((NX, NU), dtype=np.float64)
    control[3:, :] = np.eye(3)
    return dynamics, control


def discretise_cw(mean_motion: float, step_seconds: float) -> tuple[FloatArray, FloatArray]:
    """Exact zero-order-hold discretisation through an augmented matrix exponential."""

    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")

    dynamics, control = cw_continuous_matrices(mean_motion)
    augmented = np.zeros((NX + NU, NX + NU), dtype=np.float64)
    augmented[:NX, :NX] = dynamics
    augmented[:NX, NX:] = control
    exponential = la.expm(augmented * step_seconds)
    return exponential[:NX, :NX], exponential[:NX, NX:]


@dataclass(frozen=True, slots=True)
class CWRendezvousConfig:
    """Numerical configuration for the deterministic rendezvous benchmark."""

    intervals: int = 40
    step_seconds: float = 20.0
    mean_motion: float = 1.13e-3
    max_component_acceleration: float = 5.0e-2
    thrust_constraint: ThrustConstraint | str = ThrustConstraint.BOX
    state_weights: tuple[float, float, float, float, float, float] = (
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
    )
    control_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "thrust_constraint", ThrustConstraint(self.thrust_constraint))
        if self.intervals < 2:
            raise ValueError("intervals must be at least two")
        if self.step_seconds <= 0 or not np.isfinite(self.step_seconds):
            raise ValueError("step_seconds must be finite and positive")
        if self.mean_motion <= 0 or not np.isfinite(self.mean_motion):
            raise ValueError("mean_motion must be finite and positive")
        if self.max_component_acceleration <= 0:
            raise ValueError("max_component_acceleration must be positive")
        if np.any(np.asarray(self.state_weights) <= 0):
            raise ValueError("state weights must be positive")
        if np.any(np.asarray(self.control_weights) <= 0):
            raise ValueError("control weights must be positive")


@dataclass(frozen=True, slots=True)
class CWRendezvousLayout:
    intervals: int
    thrust_constraint: ThrustConstraint

    @property
    def state_variable_count(self) -> int:
        return (self.intervals + 1) * NX

    @property
    def control_variable_count(self) -> int:
        return self.intervals * NU

    @property
    def n_variables(self) -> int:
        return self.state_variable_count + self.control_variable_count

    @property
    def n_constraints(self) -> int:
        scalar_control_rows = (
            self.intervals * NU if self.thrust_constraint is ThrustConstraint.BOX else 0
        )
        return NX + self.intervals * NX + NX + scalar_control_rows

    @property
    def n_affine_constraints(self) -> int:
        if self.thrust_constraint is ThrustConstraint.SECOND_ORDER_CONE:
            return self.intervals * SOC_SLOTS
        return 0

    @property
    def initial_rows(self) -> slice:
        return slice(0, NX)

    @property
    def dynamics_rows(self) -> slice:
        start = NX
        return slice(start, start + self.intervals * NX)

    @property
    def terminal_rows(self) -> slice:
        start = self.dynamics_rows.stop
        return slice(start, start + NX)

    @property
    def control_rows(self) -> slice:
        start = self.terminal_rows.stop
        size = self.intervals * NU if self.thrust_constraint is ThrustConstraint.BOX else 0
        return slice(start, start + size)

    def state_slice(self, index: int) -> slice:
        if not 0 <= index <= self.intervals:
            raise IndexError("state index outside trajectory")
        return slice(index * NX, (index + 1) * NX)

    def control_slice(self, index: int) -> slice:
        if not 0 <= index < self.intervals:
            raise IndexError("control index outside trajectory")
        start = self.state_variable_count + index * NU
        return slice(start, start + NU)


@dataclass(frozen=True, slots=True)
class CWRendezvousDiagnostics:
    initial_error_inf: float
    terminal_error_inf: float
    dynamics_defect_inf: float
    control_violation_inf: float
    maximum_component_acceleration: float
    maximum_acceleration_norm: float

    def feasible(self, tolerance: float = 1.0e-6) -> bool:
        return max(
            self.initial_error_inf,
            self.terminal_error_inf,
            self.dynamics_defect_inf,
            self.control_violation_inf,
        ) <= tolerance


class CWRendezvousProblem:
    """Fixed-pattern CW rendezvous QP or SOCP with mutable boundary states."""

    def __init__(self, config: CWRendezvousConfig | None = None) -> None:
        self.config = config or CWRendezvousConfig()
        self.layout = CWRendezvousLayout(
            self.config.intervals,
            self.config.thrust_constraint,
        )
        self.ad, self.bd = discretise_cw(
            self.config.mean_motion,
            self.config.step_seconds,
        )
        quadratic, constraint, affine_cone = self._build_matrices()
        affine_structure = (
            None if affine_cone is None else CSCStructure.from_matrix(affine_cone)
        )
        cone_blocks = (
            tuple(
                ConeBlock(
                    kind=ConeKind.SECOND_ORDER,
                    start=interval * SOC_SLOTS,
                    vector_dimension=2,
                )
                for interval in range(self.config.intervals)
            )
            if affine_structure is not None
            else ()
        )
        self.structure = CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(constraint),
            affine_cone=affine_structure,
            affine_cones=cone_blocks,
        )
        self._quadratic_values = self.structure.quadratic.values_from(quadratic)
        self._constraint_values = self.structure.constraint.values_from(constraint)
        self._affine_values = (
            np.empty(0, dtype=np.float64)
            if affine_cone is None
            else self.structure.affine_cone.values_from(affine_cone)
        )

    def _build_matrices(
        self,
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix | None]:
        state_weights = np.asarray(self.config.state_weights, dtype=np.float64)
        control_weights = np.asarray(self.config.control_weights, dtype=np.float64)
        diagonal = np.concatenate(
            (
                np.tile(2.0 * state_weights, self.config.intervals + 1),
                np.tile(2.0 * control_weights, self.config.intervals),
            )
        )
        quadratic = sp.diags(diagonal, format="csc")

        layout = self.layout
        constraint = sp.lil_matrix(
            (layout.n_constraints, layout.n_variables),
            dtype=np.float64,
        )
        constraint[layout.initial_rows, layout.state_slice(0)] = np.eye(NX)

        dynamics_start = layout.dynamics_rows.start
        for interval in range(self.config.intervals):
            rows = slice(dynamics_start + interval * NX, dynamics_start + (interval + 1) * NX)
            constraint[rows, layout.state_slice(interval)] = -self.ad
            constraint[rows, layout.state_slice(interval + 1)] = np.eye(NX)
            constraint[rows, layout.control_slice(interval)] = -self.bd

        constraint[layout.terminal_rows, layout.state_slice(self.config.intervals)] = np.eye(NX)
        if self.config.thrust_constraint is ThrustConstraint.BOX:
            constraint[layout.control_rows, layout.state_variable_count :] = sp.eye(
                layout.control_variable_count,
                format="csc",
            )

        scalar = constraint.tocsc()
        scalar.sum_duplicates()
        scalar.sort_indices()

        if self.config.thrust_constraint is ThrustConstraint.SECOND_ORDER_CONE:
            affine = sp.lil_matrix(
                (layout.n_affine_constraints, layout.n_variables),
                dtype=np.float64,
            )
            for interval in range(self.config.intervals):
                start = interval * SOC_SLOTS
                affine[start : start + NU, layout.control_slice(interval)] = np.eye(NU)
            affine_result = affine.tocsc()
            affine_result.sum_duplicates()
            affine_result.sort_indices()
        else:
            affine_result = None
        return quadratic, scalar, affine_result

    def values(self, initial_state: FloatArray, target_state: FloatArray) -> CQPValues:
        initial = self._state(initial_state, "initial_state")
        target = self._state(target_state, "target_state")
        layout = self.layout

        reference = np.linspace(initial, target, self.config.intervals + 1)
        state_weights = np.asarray(self.config.state_weights, dtype=np.float64)
        linear = np.zeros(layout.n_variables, dtype=np.float64)
        linear[: layout.state_variable_count] = (-2.0 * reference * state_weights).reshape(-1)

        lower = np.zeros(layout.n_constraints, dtype=np.float64)
        upper = np.zeros(layout.n_constraints, dtype=np.float64)
        lower[layout.initial_rows] = initial
        upper[layout.initial_rows] = initial
        lower[layout.terminal_rows] = target
        upper[layout.terminal_rows] = target
        if self.config.thrust_constraint is ThrustConstraint.BOX:
            acceleration = self.config.max_component_acceleration
            lower[layout.control_rows] = -acceleration
            upper[layout.control_rows] = acceleration

        if self.config.thrust_constraint is ThrustConstraint.SECOND_ORDER_CONE:
            affine_offset = np.tile(
                np.array(
                    [0.0, 0.0, 0.0, self.config.max_component_acceleration],
                    dtype=np.float64,
                ),
                self.config.intervals,
            )
        else:
            affine_offset = np.empty(0, dtype=np.float64)

        return CQPValues(
            quadratic=self._quadratic_values.copy(),
            constraint=self._constraint_values.copy(),
            linear=linear,
            lower=lower,
            upper=upper,
            affine_cone=self._affine_values.copy(),
            affine_offset=affine_offset,
            variable_lower=np.full(layout.n_variables, -np.inf, dtype=np.float64),
            variable_upper=np.full(layout.n_variables, np.inf, dtype=np.float64),
        ).validated(self.structure)

    def canonical(self, initial_state: FloatArray, target_state: FloatArray) -> CanonicalCQP:
        return CanonicalCQP(self.structure, self.values(initial_state, target_state))

    def decode(self, decision: FloatArray) -> tuple[FloatArray, FloatArray]:
        vector = np.asarray(decision, dtype=np.float64)
        if vector.shape != (self.layout.n_variables,):
            raise ValueError("decision vector has the wrong shape")
        states = vector[: self.layout.state_variable_count].reshape(
            self.config.intervals + 1,
            NX,
        )
        controls = vector[self.layout.state_variable_count :].reshape(self.config.intervals, NU)
        return states.copy(), controls.copy()

    def diagnostics(
        self,
        decision: FloatArray,
        initial_state: FloatArray,
        target_state: FloatArray,
    ) -> CWRendezvousDiagnostics:
        initial = self._state(initial_state, "initial_state")
        target = self._state(target_state, "target_state")
        states, controls = self.decode(decision)
        predicted = states[:-1] @ self.ad.T + controls @ self.bd.T
        defect = states[1:] - predicted
        component_acceleration = np.abs(controls)
        acceleration_norm = np.linalg.norm(controls, axis=1)
        if self.config.thrust_constraint is ThrustConstraint.BOX:
            violation = np.maximum(
                component_acceleration - self.config.max_component_acceleration,
                0.0,
            )
        else:
            violation = np.maximum(
                acceleration_norm - self.config.max_component_acceleration,
                0.0,
            )
        return CWRendezvousDiagnostics(
            initial_error_inf=float(np.max(np.abs(states[0] - initial))),
            terminal_error_inf=float(np.max(np.abs(states[-1] - target))),
            dynamics_defect_inf=float(np.max(np.abs(defect))),
            control_violation_inf=float(np.max(violation)),
            maximum_component_acceleration=float(np.max(component_acceleration)),
            maximum_acceleration_norm=float(np.max(acceleration_norm)),
        )

    @staticmethod
    def _state(state: FloatArray, name: str) -> FloatArray:
        result = np.asarray(state, dtype=np.float64)
        if result.shape != (NX,):
            raise ValueError(f"{name} must have shape ({NX},)")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return result.copy()
