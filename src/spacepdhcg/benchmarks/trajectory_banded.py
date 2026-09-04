"""Deterministic trajectory-banded QP/SOCP fixtures with a known optimum.

The fixtures are not intended to model one particular spacecraft. They exercise the sparse
stage structure, equality dynamics, scalar bounds and native affine cones that occur inside
SCvx subproblems. The objective is centred on a generated feasible trajectory, making that
trajectory the unique global optimum and providing an exact cross-solver oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
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

FloatArray = NDArray[np.float64]


class BandedControlConstraint(StrEnum):
    """Control-set family used by a generated fixture."""

    BOX = "box"
    SECOND_ORDER_CONE = "soc"


@dataclass(frozen=True, slots=True)
class TrajectoryBandedConfig:
    intervals: int = 8
    state_dimension: int = 4
    control_dimension: int = 3
    seed: int = 17
    control_radius: float = 0.75
    dynamics_spectral_norm: float = 0.85
    weight_log10_span: float = 0.5
    control_constraint: BandedControlConstraint | str = BandedControlConstraint.BOX

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_constraint",
            BandedControlConstraint(self.control_constraint),
        )
        if self.intervals < 2:
            raise ValueError("intervals must be at least two")
        if self.state_dimension < 1:
            raise ValueError("state_dimension must be positive")
        if self.control_dimension < 2:
            raise ValueError("control_dimension must be at least two")
        if not np.isfinite(self.control_radius) or self.control_radius <= 0:
            raise ValueError("control_radius must be finite and positive")
        if not 0 < self.dynamics_spectral_norm < 1:
            raise ValueError("dynamics_spectral_norm must lie strictly between zero and one")
        if not np.isfinite(self.weight_log10_span) or self.weight_log10_span < 0:
            raise ValueError("weight_log10_span must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TrajectoryBandedLayout:
    intervals: int
    state_dimension: int
    control_dimension: int
    control_constraint: BandedControlConstraint

    @property
    def state_variable_count(self) -> int:
        return (self.intervals + 1) * self.state_dimension

    @property
    def control_variable_count(self) -> int:
        return self.intervals * self.control_dimension

    @property
    def n_variables(self) -> int:
        return self.state_variable_count + self.control_variable_count

    @property
    def n_scalar_constraints(self) -> int:
        core = self.state_dimension * (self.intervals + 2)
        if self.control_constraint is BandedControlConstraint.BOX:
            return core + self.control_variable_count
        return core

    @property
    def n_affine_cone_rows(self) -> int:
        if self.control_constraint is BandedControlConstraint.SECOND_ORDER_CONE:
            return self.intervals * (self.control_dimension + 1)
        return 0

    @property
    def initial_rows(self) -> slice:
        return slice(0, self.state_dimension)

    @property
    def dynamics_rows(self) -> slice:
        start = self.initial_rows.stop
        return slice(start, start + self.intervals * self.state_dimension)

    @property
    def terminal_rows(self) -> slice:
        start = self.dynamics_rows.stop
        return slice(start, start + self.state_dimension)

    @property
    def control_rows(self) -> slice:
        start = self.terminal_rows.stop
        size = (
            self.control_variable_count
            if self.control_constraint is BandedControlConstraint.BOX
            else 0
        )
        return slice(start, start + size)

    def state_slice(self, index: int) -> slice:
        if not 0 <= index <= self.intervals:
            raise IndexError("state index outside trajectory")
        start = index * self.state_dimension
        return slice(start, start + self.state_dimension)

    def control_slice(self, index: int) -> slice:
        if not 0 <= index < self.intervals:
            raise IndexError("control index outside trajectory")
        start = self.state_variable_count + index * self.control_dimension
        return slice(start, start + self.control_dimension)


@dataclass(frozen=True, slots=True)
class TrajectoryBandedDiagnostics:
    solution_error_inf: float
    scalar_violation_inf: float
    dynamics_defect_inf: float
    control_violation_inf: float
    objective_gap_abs: float

    def acceptable(self, tolerance: float = 1.0e-6) -> bool:
        return (
            max(
                self.solution_error_inf,
                self.scalar_violation_inf,
                self.dynamics_defect_inf,
                self.control_violation_inf,
                self.objective_gap_abs,
            )
            <= tolerance
        )


class TrajectoryBandedFixture:
    """Generated fixed-pattern problem with a unique known optimum."""

    def __init__(self, config: TrajectoryBandedConfig | None = None) -> None:
        self.config = config or TrajectoryBandedConfig()
        self.layout = TrajectoryBandedLayout(
            intervals=self.config.intervals,
            state_dimension=self.config.state_dimension,
            control_dimension=self.config.control_dimension,
            control_constraint=self.config.control_constraint,
        )
        rng = np.random.default_rng(self.config.seed)
        self.dynamics, self.control_maps = self._generate_dynamics(rng)
        self.known_states, self.known_controls = self._generate_feasible_trajectory(rng)
        self.known_solution = np.concatenate(
            (self.known_states.reshape(-1), self.known_controls.reshape(-1))
        )
        quadratic, scalar, affine = self._build_matrices()
        affine_structure = None if affine is None else CSCStructure.from_matrix(affine)
        cone_blocks = self._cone_blocks() if affine_structure is not None else ()
        self.structure = CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(scalar),
            affine_cone=affine_structure,
            affine_cones=cone_blocks,
        )
        self.values = self._build_values(quadratic, scalar, affine)
        self.canonical = CanonicalCQP(self.structure, self.values)
        self.known_objective = self.objective(self.known_solution)

    def _generate_dynamics(self, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        nx = self.config.state_dimension
        nu = self.config.control_dimension
        dynamics = np.empty((self.config.intervals, nx, nx), dtype=np.float64)
        control_maps = rng.normal(scale=0.35, size=(self.config.intervals, nx, nu))
        for interval in range(self.config.intervals):
            raw = rng.normal(size=(nx, nx))
            norm = np.linalg.norm(raw, ord=2)
            dynamics[interval] = self.config.dynamics_spectral_norm * raw / norm
        return dynamics, control_maps

    def _generate_feasible_trajectory(
        self,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray]:
        controls = rng.normal(size=(self.config.intervals, self.config.control_dimension))
        norms = np.linalg.norm(controls, axis=1, keepdims=True)
        fractions = rng.uniform(0.15, 0.55, size=(self.config.intervals, 1))
        controls *= fractions * self.config.control_radius / norms

        states = np.empty(
            (self.config.intervals + 1, self.config.state_dimension),
            dtype=np.float64,
        )
        states[0] = rng.normal(scale=0.5, size=self.config.state_dimension)
        for interval in range(self.config.intervals):
            states[interval + 1] = (
                self.dynamics[interval] @ states[interval]
                + self.control_maps[interval] @ controls[interval]
            )
        return states, controls

    def _build_matrices(
        self,
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix | None]:
        layout = self.layout
        rng = np.random.default_rng(self.config.seed + 1)
        exponents = rng.uniform(
            -self.config.weight_log10_span,
            self.config.weight_log10_span,
            layout.n_variables,
        )
        quadratic = sp.diags(np.power(10.0, exponents), format="csc")

        nx = self.config.state_dimension
        nu = self.config.control_dimension
        intervals = self.config.intervals
        dynamics_nonzeros = intervals * (nx * nx + nx + nx * nu)
        box_nonzeros = (
            intervals * nu if self.config.control_constraint is BandedControlConstraint.BOX else 0
        )
        nonzeros = 2 * nx + dynamics_nonzeros + box_nonzeros
        rows = np.empty(nonzeros, dtype=np.int64)
        columns = np.empty(nonzeros, dtype=np.int64)
        data = np.empty(nonzeros, dtype=np.float64)
        cursor = 0

        initial = np.arange(nx, dtype=np.int64)
        rows[cursor : cursor + nx] = initial
        columns[cursor : cursor + nx] = initial
        data[cursor : cursor + nx] = 1.0
        cursor += nx
        dense_rows = np.repeat(initial, nx)
        dense_columns = np.tile(initial, nx)
        control_columns = np.tile(np.arange(nu, dtype=np.int64), nx)
        control_rows = np.repeat(initial, nu)
        for interval in range(self.config.intervals):
            row_start = layout.dynamics_rows.start + interval * nx
            state_start = interval * nx
            count = nx * nx
            rows[cursor : cursor + count] = row_start + dense_rows
            columns[cursor : cursor + count] = state_start + dense_columns
            data[cursor : cursor + count] = -self.dynamics[interval].reshape(-1)
            cursor += count
            rows[cursor : cursor + nx] = row_start + initial
            columns[cursor : cursor + nx] = state_start + nx + initial
            data[cursor : cursor + nx] = 1.0
            cursor += nx
            count = nx * nu
            rows[cursor : cursor + count] = row_start + control_rows
            columns[cursor : cursor + count] = (
                layout.state_variable_count + interval * nu + control_columns
            )
            data[cursor : cursor + count] = -self.control_maps[interval].reshape(-1)
            cursor += count

        rows[cursor : cursor + nx] = layout.terminal_rows.start + initial
        columns[cursor : cursor + nx] = intervals * nx + initial
        data[cursor : cursor + nx] = 1.0
        cursor += nx
        if self.config.control_constraint is BandedControlConstraint.BOX:
            controls = np.arange(intervals * nu, dtype=np.int64)
            rows[cursor : cursor + controls.size] = layout.control_rows.start + controls
            columns[cursor : cursor + controls.size] = layout.state_variable_count + controls
            data[cursor : cursor + controls.size] = 1.0
            cursor += controls.size
        if cursor != nonzeros:
            raise AssertionError("trajectory sparse assembly count mismatch")
        scalar_result = sp.coo_matrix(
            (data, (rows, columns)),
            shape=(layout.n_scalar_constraints, layout.n_variables),
            dtype=np.float64,
        ).tocsc()
        scalar_result.sum_duplicates()
        scalar_result.sort_indices()

        if self.config.control_constraint is BandedControlConstraint.SECOND_ORDER_CONE:
            slots = nu + 1
            controls = np.arange(intervals * nu, dtype=np.int64)
            affine_rows = (controls // nu) * slots + controls % nu
            affine_columns = layout.state_variable_count + controls
            affine_result = sp.coo_matrix(
                (
                    np.ones(controls.size, dtype=np.float64),
                    (affine_rows, affine_columns),
                ),
                shape=(layout.n_affine_cone_rows, layout.n_variables),
                dtype=np.float64,
            ).tocsc()
            affine_result.sort_indices()
        else:
            affine_result = None
        return quadratic, scalar_result, affine_result

    def _cone_blocks(self) -> tuple[ConeBlock, ...]:
        slots = self.config.control_dimension + 1
        return tuple(
            ConeBlock(
                kind=ConeKind.SECOND_ORDER,
                start=interval * slots,
                vector_dimension=self.config.control_dimension - 1,
            )
            for interval in range(self.config.intervals)
        )

    def _build_values(
        self,
        quadratic: sp.csc_matrix,
        scalar: sp.csc_matrix,
        affine: sp.csc_matrix | None,
    ) -> CQPValues:
        layout = self.layout
        quadratic_values = self.structure.quadratic.values_from(quadratic)
        linear = -(quadratic @ self.known_solution)
        lower = np.zeros(layout.n_scalar_constraints, dtype=np.float64)
        upper = np.zeros(layout.n_scalar_constraints, dtype=np.float64)
        lower[layout.initial_rows] = self.known_states[0]
        upper[layout.initial_rows] = self.known_states[0]
        lower[layout.terminal_rows] = self.known_states[-1]
        upper[layout.terminal_rows] = self.known_states[-1]
        if self.config.control_constraint is BandedControlConstraint.BOX:
            lower[layout.control_rows] = -self.config.control_radius
            upper[layout.control_rows] = self.config.control_radius

        if affine is None:
            affine_values = np.empty(0, dtype=np.float64)
            affine_offset = np.empty(0, dtype=np.float64)
        else:
            affine_values = self.structure.affine_cone.values_from(affine)
            slots = self.config.control_dimension + 1
            affine_offset = np.zeros(layout.n_affine_cone_rows, dtype=np.float64)
            affine_offset.reshape(self.config.intervals, slots)[:, -1] = self.config.control_radius

        return CQPValues(
            quadratic=quadratic_values,
            constraint=self.structure.constraint.values_from(scalar),
            linear=np.asarray(linear, dtype=np.float64),
            lower=lower,
            upper=upper,
            affine_cone=affine_values,
            affine_offset=affine_offset,
            variable_lower=np.full(layout.n_variables, -np.inf, dtype=np.float64),
            variable_upper=np.full(layout.n_variables, np.inf, dtype=np.float64),
        ).validated(self.structure)

    def objective(self, decision: FloatArray) -> float:
        vector = np.asarray(decision, dtype=np.float64)
        if vector.shape != (self.layout.n_variables,):
            raise ValueError("decision vector has the wrong shape")
        quadratic = self.structure.quadratic.matrix(self.values.quadratic)
        return float(0.5 * vector @ quadratic @ vector + self.values.linear @ vector)

    def diagnostics(self, decision: FloatArray) -> TrajectoryBandedDiagnostics:
        vector = np.asarray(decision, dtype=np.float64)
        if vector.shape != (self.layout.n_variables,):
            raise ValueError("decision vector has the wrong shape")
        states = vector[: self.layout.state_variable_count].reshape(
            self.config.intervals + 1,
            self.config.state_dimension,
        )
        controls = vector[self.layout.state_variable_count :].reshape(
            self.config.intervals,
            self.config.control_dimension,
        )
        predicted = np.einsum("kij,kj->ki", self.dynamics, states[:-1]) + np.einsum(
            "kij,kj->ki",
            self.control_maps,
            controls,
        )
        dynamics_defect = states[1:] - predicted
        if self.config.control_constraint is BandedControlConstraint.BOX:
            control_violation = np.maximum(
                np.abs(controls) - self.config.control_radius,
                0.0,
            )
        else:
            control_violation = np.maximum(
                np.linalg.norm(controls, axis=1) - self.config.control_radius,
                0.0,
            )
        scalar_matrix = self.structure.constraint.matrix(self.values.constraint)
        scalar_activity = scalar_matrix @ vector
        scalar_violation = np.maximum(
            np.maximum(self.values.lower - scalar_activity, 0.0),
            np.maximum(scalar_activity - self.values.upper, 0.0),
        )
        return TrajectoryBandedDiagnostics(
            solution_error_inf=float(np.max(np.abs(vector - self.known_solution))),
            scalar_violation_inf=float(np.max(scalar_violation)),
            dynamics_defect_inf=float(np.max(np.abs(dynamics_defect))),
            control_violation_inf=float(np.max(control_violation)),
            objective_gap_abs=abs(self.objective(vector) - self.known_objective),
        )
