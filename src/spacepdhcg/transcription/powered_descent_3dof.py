"""Fixed-pattern convex subproblem for 3-DoF powered-descent SCvx.

This module is the transparent CPU reference transcription. It intentionally uses a fixed grid
and forward-Euler linearisation by default so every sparse index and cone block is known before
the first solve.  ``PoweredDescentSCvxConfig(discretisation="rk4")`` swaps the coefficient
values for the exact linearisation of a variational RK4 zero-order-hold map while keeping the
same fixed pattern and persistent backend lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from spacepdhcg.models.powered_descent_3dof import (
    CONTROL_DIMENSION,
    DISCRETISATION_METHODS,
    STATE_DIMENSION,
    PoweredDescent3DOFModel,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PoweredDescentSCvxConfig:
    """Numerical weights and trust-region scaling for one convex subproblem."""

    intervals: int = 10
    step_seconds: float = 2.0
    trust_radius: float = 1.0
    virtual_l1_weight: float = 1.0e5
    virtual_quadratic_weight: float = 1.0e-8
    virtual_epigraph_regularisation: float = 1.0e-10
    fuel_weight: float = 1.0e-3
    state_tracking_weights: tuple[float, ...] = (
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-8,
    )
    control_tracking_weights: tuple[float, ...] = (
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
    )
    state_trust_scales: tuple[float, ...] = (
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-3,
    )
    control_trust_scales: tuple[float, ...] = (
        1.0 / 15_000.0,
        1.0 / 15_000.0,
        1.0 / 15_000.0,
        1.0 / 15_000.0,
    )
    # ``forward_euler`` is the frozen reference transcription (benchmark fixtures depend on it).
    # ``rk4`` linearises the variational RK4 zero-order-hold map instead; the CSC pattern is
    # identical (dense state/control blocks), only the coefficient values change.
    discretisation: str = "forward_euler"
    integration_substeps: int = 1

    def __post_init__(self) -> None:
        if self.intervals < 2:
            raise ValueError("intervals must be at least two")
        if not np.isfinite(self.step_seconds) or self.step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")
        if self.discretisation not in DISCRETISATION_METHODS:
            raise ValueError(f"discretisation must be one of {DISCRETISATION_METHODS}")
        if int(self.integration_substeps) != self.integration_substeps or (
            self.integration_substeps < 1
        ):
            raise ValueError("integration_substeps must be a positive integer")
        for name in (
            "trust_radius",
            "virtual_l1_weight",
            "virtual_quadratic_weight",
            "virtual_epigraph_regularisation",
            "fuel_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.trust_radius <= 0:
            raise ValueError("trust_radius must be positive")
        self._positive_vector(
            self.state_tracking_weights,
            STATE_DIMENSION,
            "state_tracking_weights",
        )
        self._positive_vector(
            self.control_tracking_weights,
            CONTROL_DIMENSION,
            "control_tracking_weights",
        )
        self._positive_vector(
            self.state_trust_scales,
            STATE_DIMENSION,
            "state_trust_scales",
        )
        self._positive_vector(
            self.control_trust_scales,
            CONTROL_DIMENSION,
            "control_trust_scales",
        )

    @staticmethod
    def _positive_vector(values: tuple[float, ...], size: int, name: str) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{name} must contain {size} finite positive values")


@dataclass(frozen=True, slots=True)
class PoweredDescentSCvxLayout:
    """Variable and row layout shared by every numerical update."""

    intervals: int

    @property
    def state_count(self) -> int:
        return (self.intervals + 1) * STATE_DIMENSION

    @property
    def control_count(self) -> int:
        return self.intervals * CONTROL_DIMENSION

    @property
    def virtual_count(self) -> int:
        return self.intervals * STATE_DIMENSION

    @property
    def virtual_epigraph_count(self) -> int:
        return self.virtual_count

    @property
    def control_offset(self) -> int:
        return self.state_count

    @property
    def virtual_offset(self) -> int:
        return self.control_offset + self.control_count

    @property
    def virtual_epigraph_offset(self) -> int:
        return self.virtual_offset + self.virtual_count

    @property
    def n_variables(self) -> int:
        return self.virtual_epigraph_offset + self.virtual_epigraph_count

    @property
    def initial_rows(self) -> slice:
        return slice(0, STATE_DIMENSION)

    @property
    def dynamics_rows(self) -> slice:
        start = self.initial_rows.stop
        return slice(start, start + self.intervals * STATE_DIMENSION)

    @property
    def terminal_rows(self) -> slice:
        start = self.dynamics_rows.stop
        return slice(start, start + 6)

    @property
    def virtual_epigraph_rows(self) -> slice:
        start = self.terminal_rows.stop
        return slice(start, start + 2 * self.virtual_count)

    @property
    def tilt_rows(self) -> slice:
        start = self.virtual_epigraph_rows.stop
        return slice(start, start + self.intervals)

    @property
    def n_scalar_constraints(self) -> int:
        return self.tilt_rows.stop

    @property
    def thrust_cone_rows(self) -> slice:
        return slice(0, 4 * self.intervals)

    @property
    def glide_cone_rows(self) -> slice:
        start = self.thrust_cone_rows.stop
        return slice(start, start + 3 * (self.intervals + 1))

    @property
    def stage_trust_cone_rows(self) -> slice:
        start = self.glide_cone_rows.stop
        return slice(start, start + 12 * self.intervals)

    @property
    def terminal_trust_cone_rows(self) -> slice:
        start = self.stage_trust_cone_rows.stop
        return slice(start, start + 8)

    @property
    def n_affine_cone_rows(self) -> int:
        return self.terminal_trust_cone_rows.stop

    def state_slice(self, node: int) -> slice:
        if not 0 <= node <= self.intervals:
            raise IndexError("state node outside trajectory")
        start = node * STATE_DIMENSION
        return slice(start, start + STATE_DIMENSION)

    def control_slice(self, interval: int) -> slice:
        if not 0 <= interval < self.intervals:
            raise IndexError("control interval outside trajectory")
        start = self.control_offset + interval * CONTROL_DIMENSION
        return slice(start, start + CONTROL_DIMENSION)

    def virtual_slice(self, interval: int) -> slice:
        if not 0 <= interval < self.intervals:
            raise IndexError("virtual-control interval outside trajectory")
        start = self.virtual_offset + interval * STATE_DIMENSION
        return slice(start, start + STATE_DIMENSION)

    def virtual_epigraph_slice(self, interval: int) -> slice:
        if not 0 <= interval < self.intervals:
            raise IndexError("virtual epigraph interval outside trajectory")
        start = self.virtual_epigraph_offset + interval * STATE_DIMENSION
        return slice(start, start + STATE_DIMENSION)


@dataclass(frozen=True, slots=True)
class PoweredDescentSCvxDiagnostics:
    scalar_violation_inf: float
    variable_bound_violation_inf: float
    cone_violation_inf: float
    linearised_dynamics_defect_inf: float
    nonlinear_dynamics_defect_inf: float
    terminal_error_inf: float
    virtual_control_inf: float

    @property
    def convex_violation_inf(self) -> float:
        return max(
            self.scalar_violation_inf,
            self.variable_bound_violation_inf,
            self.cone_violation_inf,
        )

    def convex_feasible(self, tolerance: float = 1.0e-7) -> bool:
        return self.convex_violation_inf <= tolerance


class _CSCValueIndex:
    def __init__(self, structure: CSCStructure) -> None:
        self.structure = structure
        self._positions: dict[tuple[int, int], int] = {}
        for column in range(structure.shape[1]):
            for position in range(structure.indptr[column], structure.indptr[column + 1]):
                self._positions[(int(structure.indices[position]), column)] = int(position)

    def set(self, values: FloatArray, row: int, column: int, value: float) -> None:
        try:
            position = self._positions[(row, column)]
        except KeyError as error:
            raise RuntimeError(f"entry ({row}, {column}) is absent from fixed pattern") from error
        values[position] = value


class PoweredDescent3DOFSubproblem:
    """Build repeated convex subproblems around updated reference trajectories."""

    def __init__(
        self,
        model: PoweredDescent3DOFModel | None = None,
        config: PoweredDescentSCvxConfig | None = None,
    ) -> None:
        self.model = model or PoweredDescent3DOFModel()
        self.config = config or PoweredDescentSCvxConfig()
        self.layout = PoweredDescentSCvxLayout(self.config.intervals)
        quadratic, scalar, affine = self._build_patterns()
        self.structure = CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(scalar),
            affine_cone=CSCStructure.from_matrix(affine),
            affine_cones=self._cone_blocks(),
        )
        self._quadratic_index = _CSCValueIndex(self.structure.quadratic)
        self._scalar_index = _CSCValueIndex(self.structure.constraint)
        if self.structure.affine_cone is None:
            raise AssertionError("powered-descent subproblem requires affine cones")
        self._affine_index = _CSCValueIndex(self.structure.affine_cone)
        self._quadratic_values = self._make_quadratic_values()

    def _build_patterns(self) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
        layout = self.layout
        quadratic_entries = {(index, index) for index in range(layout.n_variables)}
        scalar_entries: set[tuple[int, int]] = set()
        affine_entries: set[tuple[int, int]] = set()

        for index in range(STATE_DIMENSION):
            scalar_entries.add((layout.initial_rows.start + index, index))

        for interval in range(layout.intervals):
            row_start = layout.dynamics_rows.start + interval * STATE_DIMENSION
            state = layout.state_slice(interval)
            next_state = layout.state_slice(interval + 1)
            control = layout.control_slice(interval)
            virtual = layout.virtual_slice(interval)
            for row_offset in range(STATE_DIMENSION):
                row = row_start + row_offset
                for column in range(state.start, state.stop):
                    scalar_entries.add((row, column))
                scalar_entries.add((row, next_state.start + row_offset))
                for column in range(control.start, control.stop):
                    scalar_entries.add((row, column))
                scalar_entries.add((row, virtual.start + row_offset))

        terminal_state = layout.state_slice(layout.intervals)
        for index in range(6):
            scalar_entries.add((layout.terminal_rows.start + index, terminal_state.start + index))

        for flat_index in range(layout.virtual_count):
            positive_row = layout.virtual_epigraph_rows.start + 2 * flat_index
            negative_row = positive_row + 1
            virtual_column = layout.virtual_offset + flat_index
            epigraph_column = layout.virtual_epigraph_offset + flat_index
            scalar_entries.update(
                {
                    (positive_row, virtual_column),
                    (positive_row, epigraph_column),
                    (negative_row, virtual_column),
                    (negative_row, epigraph_column),
                }
            )

        for interval in range(layout.intervals):
            row = layout.tilt_rows.start + interval
            control = layout.control_slice(interval)
            scalar_entries.add((row, control.start + 2))
            scalar_entries.add((row, control.start + 3))

            thrust_start = layout.thrust_cone_rows.start + 4 * interval
            for component in range(CONTROL_DIMENSION):
                affine_entries.add((thrust_start + component, control.start + component))

        for node in range(layout.intervals + 1):
            glide_start = layout.glide_cone_rows.start + 3 * node
            state = layout.state_slice(node)
            for component in range(3):
                affine_entries.add((glide_start + component, state.start + component))

        for interval in range(layout.intervals):
            trust_start = layout.stage_trust_cone_rows.start + 12 * interval
            state = layout.state_slice(interval)
            control = layout.control_slice(interval)
            for component in range(STATE_DIMENSION):
                affine_entries.add((trust_start + component, state.start + component))
            for component in range(CONTROL_DIMENSION):
                affine_entries.add(
                    (trust_start + STATE_DIMENSION + component, control.start + component)
                )

        terminal_trust = layout.terminal_trust_cone_rows.start
        for component in range(STATE_DIMENSION):
            affine_entries.add((terminal_trust + component, terminal_state.start + component))

        return (
            self._pattern((layout.n_variables, layout.n_variables), quadratic_entries),
            self._pattern(
                (layout.n_scalar_constraints, layout.n_variables),
                scalar_entries,
            ),
            self._pattern(
                (layout.n_affine_cone_rows, layout.n_variables),
                affine_entries,
            ),
        )

    @staticmethod
    def _pattern(shape: tuple[int, int], entries: set[tuple[int, int]]) -> sp.csc_matrix:
        if not entries:
            return sp.csc_matrix(shape, dtype=np.float64)
        ordered = sorted(entries, key=lambda item: (item[1], item[0]))
        rows = np.asarray([row for row, _ in ordered], dtype=np.int64)
        columns = np.asarray([column for _, column in ordered], dtype=np.int64)
        matrix = sp.csc_matrix(
            (np.ones(len(ordered), dtype=np.float64), (rows, columns)),
            shape=shape,
        )
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def _cone_blocks(self) -> tuple[ConeBlock, ...]:
        layout = self.layout
        blocks: list[ConeBlock] = []
        blocks.extend(
            ConeBlock(ConeKind.SECOND_ORDER, 4 * interval, 2)
            for interval in range(layout.intervals)
        )
        blocks.extend(
            ConeBlock(
                ConeKind.SECOND_ORDER,
                layout.glide_cone_rows.start + 3 * node,
                1,
            )
            for node in range(layout.intervals + 1)
        )
        blocks.extend(
            ConeBlock(
                ConeKind.SECOND_ORDER,
                layout.stage_trust_cone_rows.start + 12 * interval,
                10,
            )
            for interval in range(layout.intervals)
        )
        blocks.append(
            ConeBlock(
                ConeKind.SECOND_ORDER,
                layout.terminal_trust_cone_rows.start,
                6,
            )
        )
        return tuple(blocks)

    def _make_quadratic_values(self) -> FloatArray:
        layout = self.layout
        diagonal = np.empty(layout.n_variables, dtype=np.float64)
        diagonal[: layout.state_count] = np.tile(
            np.asarray(self.config.state_tracking_weights),
            layout.intervals + 1,
        )
        diagonal[layout.control_offset : layout.virtual_offset] = np.tile(
            np.asarray(self.config.control_tracking_weights),
            layout.intervals,
        )
        diagonal[layout.virtual_offset : layout.virtual_epigraph_offset] = (
            self.config.virtual_quadratic_weight
        )
        diagonal[layout.virtual_epigraph_offset :] = self.config.virtual_epigraph_regularisation
        values = np.zeros(self.structure.quadratic.nnz, dtype=np.float64)
        for index, value in enumerate(diagonal):
            self._quadratic_index.set(values, index, index, float(value))
        return values

    def values(
        self,
        reference_states: FloatArray,
        reference_controls: FloatArray,
        initial_state: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
        *,
        trust_radius: float | None = None,
    ) -> CQPValues:
        states, controls = self._reference(reference_states, reference_controls)
        initial = self._vector(initial_state, STATE_DIMENSION, "initial_state")
        target_position_vector = self._vector(target_position, 3, "target_position")
        target_velocity_vector = self._vector(target_velocity, 3, "target_velocity")
        radius = self.config.trust_radius if trust_radius is None else float(trust_radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("trust_radius must be finite and positive")

        layout = self.layout
        linear = np.zeros(layout.n_variables, dtype=np.float64)
        state_weights = np.asarray(self.config.state_tracking_weights)
        control_weights = np.asarray(self.config.control_tracking_weights)
        linear[: layout.state_count] = (-states * state_weights).reshape(-1)
        linear[layout.control_offset : layout.virtual_offset] = (
            -controls * control_weights
        ).reshape(-1)
        sigma_indices = layout.control_offset + np.arange(layout.intervals) * CONTROL_DIMENSION + 3
        linear[sigma_indices] += self.config.fuel_weight * self.config.step_seconds
        linear[layout.virtual_epigraph_offset :] = self.config.virtual_l1_weight

        scalar_values = np.zeros(self.structure.constraint.nnz, dtype=np.float64)
        lower = np.full(layout.n_scalar_constraints, -np.inf, dtype=np.float64)
        upper = np.full(layout.n_scalar_constraints, np.inf, dtype=np.float64)
        for index in range(STATE_DIMENSION):
            row = layout.initial_rows.start + index
            self._scalar_index.set(scalar_values, row, index, 1.0)
            lower[row] = initial[index]
            upper[row] = initial[index]

        for interval in range(layout.intervals):
            discrete_state, discrete_control, offset = self.model.linearised_discrete_dynamics(
                states[interval],
                controls[interval],
                self.config.step_seconds,
                method=self.config.discretisation,
                substeps=self.config.integration_substeps,
            )
            row_start = layout.dynamics_rows.start + interval * STATE_DIMENSION
            state = layout.state_slice(interval)
            next_state = layout.state_slice(interval + 1)
            control = layout.control_slice(interval)
            virtual = layout.virtual_slice(interval)
            for row_offset in range(STATE_DIMENSION):
                row = row_start + row_offset
                for column_offset in range(STATE_DIMENSION):
                    self._scalar_index.set(
                        scalar_values,
                        row,
                        state.start + column_offset,
                        -float(discrete_state[row_offset, column_offset]),
                    )
                self._scalar_index.set(
                    scalar_values,
                    row,
                    next_state.start + row_offset,
                    1.0,
                )
                for column_offset in range(CONTROL_DIMENSION):
                    self._scalar_index.set(
                        scalar_values,
                        row,
                        control.start + column_offset,
                        -float(discrete_control[row_offset, column_offset]),
                    )
                self._scalar_index.set(
                    scalar_values,
                    row,
                    virtual.start + row_offset,
                    -1.0,
                )
                lower[row] = offset[row_offset]
                upper[row] = offset[row_offset]

        terminal = np.concatenate((target_position_vector, target_velocity_vector))
        terminal_state = layout.state_slice(layout.intervals)
        for index in range(6):
            row = layout.terminal_rows.start + index
            self._scalar_index.set(
                scalar_values,
                row,
                terminal_state.start + index,
                1.0,
            )
            lower[row] = terminal[index]
            upper[row] = terminal[index]

        for flat_index in range(layout.virtual_count):
            positive_row = layout.virtual_epigraph_rows.start + 2 * flat_index
            negative_row = positive_row + 1
            virtual_column = layout.virtual_offset + flat_index
            epigraph_column = layout.virtual_epigraph_offset + flat_index
            self._scalar_index.set(scalar_values, positive_row, virtual_column, 1.0)
            self._scalar_index.set(scalar_values, positive_row, epigraph_column, -1.0)
            self._scalar_index.set(scalar_values, negative_row, virtual_column, -1.0)
            self._scalar_index.set(scalar_values, negative_row, epigraph_column, -1.0)
            upper[positive_row] = 0.0
            upper[negative_row] = 0.0

        for interval in range(layout.intervals):
            row = layout.tilt_rows.start + interval
            control = layout.control_slice(interval)
            self._scalar_index.set(scalar_values, row, control.start + 2, -1.0)
            self._scalar_index.set(
                scalar_values,
                row,
                control.start + 3,
                self.model.config.tilt_cosine,
            )
            upper[row] = 0.0

        affine_values = np.zeros(self.structure.affine_cone.nnz, dtype=np.float64)
        affine_offset = np.zeros(layout.n_affine_cone_rows, dtype=np.float64)
        state_scales = np.asarray(self.config.state_trust_scales)
        control_scales = np.asarray(self.config.control_trust_scales)

        for interval in range(layout.intervals):
            control = layout.control_slice(interval)
            thrust_start = layout.thrust_cone_rows.start + 4 * interval
            for component in range(CONTROL_DIMENSION):
                self._affine_index.set(
                    affine_values,
                    thrust_start + component,
                    control.start + component,
                    1.0,
                )

        for node in range(layout.intervals + 1):
            state = layout.state_slice(node)
            glide_start = layout.glide_cone_rows.start + 3 * node
            self._affine_index.set(affine_values, glide_start, state.start, 1.0)
            self._affine_index.set(affine_values, glide_start + 1, state.start + 1, 1.0)
            self._affine_index.set(
                affine_values,
                glide_start + 2,
                state.start + 2,
                self.model.config.glide_slope_tangent,
            )

        for interval in range(layout.intervals):
            state = layout.state_slice(interval)
            control = layout.control_slice(interval)
            trust_start = layout.stage_trust_cone_rows.start + 12 * interval
            for component in range(STATE_DIMENSION):
                scale = state_scales[component]
                self._affine_index.set(
                    affine_values,
                    trust_start + component,
                    state.start + component,
                    scale,
                )
                affine_offset[trust_start + component] = -scale * states[interval, component]
            for component in range(CONTROL_DIMENSION):
                row = trust_start + STATE_DIMENSION + component
                scale = control_scales[component]
                self._affine_index.set(
                    affine_values,
                    row,
                    control.start + component,
                    scale,
                )
                affine_offset[row] = -scale * controls[interval, component]
            affine_offset[trust_start + 11] = radius

        terminal_trust = layout.terminal_trust_cone_rows.start
        for component in range(STATE_DIMENSION):
            scale = state_scales[component]
            self._affine_index.set(
                affine_values,
                terminal_trust + component,
                terminal_state.start + component,
                scale,
            )
            affine_offset[terminal_trust + component] = -scale * states[-1, component]
        affine_offset[terminal_trust + 7] = radius

        variable_lower = np.full(layout.n_variables, -np.inf, dtype=np.float64)
        variable_upper = np.full(layout.n_variables, np.inf, dtype=np.float64)
        for node in range(layout.intervals + 1):
            state = layout.state_slice(node)
            variable_lower[state.start + 2] = 0.0
            variable_lower[state.start + 6] = self.model.config.minimum_mass
        for interval in range(layout.intervals):
            control = layout.control_slice(interval)
            variable_lower[control.start + 3] = self.model.config.minimum_sigma
            variable_upper[control.start + 3] = self.model.config.maximum_thrust
        variable_lower[layout.virtual_epigraph_offset :] = 0.0

        return CQPValues(
            quadratic=self._quadratic_values.copy(),
            constraint=scalar_values,
            linear=linear,
            lower=lower,
            upper=upper,
            affine_cone=affine_values,
            affine_offset=affine_offset,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
        ).validated(self.structure)

    def canonical(self, *args, **kwargs) -> CanonicalCQP:
        return CanonicalCQP(self.structure, self.values(*args, **kwargs))

    def reference_decision(
        self,
        reference_states: FloatArray,
        reference_controls: FloatArray,
    ) -> FloatArray:
        states, controls = self._reference(reference_states, reference_controls)
        decision = np.zeros(self.layout.n_variables, dtype=np.float64)
        decision[: self.layout.state_count] = states.reshape(-1)
        decision[self.layout.control_offset : self.layout.virtual_offset] = controls.reshape(-1)
        return decision

    def decode(self, decision: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        vector = self._vector(decision, self.layout.n_variables, "decision")
        states = vector[: self.layout.state_count].reshape(
            self.layout.intervals + 1,
            STATE_DIMENSION,
        )
        controls = vector[self.layout.control_offset : self.layout.virtual_offset].reshape(
            self.layout.intervals, CONTROL_DIMENSION
        )
        virtual = vector[self.layout.virtual_offset : self.layout.virtual_epigraph_offset].reshape(
            self.layout.intervals, STATE_DIMENSION
        )
        epigraph = vector[self.layout.virtual_epigraph_offset :].reshape(
            self.layout.intervals,
            STATE_DIMENSION,
        )
        return states.copy(), controls.copy(), virtual.copy(), epigraph.copy()

    def diagnostics(
        self,
        decision: FloatArray,
        values: CQPValues,
    ) -> PoweredDescentSCvxDiagnostics:
        vector = self._vector(decision, self.layout.n_variables, "decision")
        numerical = values.validated(self.structure)
        scalar = self.structure.constraint.matrix(numerical.constraint) @ vector
        scalar_violation = np.maximum(
            np.maximum(numerical.lower - scalar, 0.0),
            np.maximum(scalar - numerical.upper, 0.0),
        )
        variable_violation = np.maximum(
            np.maximum(numerical.variable_lower - vector, 0.0),
            np.maximum(vector - numerical.variable_upper, 0.0),
        )
        affine = (
            self.structure.affine_cone.matrix(numerical.affine_cone) @ vector
            + numerical.affine_offset
        )
        cone_violation = 0.0
        for cone in self.structure.affine_cones:
            segment = affine[cone.start : cone.stop]
            cone_violation = max(
                cone_violation,
                float(max(np.linalg.norm(segment[:-1]) - segment[-1], 0.0)),
            )

        states, controls, virtual, _ = self.decode(vector)
        nonlinear_defects = np.empty((self.layout.intervals, STATE_DIMENSION))
        for interval in range(self.layout.intervals):
            nonlinear_defects[interval] = states[interval + 1] - self.model.discrete_step(
                states[interval],
                controls[interval],
                self.config.step_seconds,
                method=self.config.discretisation,
                substeps=self.config.integration_substeps,
            )
        terminal_target = numerical.lower[self.layout.terminal_rows]
        terminal_actual = np.concatenate((states[-1, :3], states[-1, 3:6]))
        dynamics_activity = scalar[self.layout.dynamics_rows]
        dynamics_target = numerical.lower[self.layout.dynamics_rows]
        return PoweredDescentSCvxDiagnostics(
            scalar_violation_inf=float(np.max(scalar_violation)),
            variable_bound_violation_inf=float(np.max(variable_violation)),
            cone_violation_inf=cone_violation,
            linearised_dynamics_defect_inf=float(
                np.max(np.abs(dynamics_activity - dynamics_target))
            ),
            nonlinear_dynamics_defect_inf=float(np.max(np.abs(nonlinear_defects))),
            terminal_error_inf=float(np.max(np.abs(terminal_actual - terminal_target))),
            virtual_control_inf=float(np.max(np.abs(virtual))),
        )

    def _reference(
        self,
        states: FloatArray,
        controls: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        state_array = np.asarray(states, dtype=np.float64)
        control_array = np.asarray(controls, dtype=np.float64)
        expected_states = (self.layout.intervals + 1, STATE_DIMENSION)
        expected_controls = (self.layout.intervals, CONTROL_DIMENSION)
        if state_array.shape != expected_states:
            raise ValueError(f"reference_states must have shape {expected_states}")
        if control_array.shape != expected_controls:
            raise ValueError(f"reference_controls must have shape {expected_controls}")
        if not np.all(np.isfinite(state_array)) or not np.all(np.isfinite(control_array)):
            raise ValueError("reference trajectory must be finite")
        if np.any(state_array[:, 6] <= 0):
            raise ValueError("reference masses must be positive")
        return state_array.copy(), control_array.copy()

    @staticmethod
    def _vector(values: FloatArray, size: int, name: str) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite")
        return vector.copy()
