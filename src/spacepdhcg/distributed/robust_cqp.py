"""Monolithic reference assembly for scenario-aware conic quadratic programs.

The production multi-GPU path will never materialise this matrix on one host. This
module is the correctness oracle: it embeds identical fixed local structures, weights
scenario objectives by probability, appends exact non-anticipativity rows, and exposes
stable slices for comparing distributed implementations against the monolithic solve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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

from .layout import BlockArrowLayout
from .scenario_tree import ScenarioTree

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RobustPrimal:
    """Decoded monolithic primal vector."""

    local: tuple[FloatArray, ...]
    consensus: tuple[FloatArray, ...]


@dataclass(frozen=True, slots=True)
class RobustDual:
    """Decoded scalar, non-anticipativity, and affine-cone dual blocks."""

    local_scalar: tuple[FloatArray, ...]
    nonanticipativity: FloatArray
    local_affine: tuple[FloatArray, ...]


@dataclass(frozen=True, slots=True)
class RiskEpigraphLayout:
    """Slices for a sparse worst-case or CVaR objective epigraph."""

    base_variables: int
    scenario_costs: slice
    worst_case: int | None
    threshold: int | None
    excesses: slice
    measure: Literal["worst", "cvar"]
    alpha: float | None


class ScenarioCQPBundle:
    """Fixed-pattern block-arrow bundle of same-topology local CQPs."""

    def __init__(
        self,
        tree: ScenarioTree,
        local_structure: CQPStructure,
        *,
        state_dimension: int,
        control_dimension: int,
        local_auxiliary_dimension: int,
    ) -> None:
        self.tree = tree
        self.local_structure = local_structure
        self.layout = BlockArrowLayout(
            tree,
            state_dimension=state_dimension,
            control_dimension=control_dimension,
            local_auxiliary_dimension=local_auxiliary_dimension,
        )
        if self.layout.local_variables_per_scenario != local_structure.n_variables:
            raise ValueError("block-arrow local layout does not match the local CQP variable count")
        self.structure = self._build_structure()
        self._local_scalar_rows = local_structure.n_constraints
        self._local_affine_rows = local_structure.n_affine_constraints
        self._local_variables = local_structure.n_variables

    @property
    def scenario_count(self) -> int:
        return self.tree.scenario_count

    @property
    def nonanticipativity_rows(self) -> int:
        return self.layout.nonanticipativity_rows

    @property
    def scalar_rows_before_nonanticipativity(self) -> int:
        return self.scenario_count * self._local_scalar_rows

    def problem(self, local_values: Sequence[CQPValues]) -> CanonicalCQP:
        return CanonicalCQP(self.structure, self.values(local_values))

    def values(self, local_values: Sequence[CQPValues]) -> CQPValues:
        """Assemble numerical values while preserving the frozen global topology."""

        validated = self._validated_local_values(local_values)
        probabilities = self.tree.probabilities

        quadratic_local = [
            self.local_structure.quadratic.matrix(values.quadratic) * probability
            for probability, values in zip(probabilities, validated, strict=True)
        ]
        quadratic = self._pad_consensus_columns_and_rows(
            sp.block_diag(quadratic_local, format="csc")
        )

        scalar_local = [
            self.local_structure.constraint.matrix(values.constraint) for values in validated
        ]
        scalar = self._pad_consensus_columns(sp.block_diag(scalar_local, format="csc"))
        constraint = sp.vstack(
            (scalar, self.layout.nonanticipativity_operator(format="csc")),
            format="csc",
        )

        if self.local_structure.affine_cone is None:
            affine = sp.csc_matrix((0, self.structure.n_variables))
        else:
            affine_local = [
                self.local_structure.affine_cone.matrix(values.affine_cone) for values in validated
            ]
            affine = self._pad_consensus_columns(sp.block_diag(affine_local, format="csc"))

        linear_parts = [
            probability * values.linear
            for probability, values in zip(probabilities, validated, strict=True)
        ]
        linear_parts.append(np.zeros(self.layout.consensus_dimension, dtype=np.float64))
        linear = np.concatenate(linear_parts)

        lower_parts = [values.lower for values in validated]
        lower_parts.append(np.zeros(self.nonanticipativity_rows, dtype=np.float64))
        lower = np.concatenate(lower_parts)

        upper_parts = [values.upper for values in validated]
        upper_parts.append(np.zeros(self.nonanticipativity_rows, dtype=np.float64))
        upper = np.concatenate(upper_parts)
        affine_offset = np.concatenate([values.affine_offset for values in validated])

        variable_lower_parts = [values.variable_lower for values in validated]
        variable_lower_parts.append(
            np.full(
                self.layout.consensus_dimension,
                -np.inf,
                dtype=np.float64,
            )
        )
        variable_lower = np.concatenate(variable_lower_parts)

        variable_upper_parts = [values.variable_upper for values in validated]
        variable_upper_parts.append(
            np.full(
                self.layout.consensus_dimension,
                np.inf,
                dtype=np.float64,
            )
        )
        variable_upper = np.concatenate(variable_upper_parts)

        return CQPValues(
            quadratic=self.structure.quadratic.values_from(quadratic),
            constraint=self.structure.constraint.values_from(constraint),
            linear=linear,
            lower=lower,
            upper=upper,
            affine_cone=(
                np.empty(0, dtype=np.float64)
                if self.structure.affine_cone is None
                else self.structure.affine_cone.values_from(affine)
            ),
            affine_offset=affine_offset,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
        ).validated(self.structure)

    def decode_primal(self, primal: FloatArray) -> RobustPrimal:
        values = np.asarray(primal, dtype=np.float64)
        if values.shape != (self.structure.n_variables,):
            raise ValueError(f"primal must have shape ({self.structure.n_variables},)")
        local = tuple(
            values[self.layout.scenario_slice(scenario)].copy()
            for scenario in range(self.scenario_count)
        )
        consensus = tuple(
            values[block.variable_slice].copy() for block in self.layout.consensus_blocks
        )
        return RobustPrimal(local=local, consensus=consensus)

    def decode_dual(self, dual: FloatArray) -> RobustDual:
        values = np.asarray(dual, dtype=np.float64)
        if values.shape != (self.structure.n_duals,):
            raise ValueError(f"dual must have shape ({self.structure.n_duals},)")
        scalar_local = tuple(
            values[
                scenario * self._local_scalar_rows : (scenario + 1) * self._local_scalar_rows
            ].copy()
            for scenario in range(self.scenario_count)
        )
        nonanticipativity_start = self.scalar_rows_before_nonanticipativity
        nonanticipativity_stop = nonanticipativity_start + self.nonanticipativity_rows
        affine_start = self.structure.n_constraints
        affine_local = tuple(
            values[
                affine_start + scenario * self._local_affine_rows : affine_start
                + (scenario + 1) * self._local_affine_rows
            ].copy()
            for scenario in range(self.scenario_count)
        )
        return RobustDual(
            local_scalar=scalar_local,
            nonanticipativity=values[nonanticipativity_start:nonanticipativity_stop].copy(),
            local_affine=affine_local,
        )

    def maximum_nonanticipativity_violation(self, primal: FloatArray) -> float:
        return self.layout.nonanticipativity_violation(primal)

    def local_objectives(
        self,
        local_primals: Sequence[FloatArray],
        local_values: Sequence[CQPValues],
    ) -> FloatArray:
        if len(local_primals) != self.scenario_count:
            raise ValueError("one local primal is required per scenario")
        validated = self._validated_local_values(local_values)
        objectives = np.empty(self.scenario_count, dtype=np.float64)
        for scenario, (primal, values) in enumerate(zip(local_primals, validated, strict=True)):
            vector = np.asarray(primal, dtype=np.float64)
            if vector.shape != (self._local_variables,):
                raise ValueError(
                    f"local primal {scenario} must have shape ({self._local_variables},)"
                )
            quadratic = self.local_structure.quadratic.matrix(values.quadratic)
            objectives[scenario] = 0.5 * float(vector @ (quadratic @ vector)) + float(
                values.linear @ vector
            )
        return objectives

    def expected_objective(
        self,
        local_primals: Sequence[FloatArray],
        local_values: Sequence[CQPValues],
    ) -> float:
        return float(self.tree.probabilities @ self.local_objectives(local_primals, local_values))

    def risk_problem(
        self,
        local_values: Sequence[CQPValues],
        measure: Literal["worst", "cvar"],
        *,
        alpha: float | None = None,
    ) -> tuple[CanonicalCQP, RiskEpigraphLayout]:
        """Build a sparse exact worst-case or CVaR quadratic-cost epigraph.

        One rotated SOC represents each scenario's convex quadratic objective.
        The frozen powered-descent transcription has diagonal positive-semidefinite
        Hessians, so this adds O(SN) nonzeros without dense factorisation.
        """

        if measure not in {"worst", "cvar"}:
            raise ValueError("risk measure must be 'worst' or 'cvar'")
        if measure == "cvar" and (alpha is None or not np.isfinite(alpha) or not 0.0 < alpha < 1.0):
            raise ValueError("CVaR alpha must lie strictly between zero and one")
        validated = self._validated_local_values(local_values)
        base = self.problem(validated)
        scenarios = self.scenario_count
        base_variables = self.structure.n_variables
        cost_start = base_variables
        cost_stop = cost_start + scenarios
        if measure == "worst":
            worst_case = cost_stop
            threshold = None
            excess_start = cost_stop + 1
            total_variables = excess_start
        else:
            worst_case = None
            threshold = cost_stop
            excess_start = cost_stop + 1
            total_variables = excess_start + scenarios
        layout = RiskEpigraphLayout(
            base_variables=base_variables,
            scenario_costs=slice(cost_start, cost_stop),
            worst_case=worst_case,
            threshold=threshold,
            excesses=slice(excess_start, total_variables),
            measure=measure,
            alpha=alpha,
        )

        base_constraint = self.structure.constraint.matrix(base.values.constraint)
        padded_constraint = sp.hstack(
            (
                base_constraint,
                sp.csc_matrix((base_constraint.shape[0], total_variables - base_variables)),
            ),
            format="csc",
        )
        risk_rows = sp.lil_matrix((scenarios, total_variables), dtype=np.float64)
        for scenario in range(scenarios):
            risk_rows[scenario, cost_start + scenario] = 1.0
            if measure == "worst":
                assert worst_case is not None
                risk_rows[scenario, worst_case] = -1.0
            else:
                assert threshold is not None
                risk_rows[scenario, threshold] = -1.0
                risk_rows[scenario, excess_start + scenario] = -1.0
        constraint = sp.vstack((padded_constraint, risk_rows.tocsc()), format="csc")
        lower = np.concatenate((base.values.lower, np.full(scenarios, -np.inf, dtype=np.float64)))
        upper = np.concatenate((base.values.upper, np.zeros(scenarios, dtype=np.float64)))

        affine_blocks: list[sp.spmatrix] = []
        affine_offsets: list[FloatArray] = []
        affine_cones = list(self.structure.affine_cones)
        if self.structure.affine_cone is not None:
            base_affine = self.structure.affine_cone.matrix(base.values.affine_cone)
            affine_blocks.append(
                sp.hstack(
                    (
                        base_affine,
                        sp.csc_matrix((base_affine.shape[0], total_variables - base_variables)),
                    ),
                    format="csc",
                )
            )
            affine_offsets.append(base.values.affine_offset)
        affine_cursor = self.structure.n_affine_constraints
        for scenario, values in enumerate(validated):
            quadratic = self.local_structure.quadratic.matrix(values.quadratic)
            diagonal = np.asarray(quadratic.diagonal(), dtype=np.float64)
            off_diagonal = quadratic - sp.diags(diagonal, format="csc")
            if off_diagonal.nnz and np.max(np.abs(off_diagonal.data), initial=0.0) > 1.0e-12:
                raise NotImplementedError(
                    "risk epigraph requires diagonal local Hessians at the frozen commit"
                )
            if np.min(diagonal, initial=0.0) < -1.0e-12:
                raise ValueError("risk epigraph requires positive-semidefinite local Hessians")
            active = np.flatnonzero(diagonal > 1.0e-15)
            if active.size == 0:
                raise NotImplementedError(
                    "risk epigraph requires a positive quadratic objective coefficient"
                )
            cone_rows = int(active.size) + 2
            block = sp.lil_matrix((cone_rows, total_variables), dtype=np.float64)
            local_start = self.layout.scenario_slice(scenario).start
            for row, local_column in enumerate(active):
                block[row, local_start + int(local_column)] = np.sqrt(diagonal[local_column])
            block[active.size, cost_start + scenario] = 1.0
            for local_column in np.flatnonzero(np.abs(values.linear) > 0.0):
                block[active.size, local_start + int(local_column)] = -values.linear[local_column]
            affine_blocks.append(block.tocsc())
            offset = np.zeros(cone_rows, dtype=np.float64)
            offset[-1] = 1.0
            affine_offsets.append(offset)
            affine_cones.append(
                ConeBlock(
                    ConeKind.ROTATED_SECOND_ORDER,
                    affine_cursor,
                    int(active.size),
                )
            )
            affine_cursor += cone_rows
        affine = sp.vstack(affine_blocks, format="csc")

        linear = np.zeros(total_variables, dtype=np.float64)
        if measure == "worst":
            assert worst_case is not None
            linear[worst_case] = 1.0
        else:
            assert threshold is not None and alpha is not None
            linear[threshold] = 1.0
            linear[excess_start:total_variables] = self.tree.probabilities / (1.0 - alpha)
        variable_lower = np.concatenate(
            (
                base.values.variable_lower,
                np.full(scenarios + 1, -np.inf, dtype=np.float64),
                (
                    np.zeros(scenarios, dtype=np.float64)
                    if measure == "cvar"
                    else np.empty(0, dtype=np.float64)
                ),
            )
        )
        variable_upper = np.full(total_variables, np.inf, dtype=np.float64)
        quadratic = sp.csc_matrix((total_variables, total_variables), dtype=np.float64)
        structure = CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(constraint),
            affine_cone=CSCStructure.from_matrix(affine),
            affine_cones=tuple(affine_cones),
            variable_cones=self.structure.variable_cones,
        )
        values = CQPValues(
            quadratic=np.empty(0, dtype=np.float64),
            constraint=structure.constraint.values_from(constraint),
            linear=linear,
            lower=lower,
            upper=upper,
            affine_cone=structure.affine_cone.values_from(affine),
            affine_offset=np.concatenate(affine_offsets),
            variable_lower=variable_lower,
            variable_upper=variable_upper,
        )
        return CanonicalCQP(structure, values), layout

    def _build_structure(self) -> CQPStructure:
        local_q = self.local_structure.quadratic.matrix(
            np.ones(self.local_structure.quadratic.nnz, dtype=np.float64)
        )
        quadratic = self._pad_consensus_columns_and_rows(
            sp.block_diag([local_q] * self.scenario_count, format="csc")
        )

        local_a = self.local_structure.constraint.matrix(
            np.ones(self.local_structure.constraint.nnz, dtype=np.float64)
        )
        scalar = self._pad_consensus_columns(
            sp.block_diag([local_a] * self.scenario_count, format="csc")
        )
        constraint = sp.vstack(
            (scalar, self.layout.nonanticipativity_operator(format="csc")),
            format="csc",
        )

        affine_structure: CSCStructure | None
        affine_cones: tuple[ConeBlock, ...]
        if self.local_structure.affine_cone is None:
            affine_structure = None
            affine_cones = ()
        else:
            local_f = self.local_structure.affine_cone.matrix(
                np.ones(self.local_structure.affine_cone.nnz, dtype=np.float64)
            )
            affine = self._pad_consensus_columns(
                sp.block_diag([local_f] * self.scenario_count, format="csc")
            )
            affine_structure = CSCStructure.from_matrix(affine)
            affine_cones = tuple(
                ConeBlock(
                    kind=cone.kind,
                    start=scenario * self.local_structure.n_affine_constraints + cone.start,
                    vector_dimension=cone.vector_dimension,
                    power_alpha=cone.power_alpha,
                )
                for scenario in range(self.scenario_count)
                for cone in self.local_structure.affine_cones
            )

        variable_cones = tuple(
            ConeBlock(
                kind=cone.kind,
                start=scenario * self.local_structure.n_variables + cone.start,
                vector_dimension=cone.vector_dimension,
                power_alpha=cone.power_alpha,
            )
            for scenario in range(self.scenario_count)
            for cone in self.local_structure.variable_cones
        )
        return CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(constraint),
            affine_cone=affine_structure,
            affine_cones=affine_cones,
            variable_cones=variable_cones,
        )

    def _validated_local_values(
        self,
        local_values: Sequence[CQPValues],
    ) -> tuple[CQPValues, ...]:
        if len(local_values) != self.scenario_count:
            raise ValueError("one local CQP value set is required per scenario")
        return tuple(values.validated(self.local_structure) for values in local_values)

    def _pad_consensus_columns(self, matrix: sp.spmatrix) -> sp.csc_matrix:
        padding = sp.csc_matrix(
            (matrix.shape[0], self.layout.consensus_dimension),
            dtype=np.float64,
        )
        return sp.hstack((matrix, padding), format="csc")

    def _pad_consensus_columns_and_rows(
        self,
        matrix: sp.spmatrix,
    ) -> sp.csc_matrix:
        local_dimension = matrix.shape[0]
        consensus_dimension = self.layout.consensus_dimension
        return sp.bmat(
            [
                [matrix, sp.csc_matrix((local_dimension, consensus_dimension))],
                [
                    sp.csc_matrix((consensus_dimension, local_dimension)),
                    sp.csc_matrix((consensus_dimension, consensus_dimension)),
                ],
            ],
            format="csc",
        )
