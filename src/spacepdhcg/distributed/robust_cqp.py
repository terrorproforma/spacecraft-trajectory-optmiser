"""Monolithic reference assembly for scenario-aware conic quadratic programs.

The production multi-GPU path will never materialise this matrix on one host. This
module is the correctness oracle: it embeds identical fixed local structures, weights
scenario objectives by probability, appends exact non-anticipativity rows, and exposes
stable slices for comparing distributed implementations against the monolithic solve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from spacepdhcg.cqp import (
    CanonicalCQP,
    ConeBlock,
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
            raise ValueError(
                "block-arrow local layout does not match the local CQP variable count"
            )
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
            self.local_structure.constraint.matrix(values.constraint)
            for values in validated
        ]
        scalar = self._pad_consensus_columns(
            sp.block_diag(scalar_local, format="csc")
        )
        constraint = sp.vstack(
            (scalar, self.layout.nonanticipativity_operator(format="csc")),
            format="csc",
        )

        if self.local_structure.affine_cone is None:
            affine = sp.csc_matrix((0, self.structure.n_variables))
        else:
            affine_local = [
                self.local_structure.affine_cone.matrix(values.affine_cone)
                for values in validated
            ]
            affine = self._pad_consensus_columns(
                sp.block_diag(affine_local, format="csc")
            )

        linear_parts = [
            probability * values.linear
            for probability, values in zip(probabilities, validated, strict=True)
        ]
        linear_parts.append(
            np.zeros(self.layout.consensus_dimension, dtype=np.float64)
        )
        linear = np.concatenate(linear_parts)

        lower_parts = [values.lower for values in validated]
        lower_parts.append(
            np.zeros(self.nonanticipativity_rows, dtype=np.float64)
        )
        lower = np.concatenate(lower_parts)

        upper_parts = [values.upper for values in validated]
        upper_parts.append(
            np.zeros(self.nonanticipativity_rows, dtype=np.float64)
        )
        upper = np.concatenate(upper_parts)
        affine_offset = np.concatenate(
            [values.affine_offset for values in validated]
        )

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
            raise ValueError(
                f"primal must have shape ({self.structure.n_variables},)"
            )
        local = tuple(
            values[self.layout.scenario_slice(scenario)].copy()
            for scenario in range(self.scenario_count)
        )
        consensus = tuple(
            values[block.variable_slice].copy()
            for block in self.layout.consensus_blocks
        )
        return RobustPrimal(local=local, consensus=consensus)

    def decode_dual(self, dual: FloatArray) -> RobustDual:
        values = np.asarray(dual, dtype=np.float64)
        if values.shape != (self.structure.n_duals,):
            raise ValueError(f"dual must have shape ({self.structure.n_duals},)")
        scalar_local = tuple(
            values[
                scenario * self._local_scalar_rows :
                (scenario + 1) * self._local_scalar_rows
            ].copy()
            for scenario in range(self.scenario_count)
        )
        nonanticipativity_start = self.scalar_rows_before_nonanticipativity
        nonanticipativity_stop = (
            nonanticipativity_start + self.nonanticipativity_rows
        )
        affine_start = self.structure.n_constraints
        affine_local = tuple(
            values[
                affine_start + scenario * self._local_affine_rows :
                affine_start + (scenario + 1) * self._local_affine_rows
            ].copy()
            for scenario in range(self.scenario_count)
        )
        return RobustDual(
            local_scalar=scalar_local,
            nonanticipativity=values[
                nonanticipativity_start:nonanticipativity_stop
            ].copy(),
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
        for scenario, (primal, values) in enumerate(
            zip(local_primals, validated, strict=True)
        ):
            vector = np.asarray(primal, dtype=np.float64)
            if vector.shape != (self._local_variables,):
                raise ValueError(
                    f"local primal {scenario} must have shape ({self._local_variables},)"
                )
            quadratic = self.local_structure.quadratic.matrix(values.quadratic)
            objectives[scenario] = (
                0.5 * float(vector @ (quadratic @ vector))
                + float(values.linear @ vector)
            )
        return objectives

    def expected_objective(
        self,
        local_primals: Sequence[FloatArray],
        local_values: Sequence[CQPValues],
    ) -> float:
        return float(
            self.tree.probabilities @ self.local_objectives(local_primals, local_values)
        )

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
        return tuple(
            values.validated(self.local_structure) for values in local_values
        )

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
