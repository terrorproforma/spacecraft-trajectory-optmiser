"""Shared-column condensation for scenario-aware conic quadratic programmes.

The consensus-row formulation keeps a full local control vector per scenario and adds
``u_s - u_bar = 0`` rows.  That is convenient for distributed first-order methods but
can be poorly conditioned for an interior-point correctness oracle.  This module
constructs the equivalent block-arrow form in which each shared control appears only
once as an arrowhead column.  Scenario-local states, recourse controls, and auxiliary
variables remain contiguous.
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

from .scenario_tree import InformationNode, ScenarioTree

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CondensedConsensusBlock:
    """One information-node control stored once in the global arrowhead."""

    node: InformationNode
    offset: int
    dimension: int

    @property
    def variable_slice(self) -> slice:
        return slice(self.offset, self.offset + self.dimension)


@dataclass(frozen=True, slots=True)
class CondensedPrimal:
    """Decoded full local vectors and shared information-node controls."""

    local: tuple[FloatArray, ...]
    consensus: tuple[FloatArray, ...]


@dataclass(frozen=True, slots=True)
class CondensedDual:
    """Scenario-local scalar and affine-cone dual blocks."""

    local_scalar: tuple[FloatArray, ...]
    local_affine: tuple[FloatArray, ...]


class CondensedScenarioCQPBundle:
    """Eliminate duplicated shared controls from a same-topology scenario bundle.

    Local variables are assumed to have the ordering

    ``[all states, all controls, local auxiliaries]``.

    This is the ordering used by the committed CW and powered-descent transcriptions.
    A future generic permutation descriptor can relax the assumption without changing
    the resulting canonical CQP contract.
    """

    def __init__(
        self,
        tree: ScenarioTree,
        local_structure: CQPStructure,
        *,
        state_dimension: int,
        control_dimension: int,
        local_auxiliary_dimension: int,
    ) -> None:
        if state_dimension <= 0 or control_dimension <= 0:
            raise ValueError("state and control dimensions must be positive")
        if local_auxiliary_dimension < 0:
            raise ValueError("local_auxiliary_dimension must be non-negative")
        self.tree = tree
        self.local_structure = local_structure
        self.state_dimension = int(state_dimension)
        self.control_dimension = int(control_dimension)
        self.local_auxiliary_dimension = int(local_auxiliary_dimension)
        self.state_count = (tree.horizon + 1) * self.state_dimension
        self.control_count = tree.horizon * self.control_dimension
        expected = self.state_count + self.control_count + self.local_auxiliary_dimension
        if expected != local_structure.n_variables:
            raise ValueError(
                "state/control/auxiliary layout does not match local variable count"
            )

        self._nodes_by_scenario_stage = self._node_lookup()
        self._scenario_slices, local_dimension = self._scenario_local_slices()
        self.local_dimension = local_dimension
        blocks: list[CondensedConsensusBlock] = []
        offset = local_dimension
        for node in tree.shared_nodes:
            blocks.append(
                CondensedConsensusBlock(
                    node=node,
                    offset=offset,
                    dimension=self.control_dimension,
                )
            )
            offset += self.control_dimension
        self.consensus_blocks = tuple(blocks)
        self.total_variables = offset
        self._block_by_key = {
            (block.node.stage, block.node.history): block
            for block in self.consensus_blocks
        }
        self._local_to_global = tuple(
            self._make_local_to_global(scenario)
            for scenario in range(tree.scenario_count)
        )
        self.structure = self._build_structure()

    @property
    def scenario_count(self) -> int:
        return self.tree.scenario_count

    @property
    def consensus_dimension(self) -> int:
        return self.total_variables - self.local_dimension

    def scenario_slice(self, scenario: int) -> slice:
        self._validate_scenario(scenario)
        return self._scenario_slices[scenario]

    def local_to_global(self, scenario: int) -> IntArray:
        self._validate_scenario(scenario)
        return self._local_to_global[scenario].copy()

    def consensus_slice(self, node: InformationNode) -> slice | None:
        block = self._block_by_key.get((node.stage, node.history))
        return None if block is None else block.variable_slice

    def problem(self, local_values: Sequence[CQPValues]) -> CanonicalCQP:
        return CanonicalCQP(self.structure, self.values(local_values))

    def values(self, local_values: Sequence[CQPValues]) -> CQPValues:
        validated = self._validated_local_values(local_values)
        quadratic, constraint, affine = self._assemble_matrices(validated, symbolic=False)
        linear = np.zeros(self.total_variables, dtype=np.float64)
        variable_lower = np.full(self.total_variables, -np.inf, dtype=np.float64)
        variable_upper = np.full(self.total_variables, np.inf, dtype=np.float64)
        probabilities = self.tree.probabilities

        for probability, mapping, values in zip(
            probabilities,
            self._local_to_global,
            validated,
            strict=True,
        ):
            np.add.at(linear, mapping, probability * values.linear)
            np.maximum.at(variable_lower, mapping, values.variable_lower)
            np.minimum.at(variable_upper, mapping, values.variable_upper)
        if np.any(variable_lower > variable_upper):
            raise ValueError("scenario variable-bound intersection is empty")

        return CQPValues(
            quadratic=self.structure.quadratic.values_from(quadratic),
            constraint=self.structure.constraint.values_from(constraint),
            linear=linear,
            lower=np.concatenate([values.lower for values in validated]),
            upper=np.concatenate([values.upper for values in validated]),
            affine_cone=(
                np.empty(0, dtype=np.float64)
                if self.structure.affine_cone is None
                else self.structure.affine_cone.values_from(affine)
            ),
            affine_offset=np.concatenate(
                [values.affine_offset for values in validated]
            ),
            variable_lower=variable_lower,
            variable_upper=variable_upper,
        ).validated(self.structure)

    def decode_primal(self, primal: FloatArray) -> CondensedPrimal:
        vector = np.asarray(primal, dtype=np.float64)
        if vector.shape != (self.total_variables,):
            raise ValueError(f"primal must have shape ({self.total_variables},)")
        local = tuple(vector[mapping].copy() for mapping in self._local_to_global)
        consensus = tuple(
            vector[block.variable_slice].copy() for block in self.consensus_blocks
        )
        return CondensedPrimal(local=local, consensus=consensus)

    def decode_dual(self, dual: FloatArray) -> CondensedDual:
        vector = np.asarray(dual, dtype=np.float64)
        if vector.shape != (self.structure.n_duals,):
            raise ValueError(f"dual must have shape ({self.structure.n_duals},)")
        scalar_rows = self.local_structure.n_constraints
        affine_rows = self.local_structure.n_affine_constraints
        local_scalar = tuple(
            vector[scenario * scalar_rows : (scenario + 1) * scalar_rows].copy()
            for scenario in range(self.scenario_count)
        )
        affine_start = self.structure.n_constraints
        local_affine = tuple(
            vector[
                affine_start + scenario * affine_rows :
                affine_start + (scenario + 1) * affine_rows
            ].copy()
            for scenario in range(self.scenario_count)
        )
        return CondensedDual(
            local_scalar=local_scalar,
            local_affine=local_affine,
        )

    def maximum_nonanticipativity_violation(self, primal: FloatArray) -> float:
        decoded = self.decode_primal(primal)
        maximum = 0.0
        for block, consensus in zip(
            self.consensus_blocks,
            decoded.consensus,
            strict=True,
        ):
            local_control = self._control_local_slice(block.node.stage)
            for scenario in block.node.scenario_indices:
                maximum = max(
                    maximum,
                    float(
                        np.max(
                            np.abs(decoded.local[scenario][local_control] - consensus),
                            initial=0.0,
                        )
                    ),
                )
        return maximum

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
            if vector.shape != (self.local_structure.n_variables,):
                raise ValueError(
                    "local primal "
                    f"{scenario} must have shape ({self.local_structure.n_variables},)"
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

    def statistics(self) -> dict[str, int]:
        return {
            "scenarios": self.scenario_count,
            "stages": self.tree.horizon,
            "local_dimension": self.local_dimension,
            "consensus_blocks": len(self.consensus_blocks),
            "consensus_dimension": self.consensus_dimension,
            "total_variables": self.total_variables,
            "scalar_rows": self.structure.n_constraints,
            "affine_rows": self.structure.n_affine_constraints,
        }

    def _build_structure(self) -> CQPStructure:
        symbolic_values = CQPValues(
            quadratic=np.ones(self.local_structure.quadratic.nnz),
            constraint=np.ones(self.local_structure.constraint.nnz),
            linear=np.zeros(self.local_structure.n_variables),
            lower=np.zeros(self.local_structure.n_constraints),
            upper=np.zeros(self.local_structure.n_constraints),
            affine_cone=np.ones(
                0
                if self.local_structure.affine_cone is None
                else self.local_structure.affine_cone.nnz
            ),
            affine_offset=np.zeros(self.local_structure.n_affine_constraints),
            variable_lower=np.full(self.local_structure.n_variables, -np.inf),
            variable_upper=np.full(self.local_structure.n_variables, np.inf),
        ).validated(self.local_structure)
        symbolic = [symbolic_values] * self.scenario_count
        quadratic, constraint, affine = self._assemble_matrices(symbolic, symbolic=True)
        return CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(constraint),
            affine_cone=(
                None
                if self.local_structure.affine_cone is None
                else CSCStructure.from_matrix(affine)
            ),
            affine_cones=self._affine_cones(),
            variable_cones=self._variable_cones(),
        )

    def _assemble_matrices(
        self,
        local_values: Sequence[CQPValues],
        *,
        symbolic: bool,
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
        quadratic_rows: list[IntArray] = []
        quadratic_columns: list[IntArray] = []
        quadratic_data: list[FloatArray] = []
        constraint_rows: list[IntArray] = []
        constraint_columns: list[IntArray] = []
        constraint_data: list[FloatArray] = []
        affine_rows: list[IntArray] = []
        affine_columns: list[IntArray] = []
        affine_data: list[FloatArray] = []
        probabilities = self.tree.probabilities
        scalar_row_offset = 0
        affine_row_offset = 0

        for probability, mapping, values in zip(
            probabilities,
            self._local_to_global,
            local_values,
            strict=True,
        ):
            local_q = self.local_structure.quadratic.matrix(values.quadratic).tocoo()
            quadratic_rows.append(mapping[local_q.row])
            quadratic_columns.append(mapping[local_q.col])
            quadratic_data.append(
                np.ones(local_q.nnz, dtype=np.float64)
                if symbolic
                else probability * local_q.data
            )

            local_a = self.local_structure.constraint.matrix(values.constraint).tocoo()
            constraint_rows.append(local_a.row + scalar_row_offset)
            constraint_columns.append(mapping[local_a.col])
            constraint_data.append(
                np.ones(local_a.nnz, dtype=np.float64)
                if symbolic
                else local_a.data
            )
            scalar_row_offset += self.local_structure.n_constraints

            if self.local_structure.affine_cone is not None:
                local_f = self.local_structure.affine_cone.matrix(
                    values.affine_cone
                ).tocoo()
                affine_rows.append(local_f.row + affine_row_offset)
                affine_columns.append(mapping[local_f.col])
                affine_data.append(
                    np.ones(local_f.nnz, dtype=np.float64)
                    if symbolic
                    else local_f.data
                )
                affine_row_offset += self.local_structure.n_affine_constraints

        quadratic = self._coo(
            quadratic_rows,
            quadratic_columns,
            quadratic_data,
            shape=(self.total_variables, self.total_variables),
        )
        constraint = self._coo(
            constraint_rows,
            constraint_columns,
            constraint_data,
            shape=(
                self.scenario_count * self.local_structure.n_constraints,
                self.total_variables,
            ),
        )
        affine = self._coo(
            affine_rows,
            affine_columns,
            affine_data,
            shape=(
                self.scenario_count * self.local_structure.n_affine_constraints,
                self.total_variables,
            ),
        )
        return quadratic, constraint, affine

    @staticmethod
    def _coo(
        row_parts: Sequence[IntArray],
        column_parts: Sequence[IntArray],
        data_parts: Sequence[FloatArray],
        *,
        shape: tuple[int, int],
    ) -> sp.csc_matrix:
        if not data_parts:
            return sp.csc_matrix(shape, dtype=np.float64)
        matrix = sp.coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=shape,
            dtype=np.float64,
        ).tocsc()
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def _affine_cones(self) -> tuple[ConeBlock, ...]:
        return tuple(
            ConeBlock(
                kind=cone.kind,
                start=scenario * self.local_structure.n_affine_constraints + cone.start,
                vector_dimension=cone.vector_dimension,
                power_alpha=cone.power_alpha,
            )
            for scenario in range(self.scenario_count)
            for cone in self.local_structure.affine_cones
        )

    def _variable_cones(self) -> tuple[ConeBlock, ...]:
        blocks: dict[tuple[object, int, int, float], ConeBlock] = {}
        for _scenario, mapping in enumerate(self._local_to_global):
            for cone in self.local_structure.variable_cones:
                local_indices = np.arange(cone.start, cone.stop, dtype=np.int64)
                mapped = mapping[local_indices]
                if mapped.size > 1 and not np.all(np.diff(mapped) == 1):
                    raise NotImplementedError(
                        "condensation split a variable cone across noncontiguous columns"
                    )
                block = ConeBlock(
                    kind=cone.kind,
                    start=int(mapped[0]),
                    vector_dimension=cone.vector_dimension,
                    power_alpha=cone.power_alpha,
                )
                key = (
                    block.kind,
                    block.start,
                    block.vector_dimension,
                    block.power_alpha,
                )
                blocks[key] = block
        return tuple(sorted(blocks.values(), key=lambda block: block.start))

    def _node_lookup(self) -> tuple[tuple[InformationNode, ...], ...]:
        table: list[list[InformationNode | None]] = [
            [None] * self.tree.horizon for _ in range(self.scenario_count)
        ]
        for node in self.tree.nodes:
            for scenario in node.scenario_indices:
                table[scenario][node.stage] = node
        if any(node is None for row in table for node in row):
            raise AssertionError("scenario tree did not cover every scenario-stage pair")
        return tuple(
            tuple(node for node in row if node is not None) for row in table
        )

    def _scenario_local_slices(self) -> tuple[tuple[slice, ...], int]:
        slices: list[slice] = []
        offset = 0
        for scenario in range(self.scenario_count):
            local_stages = sum(
                not node.shared for node in self._nodes_by_scenario_stage[scenario]
            )
            size = (
                self.state_count
                + local_stages * self.control_dimension
                + self.local_auxiliary_dimension
            )
            slices.append(slice(offset, offset + size))
            offset += size
        return tuple(slices), offset

    def _make_local_to_global(self, scenario: int) -> IntArray:
        block = self._scenario_slices[scenario]
        mapping = np.empty(self.local_structure.n_variables, dtype=np.int64)
        mapping[: self.state_count] = np.arange(
            block.start,
            block.start + self.state_count,
            dtype=np.int64,
        )
        local_control_offset = block.start + self.state_count
        local_control_count = 0
        for stage, node in enumerate(self._nodes_by_scenario_stage[scenario]):
            local = self._control_local_slice(stage)
            if node.shared:
                shared = self._block_by_key[(node.stage, node.history)].variable_slice
                mapping[local] = np.arange(shared.start, shared.stop, dtype=np.int64)
            else:
                start = local_control_offset + local_control_count * self.control_dimension
                mapping[local] = np.arange(
                    start,
                    start + self.control_dimension,
                    dtype=np.int64,
                )
                local_control_count += 1
        auxiliary_local_start = self.state_count + self.control_count
        auxiliary_global_start = (
            local_control_offset + local_control_count * self.control_dimension
        )
        mapping[auxiliary_local_start:] = np.arange(
            auxiliary_global_start,
            block.stop,
            dtype=np.int64,
        )
        mapping.flags.writeable = False
        return mapping

    def _control_local_slice(self, stage: int) -> slice:
        start = self.state_count + stage * self.control_dimension
        return slice(start, start + self.control_dimension)

    def _validated_local_values(
        self,
        local_values: Sequence[CQPValues],
    ) -> tuple[CQPValues, ...]:
        if len(local_values) != self.scenario_count:
            raise ValueError("one local CQP value set is required per scenario")
        return tuple(
            values.validated(self.local_structure) for values in local_values
        )

    def _validate_scenario(self, scenario: int) -> None:
        if not 0 <= scenario < self.scenario_count:
            raise IndexError("scenario index is outside the scenario tree")
