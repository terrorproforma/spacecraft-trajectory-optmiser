"""Deterministic block-arrow layouts for scenario-aware spacecraft CQP solves."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .scenario_tree import InformationNode, ScenarioTree


@dataclass(frozen=True, slots=True)
class ConsensusBlock:
    """One shared-control block in the global block-arrow variable vector."""

    node: InformationNode
    offset: int
    dimension: int

    @property
    def variable_slice(self) -> slice:
        return slice(self.offset, self.offset + self.dimension)


@dataclass(frozen=True, slots=True)
class CommunicationProfile:
    """Ring-allreduce traffic for the shared part of one primal or dual update."""

    device_count: int
    shared_dimension: int
    payload_bytes: int
    bytes_per_device: float
    aggregate_bytes: float
    collective_count: int


@dataclass(frozen=True, slots=True)
class ScenarioPartition:
    """Deterministic scenario assignment to the scenario axis of a GPU grid."""

    assignments: tuple[tuple[int, ...], ...]
    loads: tuple[float, ...]

    @property
    def device_count(self) -> int:
        return len(self.assignments)

    @property
    def maximum_load(self) -> float:
        return max(self.loads, default=0.0)

    @property
    def mean_load(self) -> float:
        return float(np.mean(self.loads)) if self.loads else 0.0

    @property
    def imbalance(self) -> float:
        mean = self.mean_load
        return self.maximum_load / mean if mean > 0.0 else 1.0

    def owner(self, scenario_index: int) -> int:
        for device, scenarios in enumerate(self.assignments):
            if scenario_index in scenarios:
                return device
        raise IndexError(f"scenario {scenario_index} has no device owner")


@dataclass(frozen=True, slots=True)
class LogicalGPUGrid:
    """Logical ``G_s x G_t`` grid from the project architecture."""

    scenario_partitions: int
    time_partitions: int

    def __post_init__(self) -> None:
        if self.scenario_partitions <= 0 or self.time_partitions <= 0:
            raise ValueError("logical grid dimensions must be positive")

    @property
    def device_count(self) -> int:
        return self.scenario_partitions * self.time_partitions

    def rank(self, scenario_partition: int, time_partition: int) -> int:
        if not 0 <= scenario_partition < self.scenario_partitions:
            raise IndexError("scenario partition is outside the logical grid")
        if not 0 <= time_partition < self.time_partitions:
            raise IndexError("time partition is outside the logical grid")
        return scenario_partition * self.time_partitions + time_partition


class BlockArrowLayout:
    """Map scenario-local trajectories and shared controls into one global vector.

    Local scenario blocks are contiguous and ordered first. Shared consensus controls
    occupy the arrowhead at the end of the vector. Only information nodes containing
    more than one scenario receive consensus variables; singleton recourse controls
    remain entirely local.
    """

    def __init__(
        self,
        tree: ScenarioTree,
        *,
        state_dimension: int,
        control_dimension: int,
        local_auxiliary_dimension: int = 0,
    ) -> None:
        if state_dimension <= 0 or control_dimension <= 0:
            raise ValueError("state and control dimensions must be positive")
        if local_auxiliary_dimension < 0:
            raise ValueError("local_auxiliary_dimension must be non-negative")
        self.tree = tree
        self.state_dimension = int(state_dimension)
        self.control_dimension = int(control_dimension)
        self.local_auxiliary_dimension = int(local_auxiliary_dimension)
        self.state_variables_per_scenario = (tree.horizon + 1) * self.state_dimension
        self.control_variables_per_scenario = tree.horizon * self.control_dimension
        self.local_variables_per_scenario = (
            self.state_variables_per_scenario
            + self.control_variables_per_scenario
            + self.local_auxiliary_dimension
        )
        self.local_dimension = tree.scenario_count * self.local_variables_per_scenario

        consensus_blocks: list[ConsensusBlock] = []
        offset = self.local_dimension
        for node in tree.shared_nodes:
            consensus_blocks.append(
                ConsensusBlock(
                    node=node,
                    offset=offset,
                    dimension=self.control_dimension,
                )
            )
            offset += self.control_dimension
        self.consensus_blocks = tuple(consensus_blocks)
        self.total_variables = offset
        self._block_by_key = {
            (block.node.stage, block.node.history): block for block in self.consensus_blocks
        }

    @property
    def consensus_dimension(self) -> int:
        return self.total_variables - self.local_dimension

    @property
    def nonanticipativity_rows(self) -> int:
        return sum(
            len(block.node.scenario_indices) * self.control_dimension
            for block in self.consensus_blocks
        )

    def scenario_slice(self, scenario_index: int) -> slice:
        self._validate_scenario(scenario_index)
        start = scenario_index * self.local_variables_per_scenario
        return slice(start, start + self.local_variables_per_scenario)

    def state_slice(self, scenario_index: int, node: int) -> slice:
        self._validate_scenario(scenario_index)
        if not 0 <= node <= self.tree.horizon:
            raise IndexError("state node is outside the trajectory horizon")
        start = scenario_index * self.local_variables_per_scenario + node * self.state_dimension
        return slice(start, start + self.state_dimension)

    def control_slice(self, scenario_index: int, stage: int) -> slice:
        self._validate_scenario(scenario_index)
        if not 0 <= stage < self.tree.horizon:
            raise IndexError("control stage is outside the trajectory horizon")
        start = (
            scenario_index * self.local_variables_per_scenario
            + self.state_variables_per_scenario
            + stage * self.control_dimension
        )
        return slice(start, start + self.control_dimension)

    def auxiliary_slice(self, scenario_index: int) -> slice:
        block = self.scenario_slice(scenario_index)
        start = (
            block.start + self.state_variables_per_scenario + self.control_variables_per_scenario
        )
        return slice(start, block.stop)

    def consensus_slice(self, node: InformationNode) -> slice | None:
        block = self._block_by_key.get((node.stage, node.history))
        return None if block is None else block.variable_slice

    def nonanticipativity_operator(self, *, format: str = "csc") -> sp.spmatrix:
        """Build ``N`` for ``u_(s,k) - u_bar_node = 0`` with immutable ordering."""

        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        row = 0
        for block in self.consensus_blocks:
            for scenario_index in block.node.scenario_indices:
                local = self.control_slice(scenario_index, block.node.stage)
                shared = block.variable_slice
                for component in range(self.control_dimension):
                    rows.extend((row, row))
                    columns.extend((local.start + component, shared.start + component))
                    data.extend((1.0, -1.0))
                    row += 1
        operator = sp.coo_matrix(
            (data, (rows, columns)),
            shape=(self.nonanticipativity_rows, self.total_variables),
            dtype=np.float64,
        )
        try:
            return operator.asformat(format)
        except ValueError as exc:
            raise ValueError(f"unsupported sparse format {format!r}") from exc

    def nonanticipativity_violation(self, vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.total_variables,):
            raise ValueError(f"global vector must have shape ({self.total_variables},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("global vector must be finite")
        residual = self.nonanticipativity_operator() @ values
        return float(np.max(np.abs(residual), initial=0.0))

    def communication_profile(
        self,
        device_count: int,
        *,
        scalar_bytes: int = 8,
        collective_count: int = 1,
    ) -> CommunicationProfile:
        """Estimate ring-allreduce traffic for shared controls or residuals."""

        if device_count <= 0:
            raise ValueError("device_count must be positive")
        if scalar_bytes <= 0 or collective_count < 0:
            raise ValueError("scalar_bytes must be positive and collective_count non-negative")
        payload_bytes = self.consensus_dimension * scalar_bytes
        if device_count == 1 or payload_bytes == 0 or collective_count == 0:
            per_device = 0.0
            aggregate = 0.0
        else:
            per_collective = 2.0 * (device_count - 1) / device_count * payload_bytes
            per_device = collective_count * per_collective
            aggregate = device_count * per_device
        return CommunicationProfile(
            device_count=device_count,
            shared_dimension=self.consensus_dimension,
            payload_bytes=payload_bytes,
            bytes_per_device=per_device,
            aggregate_bytes=aggregate,
            collective_count=collective_count,
        )

    def statistics(self) -> dict[str, int]:
        operator = self.nonanticipativity_operator()
        return {
            "scenarios": self.tree.scenario_count,
            "stages": self.tree.horizon,
            "local_variables_per_scenario": self.local_variables_per_scenario,
            "local_dimension": self.local_dimension,
            "consensus_blocks": len(self.consensus_blocks),
            "consensus_dimension": self.consensus_dimension,
            "total_variables": self.total_variables,
            "nonanticipativity_rows": operator.shape[0],
            "nonanticipativity_nonzeros": operator.nnz,
        }

    def _validate_scenario(self, scenario_index: int) -> None:
        if not 0 <= scenario_index < self.tree.scenario_count:
            raise IndexError("scenario index is outside the scenario tree")


def partition_scenarios(
    weights: Iterable[float],
    device_count: int,
) -> ScenarioPartition:
    """Deterministic longest-processing-time partition for scenario-local work."""

    weight_vector = np.asarray(tuple(weights), dtype=np.float64)
    if weight_vector.ndim != 1 or weight_vector.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(weight_vector)) or np.any(weight_vector < 0.0):
        raise ValueError("scenario weights must be finite and non-negative")
    if device_count <= 0:
        raise ValueError("device_count must be positive")

    assignments: list[list[int]] = [[] for _ in range(device_count)]
    loads = np.zeros(device_count, dtype=np.float64)
    order = sorted(
        range(weight_vector.size),
        key=lambda index: (-weight_vector[index], index),
    )
    for scenario_index in order:
        device = min(
            range(device_count),
            key=lambda candidate: (loads[candidate], candidate),
        )
        assignments[device].append(scenario_index)
        loads[device] += weight_vector[scenario_index]
    return ScenarioPartition(
        assignments=tuple(tuple(sorted(items)) for items in assignments),
        loads=tuple(float(load) for load in loads),
    )
