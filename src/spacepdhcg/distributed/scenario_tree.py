"""Scenario trees and information-history non-anticipativity groups.

Controls selected at a stage may differ only after the corresponding scenarios have
different information histories. The representation is deliberately deterministic so
its node order can be reused by a persistent sparse CQP workspace and by distributed
checkpoint/restart code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Scenario:
    """One uncertainty realisation and its stage-wise information history."""

    name: str
    probability: float
    information_history: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name may not be empty")
        if not np.isfinite(self.probability) or self.probability <= 0.0:
            raise ValueError("scenario probability must be finite and positive")
        if not self.information_history:
            raise ValueError("information history must contain at least one stage")
        if any(not label for label in self.information_history):
            raise ValueError("information-history labels may not be empty")


@dataclass(frozen=True, slots=True)
class InformationNode:
    """A deterministic control node shared by scenarios with the same history."""

    stage: int
    history: tuple[str, ...]
    scenario_indices: tuple[int, ...]

    @property
    def shared(self) -> bool:
        return len(self.scenario_indices) > 1

    @property
    def key(self) -> str:
        return f"stage={self.stage}|history={'/'.join(self.history)}"


class ScenarioTree:
    """Finite scenario tree defined by exact information-history prefixes."""

    def __init__(
        self,
        scenarios: Iterable[Scenario],
        *,
        probability_tolerance: float = 1.0e-12,
    ) -> None:
        self.scenarios = tuple(scenarios)
        if not self.scenarios:
            raise ValueError("a scenario tree requires at least one scenario")
        if not np.isfinite(probability_tolerance) or probability_tolerance < 0.0:
            raise ValueError("probability_tolerance must be finite and non-negative")
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique")
        horizons = {len(scenario.information_history) for scenario in self.scenarios}
        if len(horizons) != 1:
            raise ValueError("all scenarios must have the same information horizon")
        total_probability = float(sum(scenario.probability for scenario in self.scenarios))
        if abs(total_probability - 1.0) > probability_tolerance:
            raise ValueError("scenario probabilities must sum to one within probability_tolerance")
        self._nodes = tuple(
            node for stage in range(self.horizon) for node in self._nodes_at_stage(stage)
        )

    @property
    def horizon(self) -> int:
        return len(self.scenarios[0].information_history)

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def probabilities(self) -> np.ndarray:
        return np.asarray(
            [scenario.probability for scenario in self.scenarios],
            dtype=np.float64,
        )

    @property
    def nodes(self) -> tuple[InformationNode, ...]:
        return self._nodes

    @property
    def shared_nodes(self) -> tuple[InformationNode, ...]:
        return tuple(node for node in self._nodes if node.shared)

    def nodes_at_stage(self, stage: int) -> tuple[InformationNode, ...]:
        self._validate_stage(stage)
        return tuple(node for node in self._nodes if node.stage == stage)

    def shared_nodes_at_stage(self, stage: int) -> tuple[InformationNode, ...]:
        return tuple(node for node in self.nodes_at_stage(stage) if node.shared)

    def control_groups(self, stage: int) -> tuple[tuple[int, ...], ...]:
        """Return all classes for a stage, including singleton recourse nodes."""

        return tuple(node.scenario_indices for node in self.nodes_at_stage(stage))

    def anchor_edges(self) -> tuple[tuple[int, int, int], ...]:
        """Minimal pairwise non-anticipativity edges ``(stage, anchor, scenario)``."""

        edges: list[tuple[int, int, int]] = []
        for node in self.shared_nodes:
            anchor = node.scenario_indices[0]
            edges.extend((node.stage, anchor, scenario) for scenario in node.scenario_indices[1:])
        return tuple(edges)

    def _nodes_at_stage(self, stage: int) -> tuple[InformationNode, ...]:
        groups: dict[tuple[str, ...], list[int]] = {}
        for index, scenario in enumerate(self.scenarios):
            history = scenario.information_history[: stage + 1]
            groups.setdefault(history, []).append(index)
        ordered = sorted(groups.items(), key=lambda item: (item[1][0], item[0]))
        return tuple(
            InformationNode(
                stage=stage,
                history=history,
                scenario_indices=tuple(indices),
            )
            for history, indices in ordered
        )

    def _validate_stage(self, stage: int) -> None:
        if not 0 <= stage < self.horizon:
            raise IndexError(f"stage {stage} is outside [0, {self.horizon})")

    @classmethod
    def common_open_loop(
        cls,
        scenario_count: int,
        horizon: int,
        *,
        common_prefix: int | None = None,
        probabilities: Iterable[float] | None = None,
    ) -> ScenarioTree:
        """Construct a shared prefix followed by scenario-local recourse.

        ``common_prefix=None`` means every control stage is shared. A value of zero
        creates scenario-local controls from the first stage.
        """

        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        prefix = horizon if common_prefix is None else int(common_prefix)
        if not 0 <= prefix <= horizon:
            raise ValueError("common_prefix must lie within the control horizon")
        if probabilities is None:
            probability_vector = np.full(scenario_count, 1.0 / scenario_count)
        else:
            probability_vector = np.asarray(tuple(probabilities), dtype=np.float64)
            if probability_vector.shape != (scenario_count,):
                raise ValueError("probabilities must have one entry per scenario")

        scenarios = []
        for scenario_index in range(scenario_count):
            history = tuple(
                "open-loop" if stage < prefix else f"scenario-{scenario_index}/recourse-{stage}"
                for stage in range(horizon)
            )
            scenarios.append(
                Scenario(
                    name=f"scenario-{scenario_index}",
                    probability=float(probability_vector[scenario_index]),
                    information_history=history,
                )
            )
        return cls(scenarios)
