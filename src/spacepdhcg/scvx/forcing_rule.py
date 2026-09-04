"""Adaptive inner-accuracy requests for inexact successive convexification.

The forcing rule follows the project brief directly. It is solver independent: a
backend receives only a requested tolerance and iteration budget, while the outer
loop records why that request was made.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class SolvePhase(StrEnum):
    """Accuracy regimes used by the hybrid SCvx programme."""

    EXPLORATION = "exploration"
    CONVERGENCE = "convergence"
    POLISH = "polish"


@dataclass(frozen=True, slots=True)
class OuterResidual:
    """Scaled nonlinear residual components used by the forcing rule."""

    dynamics: float
    path: float
    terminal: float
    step: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.dynamics, self.path, self.terminal, self.step],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("outer residual components must be finite and non-negative")

    @property
    def feasibility(self) -> float:
        return max(self.dynamics, self.path, self.terminal)

    @property
    def maximum(self) -> float:
        return max(self.feasibility, self.step)


@dataclass(frozen=True, slots=True)
class ForcingRuleConfig:
    """Parameters for the practical and theoretical forcing schedules."""

    epsilon_max: float = 1.0e-3
    epsilon_floor: float = 1.0e-8
    epsilon_0: float = 1.0e-3
    coefficient: float = 0.2
    alpha: float = 0.5
    gamma: float = 0.6
    exploration_iterations: int = 2
    switch_residual: float = 2.0e-3
    switch_step: float = 5.0e-2
    good_agreement: float = 0.75
    polish_tolerance: float = 1.0e-9
    exploration_iteration_limit: int = 250
    convergence_iteration_limit: int = 750
    polish_iteration_limit: int = 2_000
    resolve_factor: float = 0.1
    resolve_kkt_multiple: float = 5.0
    theoretical: bool = False

    def __post_init__(self) -> None:
        positive = (
            "epsilon_max",
            "epsilon_0",
            "coefficient",
            "alpha",
            "switch_residual",
            "switch_step",
            "polish_tolerance",
            "resolve_kkt_multiple",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.epsilon_floor) or self.epsilon_floor < 0.0:
            raise ValueError("epsilon_floor must be finite and non-negative")
        if self.epsilon_floor > self.epsilon_max:
            raise ValueError("epsilon_floor may not exceed epsilon_max")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if not 0.0 < self.resolve_factor < 1.0:
            raise ValueError("resolve_factor must lie strictly between zero and one")
        if not 0.0 <= self.good_agreement <= 1.0:
            raise ValueError("good_agreement must lie between zero and one")
        if self.exploration_iterations < 0:
            raise ValueError("exploration_iterations must be non-negative")
        for name in (
            "exploration_iteration_limit",
            "convergence_iteration_limit",
            "polish_iteration_limit",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ForcingDecision:
    """One auditable inner-solve request."""

    tolerance: float
    raw_target: float
    iteration_limit: int
    phase: SolvePhase
    reason: str


class AdaptiveForcingRule:
    """Compute inexact CQP tolerances from the current nonlinear residual."""

    def __init__(self, config: ForcingRuleConfig | None = None) -> None:
        self.config = config or ForcingRuleConfig()

    def request(
        self,
        iteration: int,
        residual: OuterResidual,
        *,
        accepted_streak: int = 0,
        agreement: float | None = None,
    ) -> ForcingDecision:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if accepted_streak < 0:
            raise ValueError("accepted_streak must be non-negative")
        if agreement is not None and not np.isfinite(agreement):
            agreement = None

        maximum = residual.maximum
        residual_target = self.config.coefficient * maximum ** (1.0 + self.config.alpha)
        geometric_target = self.config.epsilon_0 * self.config.gamma**iteration
        raw_target = min(
            self.config.epsilon_max,
            residual_target,
            geometric_target,
        )
        if maximum == 0.0:
            raw_target = min(raw_target, self.config.polish_tolerance)

        polish_ready = (
            accepted_streak >= 2
            and maximum < self.config.switch_residual
            and residual.step < self.config.switch_step
            and agreement is not None
            and agreement > self.config.good_agreement
        )
        if polish_ready:
            return ForcingDecision(
                tolerance=self.config.polish_tolerance,
                raw_target=raw_target,
                iteration_limit=self.config.polish_iteration_limit,
                phase=SolvePhase.POLISH,
                reason="accepted streak, residual, step and agreement passed the polish gate",
            )

        floor = 0.0 if self.config.theoretical else self.config.epsilon_floor
        tolerance = max(floor, raw_target)
        if iteration < self.config.exploration_iterations:
            exploration_target = min(self.config.epsilon_max, geometric_target)
            tolerance = max(tolerance, exploration_target)
            phase = SolvePhase.EXPLORATION
            iteration_limit = self.config.exploration_iteration_limit
            reason = "early outer iteration: retain a coarse first-order solve"
        else:
            phase = SolvePhase.CONVERGENCE
            iteration_limit = self.config.convergence_iteration_limit
            reason = "tighten according to residual and geometric forcing terms"
        return ForcingDecision(
            tolerance=tolerance,
            raw_target=raw_target,
            iteration_limit=iteration_limit,
            phase=phase,
            reason=reason,
        )

    def should_resolve(
        self,
        *,
        accepted: bool,
        primal_residual: float,
        dual_residual: float,
        requested_tolerance: float,
    ) -> bool:
        """Return whether solver error may explain a rejected candidate."""

        if accepted:
            return False
        values = np.asarray(
            [primal_residual, dual_residual, requested_tolerance],
            dtype=np.float64,
        )
        if np.any(np.isnan(values)) or requested_tolerance <= 0.0:
            return False
        kkt = max(abs(primal_residual), abs(dual_residual))
        return kkt > self.config.resolve_kkt_multiple * requested_tolerance

    def refined_tolerance(self, requested_tolerance: float) -> float:
        if not np.isfinite(requested_tolerance) or requested_tolerance <= 0.0:
            raise ValueError("requested_tolerance must be finite and positive")
        return max(
            self.config.polish_tolerance,
            self.config.resolve_factor * requested_tolerance,
        )
