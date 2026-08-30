"""Trust-region state machine for successive convexification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class RadiusAction(StrEnum):
    SHRINK = "shrink"
    KEEP = "keep"
    GROW = "grow"


@dataclass(frozen=True, slots=True)
class TrustRegionConfig:
    initial_radius: float = 1.0
    minimum_radius: float = 1.0e-4
    maximum_radius: float = 8.0
    shrink_factor: float = 0.5
    growth_factor: float = 1.6
    rejection_threshold: float = 0.05
    strong_agreement: float = 0.8
    boundary_fraction: float = 0.8

    def __post_init__(self) -> None:
        for name in ("initial_radius", "minimum_radius", "maximum_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.minimum_radius <= self.initial_radius <= self.maximum_radius:
            raise ValueError("initial radius must lie between minimum and maximum")
        if not 0.0 < self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must lie strictly between zero and one")
        if not self.growth_factor > 1.0:
            raise ValueError("growth_factor must exceed one")
        if not 0.0 <= self.rejection_threshold < self.strong_agreement:
            raise ValueError("agreement thresholds are inconsistent")
        if not 0.0 < self.boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class TrustRegionUpdate:
    radius_before: float
    radius_after: float
    action: RadiusAction
    reason: str


class TrustRegionController:
    """Update a scalar trust radius from model agreement and step usage."""

    def __init__(self, config: TrustRegionConfig | None = None) -> None:
        self.config = config or TrustRegionConfig()
        self.radius = self.config.initial_radius

    @property
    def exhausted(self) -> bool:
        return self.radius <= self.config.minimum_radius * (1.0 + 1.0e-12)

    def update(
        self,
        *,
        accepted: bool,
        agreement: float,
        step_fraction: float,
    ) -> TrustRegionUpdate:
        if not np.isfinite(step_fraction) or step_fraction < 0.0:
            raise ValueError("step_fraction must be finite and non-negative")
        before = self.radius

        if not accepted or not np.isfinite(agreement):
            after = max(self.config.minimum_radius, before * self.config.shrink_factor)
            action = RadiusAction.SHRINK
            reason = "candidate rejected or agreement was not finite"
        elif agreement < self.config.rejection_threshold:
            after = max(self.config.minimum_radius, before * self.config.shrink_factor)
            action = RadiusAction.SHRINK
            reason = "accepted safeguard step had poor model agreement"
        elif (
            agreement >= self.config.strong_agreement
            and step_fraction >= self.config.boundary_fraction
        ):
            after = min(self.config.maximum_radius, before * self.config.growth_factor)
            action = RadiusAction.GROW
            reason = "strong agreement on a boundary-active step"
        else:
            after = before
            action = RadiusAction.KEEP
            reason = "agreement and step usage support retaining the radius"

        self.radius = after
        return TrustRegionUpdate(
            radius_before=before,
            radius_after=after,
            action=action,
            reason=reason,
        )
