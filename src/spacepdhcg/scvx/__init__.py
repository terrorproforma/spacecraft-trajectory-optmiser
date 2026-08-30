"""Successive-convexification policies and executable outer loops."""

from .forcing_rule import (
    AdaptiveForcingRule,
    ForcingDecision,
    ForcingRuleConfig,
    OuterResidual,
    SolvePhase,
)
from .powered_descent_3dof import (
    PoweredDescentOuterConfig,
    PoweredDescentSCvxResult,
    PoweredDescentSCvxSolver,
    SCvxIterationRecord,
    clarabel_reference_builder,
    make_dynamics_consistent_reference,
)
from .trust_region import (
    RadiusAction,
    TrustRegionConfig,
    TrustRegionController,
    TrustRegionUpdate,
)

__all__ = [
    "AdaptiveForcingRule",
    "ForcingDecision",
    "ForcingRuleConfig",
    "OuterResidual",
    "PoweredDescentOuterConfig",
    "PoweredDescentSCvxResult",
    "PoweredDescentSCvxSolver",
    "RadiusAction",
    "SCvxIterationRecord",
    "SolvePhase",
    "TrustRegionConfig",
    "TrustRegionController",
    "TrustRegionUpdate",
    "clarabel_reference_builder",
    "make_dynamics_consistent_reference",
]
