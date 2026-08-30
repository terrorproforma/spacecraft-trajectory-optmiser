import numpy as np

from spacepdhcg.scvx import (
    AdaptiveForcingRule,
    ForcingRuleConfig,
    OuterResidual,
    RadiusAction,
    SolvePhase,
    TrustRegionConfig,
    TrustRegionController,
)


def test_forcing_rule_matches_residual_and_geometric_schedule() -> None:
    config = ForcingRuleConfig(exploration_iterations=1)
    rule = AdaptiveForcingRule(config)
    residual = OuterResidual(dynamics=0.1, path=0.02, terminal=0.03, step=0.01)

    decision = rule.request(3, residual)
    expected = min(
        config.epsilon_max,
        config.coefficient * residual.maximum ** (1.0 + config.alpha),
        config.epsilon_0 * config.gamma**3,
    )

    assert decision.phase is SolvePhase.CONVERGENCE
    assert decision.tolerance == max(config.epsilon_floor, expected)
    assert decision.raw_target == expected


def test_forcing_rule_enters_polish_only_after_all_gates_pass() -> None:
    config = ForcingRuleConfig()
    rule = AdaptiveForcingRule(config)
    residual = OuterResidual(1.0e-4, 2.0e-4, 1.5e-4, 1.0e-3)

    decision = rule.request(5, residual, accepted_streak=2, agreement=0.9)

    assert decision.phase is SolvePhase.POLISH
    assert decision.tolerance == config.polish_tolerance
    assert decision.iteration_limit == config.polish_iteration_limit


def test_rejected_inaccurate_subproblem_is_resolved_before_radius_shrink() -> None:
    config = ForcingRuleConfig(resolve_kkt_multiple=4.0)
    rule = AdaptiveForcingRule(config)

    assert rule.should_resolve(
        accepted=False,
        primal_residual=8.0e-4,
        dual_residual=2.0e-4,
        requested_tolerance=1.0e-4,
    )
    assert not rule.should_resolve(
        accepted=True,
        primal_residual=8.0e-4,
        dual_residual=2.0e-4,
        requested_tolerance=1.0e-4,
    )
    assert rule.refined_tolerance(1.0e-4) == 1.0e-5


def test_trust_region_shrinks_rejections_and_grows_good_boundary_steps() -> None:
    controller = TrustRegionController(
        TrustRegionConfig(initial_radius=1.0, minimum_radius=0.1, maximum_radius=4.0)
    )

    rejected = controller.update(accepted=False, agreement=-1.0, step_fraction=0.5)
    assert rejected.action is RadiusAction.SHRINK
    assert rejected.radius_after == 0.5

    grown = controller.update(accepted=True, agreement=0.95, step_fraction=0.9)
    assert grown.action is RadiusAction.GROW
    assert grown.radius_after > grown.radius_before


def test_outer_residual_rejects_nonfinite_components() -> None:
    with np.testing.assert_raises(ValueError):
        OuterResidual(np.inf, 0.0, 0.0, 0.0)
