import numpy as np

from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.scvx import (
    ForcingRuleConfig,
    PoweredDescentOuterConfig,
    PoweredDescentSCvxSolver,
    TrustRegionConfig,
    make_dynamics_consistent_reference,
)
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def _problem(intervals: int = 8, step_seconds: float = 2.0):
    model = PoweredDescent3DOFModel()
    initial = np.array([10.0, 0.0, 80.0, 0.0, 0.0, -5.0, 2_000.0])
    target_position = np.zeros(3)
    target_velocity = np.zeros(3)
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(
            intervals=intervals,
            step_seconds=step_seconds,
            trust_radius=1.0,
        ),
    )
    return model, subproblem, initial, target_position, target_velocity


def test_initial_reference_is_dynamics_consistent_and_reaches_easy_target() -> None:
    model, subproblem, initial, target_position, target_velocity = _problem()

    states, controls = make_dynamics_consistent_reference(
        model,
        initial,
        target_position,
        target_velocity,
        intervals=subproblem.layout.intervals,
        step_seconds=subproblem.config.step_seconds,
    )

    replay = model.rollout(initial, controls, subproblem.config.step_seconds)
    np.testing.assert_allclose(states, replay, atol=1.0e-11, rtol=0.0)
    np.testing.assert_allclose(states[-1, :3], target_position, atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(states[-1, 3:6], target_velocity, atol=1.0e-10, rtol=0.0)
    assert model.path_diagnostics(states, controls).feasible(1.0e-9)


def test_reference_scvx_outer_loop_accepts_a_finite_candidate() -> None:
    _, subproblem, initial, target_position, target_velocity = _problem()
    solver = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(
            max_iterations=4,
            minimum_iterations=1,
            convergence_tolerance=2.0e-3,
            step_tolerance=0.2,
        ),
        forcing_config=ForcingRuleConfig(
            exploration_iterations=1,
            epsilon_floor=1.0e-7,
            polish_tolerance=1.0e-8,
        ),
        trust_config=TrustRegionConfig(
            initial_radius=1.0,
            minimum_radius=0.05,
            maximum_radius=4.0,
        ),
    )

    result = solver.solve(initial, target_position, target_velocity)

    assert result.outer_iterations >= 1
    assert result.accepted_iterations >= 1
    assert np.isfinite(result.merit)
    assert np.all(np.isfinite(result.states))
    assert np.all(np.isfinite(result.controls))
    assert result.path_diagnostics.maximum_violation < 2.0e-2
    assert result.residual.terminal < 5.0e-2
    assert result.residual.path < 2.0e-5
    for record in result.iterations:
        if record.accepted and not record.restoration_accepted:
            assert record.actual_reduction > 0.0
    assert all(record.solver_status.lower().startswith("solved") for record in result.iterations)


def test_outer_loop_records_tolerance_trust_and_reduction_evidence() -> None:
    _, subproblem, initial, target_position, target_velocity = _problem(intervals=6)
    solver = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(max_iterations=2, minimum_iterations=1),
        forcing_config=ForcingRuleConfig(exploration_iterations=1),
    )

    result = solver.solve(initial, target_position, target_velocity)
    record = result.iterations[0]
    payload = record.as_dict()

    assert record.requested_tolerance > 0.0
    assert record.effective_tolerance > 0.0
    assert record.trust_radius_before > 0.0
    assert record.trust_radius_after > 0.0
    assert np.isfinite(record.predicted_reduction)
    assert np.isfinite(record.actual_reduction)
    assert payload["phase"] in {"exploration", "convergence", "polish"}
    assert isinstance(payload["residual"], dict)
