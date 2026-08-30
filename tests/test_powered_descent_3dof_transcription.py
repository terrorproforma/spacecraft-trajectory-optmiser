import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def _vertical_reference(intervals=6, step_seconds=2.0, force=8_000.0):
    model = PoweredDescent3DOFModel()
    initial = np.array([0.0, 0.0, 1_000.0, 0.0, 0.0, -20.0, 2_000.0])
    controls = np.tile(np.array([0.0, 0.0, force, force]), (intervals, 1))
    states = model.rollout(initial, controls, step_seconds)
    return model, initial, states, controls


def test_euler_feasible_reference_satisfies_fixed_convex_subproblem() -> None:
    intervals = 6
    step_seconds = 2.0
    model, initial, states, controls = _vertical_reference(intervals, step_seconds)
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(intervals=intervals, step_seconds=step_seconds),
    )
    values = subproblem.values(
        states,
        controls,
        initial,
        states[-1, :3],
        states[-1, 3:6],
    )
    decision = subproblem.reference_decision(states, controls)
    diagnostics = subproblem.diagnostics(decision, values)

    assert diagnostics.scalar_violation_inf < 1.0e-11
    assert diagnostics.variable_bound_violation_inf == 0.0
    assert diagnostics.cone_violation_inf < 1.0e-12
    assert diagnostics.linearised_dynamics_defect_inf < 1.0e-11
    assert diagnostics.nonlinear_dynamics_defect_inf < 1.0e-11
    assert diagnostics.terminal_error_inf == 0.0
    assert diagnostics.virtual_control_inf == 0.0
    assert diagnostics.convex_feasible(1.0e-10)


def test_numerical_updates_preserve_sparse_and_cone_structure() -> None:
    intervals = 5
    step_seconds = 2.0
    model, initial, first_states, first_controls = _vertical_reference(
        intervals,
        step_seconds,
        force=8_000.0,
    )
    second_controls = np.tile(np.array([0.0, 0.0, 7_500.0, 7_500.0]), (intervals, 1))
    second_states = model.rollout(initial, second_controls, step_seconds)
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(intervals=intervals, step_seconds=step_seconds),
    )
    first = subproblem.values(
        first_states,
        first_controls,
        initial,
        first_states[-1, :3],
        first_states[-1, 3:6],
    )
    second = subproblem.values(
        second_states,
        second_controls,
        initial,
        second_states[-1, :3],
        second_states[-1, 3:6],
    )

    np.testing.assert_array_equal(
        subproblem.structure.constraint.indices,
        subproblem.structure.constraint.indices.copy(),
    )
    np.testing.assert_array_equal(
        subproblem.structure.affine_cone.indptr,
        subproblem.structure.affine_cone.indptr.copy(),
    )
    assert subproblem.structure.affine_cones == subproblem.structure.affine_cones
    assert not np.array_equal(first.constraint, second.constraint)
    assert not np.array_equal(first.linear, second.linear)
    assert not np.array_equal(first.affine_offset, second.affine_offset)
    assert first.constraint.shape == second.constraint.shape
    assert first.affine_cone.shape == second.affine_cone.shape


def test_cone_and_layout_counts_match_transcription_design() -> None:
    intervals = 4
    subproblem = PoweredDescent3DOFSubproblem(
        config=PoweredDescentSCvxConfig(intervals=intervals),
    )
    layout = subproblem.layout

    assert layout.n_variables == (intervals + 1) * 7 + intervals * 4 + 2 * intervals * 7
    assert layout.n_scalar_constraints == 7 + intervals * 7 + 6 + 2 * intervals * 7 + intervals
    assert layout.n_affine_cone_rows == 4 * intervals + 3 * (intervals + 1) + 12 * intervals + 8
    assert len(subproblem.structure.affine_cones) == intervals + (intervals + 1) + intervals + 1


def test_clarabel_solves_first_powered_descent_convex_subproblem() -> None:
    intervals = 6
    step_seconds = 2.0
    model, initial, states, controls = _vertical_reference(intervals, step_seconds)
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(
            intervals=intervals,
            step_seconds=step_seconds,
            trust_radius=0.5,
        ),
    )
    values = subproblem.values(
        states,
        controls,
        initial,
        states[-1, :3],
        states[-1, 3:6],
    )
    backend = PersistentClarabel(
        subproblem.canonical(
            states,
            controls,
            initial,
            states[-1, :3],
            states[-1, 3:6],
        ),
        tolerance=1.0e-8,
    )
    solution = backend.solve()
    diagnostics = subproblem.diagnostics(solution.primal, values)
    solved_states, solved_controls, _, _ = subproblem.decode(solution.primal)
    path = model.path_diagnostics(solved_states, solved_controls)

    assert solution.solved, solution.status
    assert diagnostics.convex_feasible(2.0e-6)
    assert diagnostics.terminal_error_inf < 2.0e-6
    assert path.feasible(2.0e-6)
