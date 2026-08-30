import numpy as np

from spacepdhcg.models import (
    CWRendezvousConfig,
    CWRendezvousProblem,
    ThrustConstraint,
    cw_continuous_matrices,
    discretise_cw,
)


def test_exact_discretisation_has_correct_small_step_limit() -> None:
    mean_motion = 1.13e-3
    step = 1.0e-6
    continuous_a, continuous_b = cw_continuous_matrices(mean_motion)
    discrete_a, discrete_b = discretise_cw(mean_motion, step)

    np.testing.assert_allclose(discrete_a, np.eye(6) + continuous_a * step, atol=1.0e-12)
    np.testing.assert_allclose(discrete_b, continuous_b * step, atol=1.0e-12)


def test_initial_and_target_updates_preserve_symbolic_structure() -> None:
    problem = CWRendezvousProblem(CWRendezvousConfig(intervals=8))
    first = problem.values(np.array([100.0, 0, 0, 0, 0, 0]), np.zeros(6))
    second = problem.values(np.array([50.0, 20, 0, 0, 0, 0]), np.ones(6))

    np.testing.assert_array_equal(first.quadratic, second.quadratic)
    np.testing.assert_array_equal(first.constraint, second.constraint)
    assert not np.array_equal(first.linear, second.linear)
    assert not np.array_equal(first.lower, second.lower)
    assert problem.structure.n_variables == (8 + 1) * 6 + 8 * 3
    assert problem.structure.n_constraints == 6 + 8 * 6 + 6 + 8 * 3
    assert problem.structure.n_affine_constraints == 0


def test_soc_thrust_uses_native_affine_cone_rows() -> None:
    intervals = 8
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=intervals,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    values = problem.values(np.array([100.0, 0, 0, 0, 0, 0]), np.zeros(6))

    assert problem.structure.n_constraints == 6 + intervals * 6 + 6
    assert problem.structure.n_affine_constraints == intervals * 4
    assert len(problem.structure.affine_cones) == intervals
    assert all(cone.slot_count == 4 for cone in problem.structure.affine_cones)
    assert values.affine_offset.shape == (intervals * 4,)
    np.testing.assert_allclose(values.affine_offset.reshape(intervals, 4)[:, -1], 5.0e-2)


def test_decode_rejects_wrong_decision_shape() -> None:
    problem = CWRendezvousProblem(CWRendezvousConfig(intervals=4))
    try:
        problem.decode(np.zeros(problem.layout.n_variables - 1))
    except ValueError as error:
        assert "wrong shape" in str(error)
    else:
        raise AssertionError("wrong-sized decision vector was accepted")
