import numpy as np

from spacepdhcg.models import PoweredDescent3DOFConfig, PoweredDescent3DOFModel


def _state_difference(model, state, control, index, step=1.0e-5):
    direction = np.zeros_like(state)
    direction[index] = step
    return (
        model.dynamics(state + direction, control)
        - model.dynamics(state - direction, control)
    ) / (2.0 * step)


def _control_difference(model, state, control, index, step=1.0e-5):
    direction = np.zeros_like(control)
    direction[index] = step
    return (
        model.dynamics(state, control + direction)
        - model.dynamics(state, control - direction)
    ) / (2.0 * step)


def test_vertical_hover_cancels_constant_gravity() -> None:
    model = PoweredDescent3DOFModel()
    mass = 2_000.0
    force = -mass * model.config.gravity_vector
    sigma = np.linalg.norm(force)
    state = np.array([0.0, 0.0, 1_000.0, 0.0, 0.0, 0.0, mass])
    control = np.concatenate((force, np.array([sigma])))

    derivative = model.dynamics(state, control)

    np.testing.assert_allclose(derivative[:6], 0.0, atol=1.0e-14)
    assert derivative[6] == -model.config.mass_flow_coefficient * sigma


def test_analytic_jacobians_match_central_differences() -> None:
    model = PoweredDescent3DOFModel()
    state = np.array([120.0, -45.0, 900.0, -8.0, 3.0, -22.0, 1_850.0])
    control = np.array([350.0, -220.0, 7_000.0, 7_050.0])
    state_jacobian, control_jacobian = model.jacobians(state, control)
    numerical_state = np.column_stack(
        [_state_difference(model, state, control, index) for index in range(7)]
    )
    numerical_control = np.column_stack(
        [_control_difference(model, state, control, index) for index in range(4)]
    )

    np.testing.assert_allclose(state_jacobian, numerical_state, atol=2.0e-9, rtol=2.0e-7)
    np.testing.assert_allclose(control_jacobian, numerical_control, atol=2.0e-9, rtol=2.0e-7)


def test_affine_linearisation_is_exact_at_reference_and_second_order_nearby() -> None:
    model = PoweredDescent3DOFModel()
    state = np.array([50.0, -10.0, 700.0, -4.0, 1.0, -18.0, 1_700.0])
    control = np.array([250.0, 100.0, 6_500.0, 6_520.0])
    state_jacobian, control_jacobian, offset = model.affine_linearisation(state, control)
    np.testing.assert_allclose(
        state_jacobian @ state + control_jacobian @ control + offset,
        model.dynamics(state, control),
        atol=1.0e-13,
    )

    state_delta = np.array([0.02, -0.03, 0.01, 0.004, 0.002, -0.003, 0.8])
    control_delta = np.array([1.5, -2.0, 3.0, 4.0])

    def error(scale):
        candidate_state = state + scale * state_delta
        candidate_control = control + scale * control_delta
        nonlinear = model.dynamics(candidate_state, candidate_control)
        affine = (
            state_jacobian @ candidate_state
            + control_jacobian @ candidate_control
            + offset
        )
        return np.linalg.norm(nonlinear - affine)

    full_error = error(1.0)
    half_error = error(0.5)
    assert full_error > 0.0
    assert half_error < 0.3 * full_error


def test_rollout_mass_and_vertical_path_diagnostics() -> None:
    model = PoweredDescent3DOFModel()
    initial = np.array([0.0, 0.0, 1_000.0, 0.0, 0.0, -20.0, 2_000.0])
    controls = np.tile(np.array([0.0, 0.0, 8_000.0, 8_000.0]), (5, 1))
    step_seconds = 2.0
    states = model.rollout(initial, controls, step_seconds)
    expected_mass = initial[6] - (
        controls.shape[0]
        * step_seconds
        * model.config.mass_flow_coefficient
        * controls[0, 3]
    )

    assert states.shape == (6, 7)
    np.testing.assert_allclose(states[-1, 6], expected_mass, atol=1.0e-12)
    assert np.all(np.diff(states[:, 6]) < 0.0)
    assert model.path_diagnostics(states, controls).feasible(1.0e-12)


def test_path_diagnostics_detect_violations() -> None:
    config = PoweredDescent3DOFConfig(
        minimum_mass=1_000.0,
        maximum_thrust=10_000.0,
        minimum_sigma=500.0,
    )
    model = PoweredDescent3DOFModel(config)
    states = np.array(
        [
            [200.0, 0.0, 10.0, 0.0, 0.0, 0.0, 900.0],
            [200.0, 0.0, -2.0, 0.0, 0.0, 0.0, 900.0],
        ]
    )
    controls = np.array([[6_000.0, 0.0, 100.0, 400.0]])
    diagnostics = model.path_diagnostics(states, controls)

    assert diagnostics.thrust_epigraph > 0.0
    assert diagnostics.throttle_lower == 100.0
    assert diagnostics.tilt > 0.0
    assert diagnostics.minimum_mass == 100.0
    assert diagnostics.altitude == 2.0
    assert diagnostics.glide_slope > 0.0
    assert not diagnostics.feasible()
