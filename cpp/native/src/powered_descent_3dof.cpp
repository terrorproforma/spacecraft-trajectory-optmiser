#include "spacepdhcg/native/powered_descent_3dof.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace spacepdhcg::native {
namespace {

template <std::size_t Columns, typename Matrix>
double& entry(Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

template <std::size_t Columns, typename Matrix>
const double& entry(const Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

[[nodiscard]] PoweredDescentState add_scaled(
    std::span<const double, powered_descent_state_dimension> state,
    std::span<const double, powered_descent_state_dimension> derivative,
    double scale
) {
    PoweredDescentState result{};
    for (std::size_t index = 0; index < powered_descent_state_dimension; ++index) {
        result[index] = state[index] + scale * derivative[index];
    }
    return result;
}

}  // namespace

void PoweredDescent3DofConfig::validate() const {
    for (double value : gravity) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("gravity components must be finite");
        }
    }
    if (!std::isfinite(mass_flow_coefficient) || mass_flow_coefficient <= 0.0) {
        throw std::invalid_argument("mass-flow coefficient must be finite and positive");
    }
    if (!std::isfinite(minimum_mass) || minimum_mass <= 0.0) {
        throw std::invalid_argument("minimum mass must be finite and positive");
    }
    if (!std::isfinite(maximum_thrust) || maximum_thrust <= 0.0) {
        throw std::invalid_argument("maximum thrust must be finite and positive");
    }
    if (!std::isfinite(minimum_sigma) || minimum_sigma < 0.0 ||
        minimum_sigma > maximum_thrust) {
        throw std::invalid_argument("minimum sigma must lie in [0, maximum thrust]");
    }
    const double half_pi = 0.5 * std::numbers::pi_v<double>;
    if (!(maximum_tilt_radians > 0.0 && maximum_tilt_radians < half_pi) ||
        !(glide_slope_radians > 0.0 && glide_slope_radians < half_pi)) {
        throw std::invalid_argument("tilt and glide angles must lie strictly inside (0, pi/2)");
    }
}

double PoweredDescent3DofConfig::tilt_cosine() const {
    return std::cos(maximum_tilt_radians);
}

double PoweredDescent3DofConfig::glide_slope_tangent() const {
    return std::tan(glide_slope_radians);
}

double PoweredDescentPathDiagnostics::maximum_violation() const noexcept {
    return std::max({
        thrust_epigraph,
        throttle_lower,
        throttle_upper,
        tilt,
        minimum_mass,
        altitude,
        glide_slope,
    });
}

PoweredDescent3DofModel::PoweredDescent3DofModel(PoweredDescent3DofConfig config)
    : config_(std::move(config)) {
    config_.validate();
}

void PoweredDescent3DofModel::require_state(
    std::span<const double, powered_descent_state_dimension> state
) {
    for (double value : state) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("powered-descent state must be finite");
        }
    }
    if (state[6] <= 0.0) {
        throw std::invalid_argument("powered-descent mass must be positive");
    }
}

void PoweredDescent3DofModel::require_control(
    std::span<const double, powered_descent_control_dimension> control
) {
    for (double value : control) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("powered-descent control must be finite");
        }
    }
}

void PoweredDescent3DofModel::require_step(double step_seconds) {
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("integration step must be finite and positive");
    }
}

PoweredDescentState PoweredDescent3DofModel::dynamics(
    std::span<const double, powered_descent_state_dimension> state,
    std::span<const double, powered_descent_control_dimension> control
) const {
    require_state(state);
    require_control(control);
    const double mass = state[6];

    PoweredDescentState derivative{};
    derivative[0] = state[3];
    derivative[1] = state[4];
    derivative[2] = state[5];
    for (std::size_t axis = 0; axis < 3; ++axis) {
        derivative[3 + axis] = control[axis] / mass + config_.gravity[axis];
    }
    derivative[6] = -config_.mass_flow_coefficient * control[3];
    return derivative;
}

PoweredDescentLinearisation PoweredDescent3DofModel::linearise(
    std::span<const double, powered_descent_state_dimension> state,
    std::span<const double, powered_descent_control_dimension> control
) const {
    require_state(state);
    require_control(control);
    const double mass = state[6];

    PoweredDescentLinearisation result{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        entry<powered_descent_state_dimension>(
            result.state_jacobian,
            axis,
            3 + axis
        ) = 1.0;
        entry<powered_descent_state_dimension>(
            result.state_jacobian,
            3 + axis,
            6
        ) = -control[axis] / (mass * mass);
        entry<powered_descent_control_dimension>(
            result.control_jacobian,
            3 + axis,
            axis
        ) = 1.0 / mass;
    }
    entry<powered_descent_control_dimension>(result.control_jacobian, 6, 3) =
        -config_.mass_flow_coefficient;

    const auto derivative = dynamics(state, control);
    for (std::size_t row = 0; row < powered_descent_state_dimension; ++row) {
        double linearised_value = 0.0;
        for (std::size_t column = 0; column < powered_descent_state_dimension; ++column) {
            linearised_value += entry<powered_descent_state_dimension>(
                result.state_jacobian,
                row,
                column
            ) * state[column];
        }
        for (std::size_t column = 0; column < powered_descent_control_dimension; ++column) {
            linearised_value += entry<powered_descent_control_dimension>(
                result.control_jacobian,
                row,
                column
            ) * control[column];
        }
        result.offset[row] = derivative[row] - linearised_value;
    }
    return result;
}

PoweredDescentDiscreteLinearisation PoweredDescent3DofModel::linearised_euler(
    std::span<const double, powered_descent_state_dimension> state,
    std::span<const double, powered_descent_control_dimension> control,
    double step_seconds
) const {
    require_step(step_seconds);
    const auto continuous = linearise(state, control);
    PoweredDescentDiscreteLinearisation result{};
    for (std::size_t row = 0; row < powered_descent_state_dimension; ++row) {
        for (std::size_t column = 0; column < powered_descent_state_dimension; ++column) {
            entry<powered_descent_state_dimension>(result.state_matrix, row, column) =
                step_seconds * entry<powered_descent_state_dimension>(
                    continuous.state_jacobian,
                    row,
                    column
                );
        }
        entry<powered_descent_state_dimension>(result.state_matrix, row, row) += 1.0;
        for (std::size_t column = 0; column < powered_descent_control_dimension; ++column) {
            entry<powered_descent_control_dimension>(result.control_matrix, row, column) =
                step_seconds * entry<powered_descent_control_dimension>(
                    continuous.control_jacobian,
                    row,
                    column
                );
        }
        result.offset[row] = step_seconds * continuous.offset[row];
    }
    return result;
}

PoweredDescentState PoweredDescent3DofModel::euler_step(
    std::span<const double, powered_descent_state_dimension> state,
    std::span<const double, powered_descent_control_dimension> control,
    double step_seconds
) const {
    require_step(step_seconds);
    const auto derivative = dynamics(state, control);
    const auto result = add_scaled(state, derivative, step_seconds);
    require_state(result);
    return result;
}

PoweredDescentState PoweredDescent3DofModel::rk4_step(
    std::span<const double, powered_descent_state_dimension> state,
    std::span<const double, powered_descent_control_dimension> control,
    double step_seconds
) const {
    require_step(step_seconds);
    const auto first = dynamics(state, control);
    const auto second_state = add_scaled(state, first, 0.5 * step_seconds);
    const auto second = dynamics(second_state, control);
    const auto third_state = add_scaled(state, second, 0.5 * step_seconds);
    const auto third = dynamics(third_state, control);
    const auto fourth_state = add_scaled(state, third, step_seconds);
    const auto fourth = dynamics(fourth_state, control);

    PoweredDescentState result{};
    for (std::size_t index = 0; index < powered_descent_state_dimension; ++index) {
        result[index] = state[index] + step_seconds / 6.0 *
            (first[index] + 2.0 * second[index] + 2.0 * third[index] + fourth[index]);
    }
    require_state(result);
    return result;
}

std::vector<PoweredDescentState> PoweredDescent3DofModel::rollout_euler(
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const PoweredDescentControl> controls,
    double step_seconds
) const {
    require_state(initial_state);
    require_step(step_seconds);
    std::vector<PoweredDescentState> states(controls.size() + 1U);
    std::copy(initial_state.begin(), initial_state.end(), states.front().begin());
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        states[interval + 1U] = euler_step(states[interval], controls[interval], step_seconds);
    }
    return states;
}

std::vector<PoweredDescentState> PoweredDescent3DofModel::rollout_rk4(
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const PoweredDescentControl> controls,
    double step_seconds
) const {
    require_state(initial_state);
    require_step(step_seconds);
    std::vector<PoweredDescentState> states(controls.size() + 1U);
    std::copy(initial_state.begin(), initial_state.end(), states.front().begin());
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        states[interval + 1U] = rk4_step(states[interval], controls[interval], step_seconds);
    }
    return states;
}

PoweredDescentPathDiagnostics PoweredDescent3DofModel::path_diagnostics(
    std::span<const PoweredDescentState> states,
    std::span<const PoweredDescentControl> controls
) const {
    if (states.size() != controls.size() + 1U || states.empty()) {
        throw std::invalid_argument("path diagnostics require one more state than control");
    }

    PoweredDescentPathDiagnostics result{};
    for (const auto& state : states) {
        require_state(state);
        const double horizontal = std::hypot(state[0], state[1]);
        result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[6]);
        result.altitude = std::max(result.altitude, -state[2]);
        result.glide_slope = std::max(
            result.glide_slope,
            horizontal - config_.glide_slope_tangent() * state[2]
        );
    }
    for (const auto& control : controls) {
        require_control(control);
        const double thrust_norm = std::hypot(control[0], control[1], control[2]);
        result.thrust_epigraph = std::max(result.thrust_epigraph, thrust_norm - control[3]);
        result.throttle_lower = std::max(result.throttle_lower, config_.minimum_sigma - control[3]);
        result.throttle_upper = std::max(result.throttle_upper, control[3] - config_.maximum_thrust);
        result.tilt = std::max(
            result.tilt,
            config_.tilt_cosine() * control[3] - control[2]
        );
    }

    result.thrust_epigraph = std::max(result.thrust_epigraph, 0.0);
    result.throttle_lower = std::max(result.throttle_lower, 0.0);
    result.throttle_upper = std::max(result.throttle_upper, 0.0);
    result.tilt = std::max(result.tilt, 0.0);
    result.minimum_mass = std::max(result.minimum_mass, 0.0);
    result.altitude = std::max(result.altitude, 0.0);
    result.glide_slope = std::max(result.glide_slope, 0.0);
    return result;
}

}  // namespace spacepdhcg::native
