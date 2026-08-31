#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

template <std::size_t Rows, std::size_t Columns>
constexpr std::size_t matrix_index(std::size_t row, std::size_t column) noexcept {
    return row * Columns + column;
}

bool close(double left, double right, double tolerance) {
    return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

}  // namespace

int main() {
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::dynamics::ThrustVector;

    const PoweredDescent3DofModel model{};
    const PoweredDescentState state{20.0, -10.0, 120.0, 0.4, -0.2, -7.0, 2'000.0};
    const auto thrust = model.project_thrust(ThrustVector{1'200.0, -500.0, 8'000.0});
    const auto sigma = std::sqrt(
        thrust[0] * thrust[0] + thrust[1] * thrust[1] + thrust[2] * thrust[2]
    );
    const PoweredDescentControl control{thrust[0], thrust[1], thrust[2], sigma};
    const auto analytic = model.jacobians(state, control);

    constexpr double epsilon = 1.0e-5;
    for (std::size_t column = 0; column < state.size(); ++column) {
        auto plus = state;
        auto minus = state;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto forward = model.dynamics(plus, control);
        const auto backward = model.dynamics(minus, control);
        for (std::size_t row = 0; row < state.size(); ++row) {
            const auto finite_difference = (forward[row] - backward[row]) / (2.0 * epsilon);
            const auto expected = analytic.state[matrix_index<7, 7>(row, column)];
            if (!close(finite_difference, expected, 2.0e-6)) {
                return 1;
            }
        }
    }
    for (std::size_t column = 0; column < control.size(); ++column) {
        auto plus = control;
        auto minus = control;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto forward = model.dynamics(state, plus);
        const auto backward = model.dynamics(state, minus);
        for (std::size_t row = 0; row < state.size(); ++row) {
            const auto finite_difference = (forward[row] - backward[row]) / (2.0 * epsilon);
            const auto expected = analytic.control[matrix_index<7, 4>(row, column)];
            if (!close(finite_difference, expected, 2.0e-6)) {
                return 2;
            }
        }
    }

    constexpr double step_seconds = 0.5;
    const auto affine = model.linearised_euler_dynamics(state, control, step_seconds);
    const auto direct = model.euler_step(state, control, step_seconds);
    for (std::size_t row = 0; row < state.size(); ++row) {
        auto reconstructed = affine.offset[row];
        for (std::size_t column = 0; column < state.size(); ++column) {
            reconstructed += affine.state[matrix_index<7, 7>(row, column)] * state[column];
        }
        for (std::size_t column = 0; column < control.size(); ++column) {
            reconstructed += affine.control[matrix_index<7, 4>(row, column)] * control[column];
        }
        if (!close(reconstructed, direct[row], 1.0e-11)) {
            return 3;
        }
    }

    const std::vector<PoweredDescentControl> controls(8U, control);
    const auto states = model.rollout(state, controls, 0.25, true);
    if (states.size() != controls.size() + 1U || states.back()[6] >= state[6]) {
        return 4;
    }
    const auto diagnostics = model.path_diagnostics(states, controls);
    if (diagnostics.thrust_epigraph > 1.0e-9 || diagnostics.tilt > 1.0e-9
        || diagnostics.throttle_upper > 1.0e-9) {
        return 5;
    }
    return 0;
}
