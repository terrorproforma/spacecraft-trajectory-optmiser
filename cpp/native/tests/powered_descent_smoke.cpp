#include "spacepdhcg/native/powered_descent_3dof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace native = spacepdhcg::native;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <std::size_t Columns, typename Matrix>
const double& entry(const Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

void test_linearisation() {
    native::PoweredDescent3DofModel model;
    native::PoweredDescentState state{20.0, -10.0, 120.0, 1.0, -0.5, -7.0, 2'000.0};
    native::PoweredDescentControl control{300.0, -150.0, 8'000.0, 8'010.0};
    const auto nominal = model.dynamics(state, control);
    const auto linearisation = model.linearise(state, control);

    native::PoweredDescentState reconstruction = linearisation.offset;
    for (std::size_t row = 0; row < native::powered_descent_state_dimension; ++row) {
        for (std::size_t column = 0; column < native::powered_descent_state_dimension; ++column) {
            reconstruction[row] += entry<native::powered_descent_state_dimension>(
                linearisation.state_jacobian,
                row,
                column
            ) * state[column];
        }
        for (std::size_t column = 0; column < native::powered_descent_control_dimension; ++column) {
            reconstruction[row] += entry<native::powered_descent_control_dimension>(
                linearisation.control_jacobian,
                row,
                column
            ) * control[column];
        }
        require(std::abs(reconstruction[row] - nominal[row]) < 2.0e-12,
                "affine powered-descent reconstruction is wrong");
    }

    constexpr double state_step = 1.0e-5;
    for (std::size_t column = 0; column < native::powered_descent_state_dimension; ++column) {
        auto plus = state;
        auto minus = state;
        plus[column] += state_step;
        minus[column] -= state_step;
        const auto plus_value = model.dynamics(plus, control);
        const auto minus_value = model.dynamics(minus, control);
        for (std::size_t row = 0; row < native::powered_descent_state_dimension; ++row) {
            const double finite_difference =
                (plus_value[row] - minus_value[row]) / (2.0 * state_step);
            require(
                std::abs(
                    finite_difference - entry<native::powered_descent_state_dimension>(
                        linearisation.state_jacobian,
                        row,
                        column
                    )
                ) < 2.0e-7,
                "powered-descent state Jacobian failed finite differences"
            );
        }
    }

    constexpr double control_step = 1.0e-5;
    for (std::size_t column = 0; column < native::powered_descent_control_dimension; ++column) {
        auto plus = control;
        auto minus = control;
        plus[column] += control_step;
        minus[column] -= control_step;
        const auto plus_value = model.dynamics(state, plus);
        const auto minus_value = model.dynamics(state, minus);
        for (std::size_t row = 0; row < native::powered_descent_state_dimension; ++row) {
            const double finite_difference =
                (plus_value[row] - minus_value[row]) / (2.0 * control_step);
            require(
                std::abs(
                    finite_difference - entry<native::powered_descent_control_dimension>(
                        linearisation.control_jacobian,
                        row,
                        column
                    )
                ) < 2.0e-8,
                "powered-descent control Jacobian failed finite differences"
            );
        }
    }
}

void test_discretisation_rollout_and_constraints() {
    native::PoweredDescent3DofModel model;
    native::PoweredDescentState state{20.0, -10.0, 120.0, 0.0, 0.0, -7.0, 2'000.0};
    native::PoweredDescentControl control{0.0, 0.0, 8'000.0, 8'000.0};
    constexpr double step_seconds = 2.0;

    const auto discrete = model.linearised_euler(state, control, step_seconds);
    native::PoweredDescentState affine_step = discrete.offset;
    for (std::size_t row = 0; row < native::powered_descent_state_dimension; ++row) {
        for (std::size_t column = 0; column < native::powered_descent_state_dimension; ++column) {
            affine_step[row] += entry<native::powered_descent_state_dimension>(
                discrete.state_matrix,
                row,
                column
            ) * state[column];
        }
        for (std::size_t column = 0; column < native::powered_descent_control_dimension; ++column) {
            affine_step[row] += entry<native::powered_descent_control_dimension>(
                discrete.control_matrix,
                row,
                column
            ) * control[column];
        }
    }
    const auto direct_step = model.euler_step(state, control, step_seconds);
    for (std::size_t index = 0; index < native::powered_descent_state_dimension; ++index) {
        require(std::abs(affine_step[index] - direct_step[index]) < 2.0e-12,
                "linearised Euler dynamics do not interpolate the reference");
    }

    const std::vector<native::PoweredDescentControl> controls(5, control);
    const auto euler_states = model.rollout_euler(state, controls, step_seconds);
    const auto rk4_states = model.rollout_rk4(state, controls, step_seconds);
    require(euler_states.size() == 6 && rk4_states.size() == 6,
            "powered-descent rollout has the wrong length");
    require(euler_states.back()[6] < state[6] && rk4_states.back()[6] < state[6],
            "powered-descent rollout did not consume propellant");

    const auto path = model.path_diagnostics(euler_states, controls);
    require(path.maximum_violation() < 1.0e-12,
            "nominal powered-descent path should be feasible");

    auto invalid_control = control;
    invalid_control[3] = 1'000.0;
    const std::array<native::PoweredDescentControl, 1> invalid_controls{invalid_control};
    const std::array<native::PoweredDescentState, 2> invalid_states{state, direct_step};
    const auto violation = model.path_diagnostics(invalid_states, invalid_controls);
    require(violation.thrust_epigraph > 6'900.0,
            "thrust-epigraph violation was not detected");
}

}  // namespace

int main() {
    test_linearisation();
    test_discretisation_rollout_and_constraints();
    return 0;
}
