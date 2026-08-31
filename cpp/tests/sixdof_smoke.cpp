#include "spacepdhcg/core/powered_descent_6dof.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    const PoweredDescent6DOF model;
    const PoweredDescent6DOFState state{
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        -5.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2'000.0,
    };
    const PoweredDescent6DOFControl control{
        0.0,
        0.0,
        8'000.0,
        0.0,
        0.0,
        0.0,
        8'000.0,
    };

    const auto derivative = model.dynamics(state, control);
    if (std::abs(derivative[5] - 0.289) > 1.0e-12 || derivative[13] >= 0.0) {
        return 1;
    }
    for (std::size_t index = 6; index < 13; ++index) {
        if (std::abs(derivative[index]) > 1.0e-14) {
            return 2;
        }
    }

    const auto jacobians = model.numerical_jacobians(state, control);
    const auto mass_column = jacobians.state[
        powered_descent_6dof_state_index(5, 13)
    ];
    if (std::abs(mass_column + 8'000.0 / (2'000.0 * 2'000.0)) > 1.0e-8) {
        return 3;
    }
    const auto thrust_derivative = jacobians.control[
        powered_descent_6dof_control_index(5, 2)
    ];
    if (std::abs(thrust_derivative - 1.0 / 2'000.0) > 1.0e-9) {
        return 4;
    }

    const std::vector<PoweredDescent6DOFControl> controls(10U, control);
    const auto states = model.rollout(state, controls, 0.1);
    if (states.size() != 11U || states.back()[13] >= state[13]) {
        return 5;
    }
    const Quaternion final_quaternion{
        states.back()[6],
        states.back()[7],
        states.back()[8],
        states.back()[9],
    };
    if (std::abs(quaternion_norm(final_quaternion) - 1.0) > 1.0e-12) {
        return 6;
    }
    if (model.path_diagnostics(states, controls).maximum_violation() > 1.0e-12) {
        return 7;
    }

    constexpr double root_half = 0.7071067811865475244;
    const auto rotation = quaternion_rotation_matrix(
        Quaternion{root_half, 0.0, root_half, 0.0}
    );
    const double rotated_x = rotation[2];
    const double rotated_y = rotation[5];
    const double rotated_z = rotation[8];
    if (std::abs(rotated_x - 1.0) > 1.0e-12 || std::abs(rotated_y) > 1.0e-12 ||
        std::abs(rotated_z) > 1.0e-12) {
        return 8;
    }
    return 0;
}
