#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

template <std::size_t Rows, std::size_t Columns>
constexpr std::size_t matrix_index(std::size_t row, std::size_t column) noexcept {
    return row * Columns + column;
}

bool close(double left, double right, double relative_tolerance) {
    return std::abs(left - right)
           <= relative_tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

}  // namespace

int main() {
    using spacepdhcg::dynamics::PoweredDescent6DofControl;
    using spacepdhcg::dynamics::PoweredDescent6DofModel;
    using spacepdhcg::dynamics::PoweredDescent6DofState;

    const PoweredDescent6DofModel model{};
    PoweredDescent6DofState state{
        10.0,
        -5.0,
        100.0,
        0.2,
        -0.1,
        -4.0,
        0.9805806756909202,
        0.0980580675690920,
        -0.1470871013536380,
        0.0490290337845460,
        0.03,
        -0.04,
        0.02,
        2'000.0,
    };
    const PoweredDescent6DofControl control{
        500.0,
        -300.0,
        8'000.0,
        20.0,
        -15.0,
        10.0,
        8'021.221850622,
    };
    const auto analytic = model.jacobians(state, control);
    constexpr double epsilon = 1.0e-6;

    for (std::size_t column = 0; column < state.size(); ++column) {
        auto plus = state;
        auto minus = state;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto forward = model.dynamics(plus, control);
        const auto backward = model.dynamics(minus, control);
        for (std::size_t row = 0; row < state.size(); ++row) {
            const auto finite_difference = (forward[row] - backward[row]) / (2.0 * epsilon);
            const auto expected = analytic.state[matrix_index<14U, 14U>(row, column)];
            if (!close(finite_difference, expected, 5.0e-5)) {
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
            const auto expected = analytic.control[matrix_index<14U, 7U>(row, column)];
            if (!close(finite_difference, expected, 5.0e-5)) {
                return 2;
            }
        }
    }

    const std::vector<PoweredDescent6DofControl> controls(10U, control);
    const auto states = model.rollout(state, controls, 0.05);
    const auto& final = states.back();
    const auto quaternion_norm = std::sqrt(
        final[6U] * final[6U] + final[7U] * final[7U] + final[8U] * final[8U]
        + final[9U] * final[9U]
    );
    if (std::abs(quaternion_norm - 1.0) > 1.0e-10 || final[13U] >= state[13U]) {
        return 3;
    }
    const auto diagnostics = model.path_diagnostics(states, controls);
    if (diagnostics.quaternion_norm_error > 1.0e-8 || diagnostics.torque > 1.0e-9
        || diagnostics.throttle_upper > 1.0e-9) {
        return 4;
    }
    return 0;
}
