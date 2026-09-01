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
        || diagnostics.pointing > 1.0e-9 || diagnostics.glide_slope > 1.0e-9
        || diagnostics.throttle_upper > 1.0e-9) {
        return 4;
    }
    auto violated_states = states;
    auto violated_controls = controls;
    violated_states[3U][0U] = 200.0;
    violated_states[3U][2U] = 1.0;
    violated_controls[4U][2U] = 0.0;
    violated_controls[4U][6U] = 1'000.0;
    const auto violations = model.path_diagnostics(
        violated_states,
        violated_controls
    );
    if (violations.glide_slope < 190.0 || violations.pointing < 800.0) {
        return 5;
    }

    const PoweredDescent6DofState qualification_initial{
        0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 2'000.0,
    };
    const PoweredDescent6DofControl qualification_control{
        0.0, 0.0, 7'422.0, 0.0, 0.0, 0.0, 7'422.0,
    };
    PoweredDescent6DofState displaced = qualification_initial;
    constexpr double attitude_dispersion = 0.05;
    constexpr double rate_dispersion = 0.05;
    displaced[6U] = std::cos(0.5 * attitude_dispersion);
    displaced[7U] = std::sin(0.5 * attitude_dispersion);
    displaced[8U] = 0.0;
    displaced[9U] = 0.0;
    displaced[10U] = rate_dispersion;
    displaced[11U] = 0.0;
    displaced[12U] = 0.0;
    std::vector<PoweredDescent6DofControl> reference_controls(
        20U,
        qualification_control
    );
    auto half_step_controls = reference_controls;
    auto full_step_controls = reference_controls;
    for (std::size_t interval = 0U; interval < reference_controls.size(); ++interval) {
        const double full_torque = interval < 10U ? -875.0 : 625.0;
        half_step_controls[interval][3U] = 0.5 * full_torque;
        full_step_controls[interval][3U] = full_torque;
    }
    const auto target = model.rollout(
        qualification_initial,
        reference_controls,
        0.05
    ).back();
    const auto reference_final =
        model.rollout(displaced, reference_controls, 0.05).back();
    const auto half_step_final =
        model.rollout(displaced, half_step_controls, 0.05).back();
    const auto full_step_final =
        model.rollout(displaced, full_step_controls, 0.05).back();
    const auto terminal_error = [&target](const PoweredDescent6DofState& value) {
        double maximum = 0.0;
        for (std::size_t component = 0U; component < 13U; ++component) {
            maximum = std::max(
                maximum,
                std::abs(value[component] - target[component])
            );
        }
        return maximum;
    };
    const double initial_error = terminal_error(reference_final);
    const double first_accepted_error = terminal_error(half_step_final);
    const double second_accepted_error = terminal_error(full_step_final);
    if (!(first_accepted_error < initial_error)
        || !(second_accepted_error < first_accepted_error)
        || std::abs(full_step_controls.front()[3U]) <= 0.0) {
        return 6;
    }
    return 0;
}
