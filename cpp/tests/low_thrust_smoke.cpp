#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"

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
    using spacepdhcg::dynamics::LowThrustControl;
    using spacepdhcg::dynamics::LowThrustState;
    using spacepdhcg::dynamics::LowThrustTwoBodyModel;
    using spacepdhcg::transcription::LowThrustScvxConfig;
    using spacepdhcg::transcription::LowThrustSubproblem;

    const LowThrustTwoBodyModel model{};
    const LowThrustState state{7'000.0, 0.0, 50.0, 0.0, 7.5, 0.1, 500.0};
    const LowThrustControl control{0.2, -0.1, 0.05, 0.25};
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
            const auto expected = analytic.state[matrix_index<7U, 7U>(row, column)];
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
            const auto expected = analytic.control[matrix_index<7U, 4U>(row, column)];
            if (!close(finite_difference, expected, 2.0e-6)) {
                return 2;
            }
        }
    }

    const LowThrustScvxConfig config{
        .intervals = 5U,
        .step_seconds = 10.0,
        .trust_radius = 1.0,
    };
    const LowThrustSubproblem subproblem(model, config);
    if (subproblem.layout().variables() != 132U
        || subproblem.layout().scalar_rows() != 124U
        || subproblem.layout().affine_rows() != 88U
        || subproblem.structure().quadratic.nonzeros() != 132U
        || subproblem.structure().scalar_constraint.nonzeros() != 626U
        || subproblem.structure().affine_cone->nonzeros() != 82U
        || subproblem.structure().affine_cones.size() != 11U) {
        return 3;
    }

    const LowThrustState initial{
        7'000.0,
        0.0,
        0.0,
        0.0,
        std::sqrt(model.config().gravitational_parameter / 7'000.0),
        0.0,
        500.0,
    };
    const LowThrustControl coast{0.0, 0.0, 0.0, 0.0};
    const std::vector<LowThrustControl> controls(config.intervals, coast);
    const auto states = model.rollout(initial, controls, config.step_seconds, false);
    auto problem = subproblem.problem(states, controls, initial, states.back());
    const auto fingerprint = problem.topology_fingerprint();
    const auto decision = subproblem.reference_decision(states, controls);
    const auto diagnostics = subproblem.diagnostics(decision, problem.values());
    if (diagnostics.maximum_violation() > 1.0e-9
        || diagnostics.linearised_dynamics_defect_inf > 1.0e-9
        || diagnostics.terminal_error_inf > 1.0e-9
        || diagnostics.radial_linearisation_error_inf > 1.0e-9
        || diagnostics.virtual_control_inf > 1.0e-12) {
        return 4;
    }

    problem.update_values(
        subproblem.values(states, controls, initial, states.back(), 0.5)
    );
    if (problem.update_count() != 1U || problem.topology_fingerprint() != fingerprint) {
        return 5;
    }
    return 0;
}
