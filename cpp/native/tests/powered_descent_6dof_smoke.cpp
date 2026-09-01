#include "spacepdhcg/native/powered_descent_6dof.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <span>
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

bool close(double left, double right, double relative_tolerance) {
    return std::abs(left - right) <=
           relative_tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

void test_linearisation_and_rollout() {
    const native::PoweredDescent6DofModel model{};
    native::PoweredDescent6DofState state{
        10.0, -5.0, 100.0,
        0.2, -0.1, -4.0,
        0.9805806756909202, 0.0980580675690920,
        -0.1470871013536380, 0.0490290337845460,
        0.03, -0.04, 0.02,
        2'000.0,
    };
    const auto normalised = native::normalise_quaternion(
        std::span<const double, 4>{state.data() + 6, 4}
    );
    std::copy(normalised.begin(), normalised.end(), state.begin() + 6);
    const native::PoweredDescent6DofControl control{
        500.0, -300.0, 8'000.0,
        20.0, -15.0, 10.0,
        8'021.221850622,
    };
    const auto derivative = model.dynamics(state, control);
    const auto linearisation = model.linearise(state, control);

    for (std::size_t row = 0; row < state.size(); ++row) {
        double reconstructed = linearisation.offset[row];
        for (std::size_t column = 0; column < state.size(); ++column) {
            reconstructed += entry<native::powered_descent_6dof_state_dimension>(
                linearisation.state_jacobian, row, column
            ) * state[column];
        }
        for (std::size_t column = 0; column < control.size(); ++column) {
            reconstructed += entry<native::powered_descent_6dof_control_dimension>(
                linearisation.control_jacobian, row, column
            ) * control[column];
        }
        require(close(reconstructed, derivative[row], 2.0e-12),
                "six-DoF affine linearisation misses its reference point");
    }

    constexpr double epsilon = 1.0e-6;
    for (std::size_t column = 0; column < state.size(); ++column) {
        auto plus = state;
        auto minus = state;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto forward = model.dynamics(plus, control);
        const auto backward = model.dynamics(minus, control);
        for (std::size_t row = 0; row < state.size(); ++row) {
            const double finite_difference = (forward[row] - backward[row]) / (2.0 * epsilon);
            require(
                close(
                    finite_difference,
                    entry<native::powered_descent_6dof_state_dimension>(
                        linearisation.state_jacobian, row, column
                    ),
                    5.0e-5
                ),
                "six-DoF state Jacobian failed finite differences"
            );
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
            const double finite_difference = (forward[row] - backward[row]) / (2.0 * epsilon);
            require(
                close(
                    finite_difference,
                    entry<native::powered_descent_6dof_control_dimension>(
                        linearisation.control_jacobian, row, column
                    ),
                    5.0e-5
                ),
                "six-DoF control Jacobian failed finite differences"
            );
        }
    }

    const std::vector<native::PoweredDescent6DofControl> controls(10U, control);
    const auto states = model.rollout_rk4(state, controls, 0.05);
    const auto& final = states.back();
    const double quaternion_norm = std::sqrt(
        final[6] * final[6] + final[7] * final[7] +
        final[8] * final[8] + final[9] * final[9]
    );
    require(std::abs(quaternion_norm - 1.0) < 1.0e-10,
            "six-DoF RK4 rollout did not normalise attitude");
    require(final[13] < state[13], "six-DoF RK4 rollout did not consume propellant");
    const auto diagnostics = model.path_diagnostics(states, controls);
    require(diagnostics.quaternion_norm < 1.0e-8,
            "six-DoF path diagnostics rejected normalised attitudes");
    require(diagnostics.torque < 1.0e-9 && diagnostics.throttle_upper < 1.0e-9,
            "six-DoF path diagnostics rejected bounded controls");
}

}  // namespace

int main() {
    test_linearisation_and_rollout();
    return 0;
}
