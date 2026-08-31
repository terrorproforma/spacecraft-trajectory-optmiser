#pragma once

#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spacepdhcg::dynamics {

/// Apply the exact Jacobian of quaternion normalisation to RK4 sensitivities.
///
/// For N(q)=q/||q||,
///
///     DN(q) = (I - q_hat q_hat^T) / ||q||.
///
/// Only the four quaternion output rows are transformed. This makes the returned state and
/// control sensitivities derivatives of the actual `PoweredDescent6DofModel::rk4_step`, whose
/// final operation is quaternion normalisation. The transformed quaternion columns are tangent
/// to S^3 at the normalised output.
inline void project_rk4_variational(
    const PoweredDescent6DofModel&,
    PoweredDescent6DofState& state,
    Matrix<14U, 14U>& transition,
    Matrix<14U, 7U>& control_sensitivity
) {
    const Quaternion raw{
        state[6U],
        state[7U],
        state[8U],
        state[9U],
    };
    const auto norm = std::sqrt(
        raw[0U] * raw[0U] + raw[1U] * raw[1U]
        + raw[2U] * raw[2U] + raw[3U] * raw[3U]
    );
    if (!std::isfinite(norm) || norm <= 0.0) {
        throw std::runtime_error(
            "variational quaternion normalisation has a singular output"
        );
    }
    Quaternion normalised{};
    for (std::size_t component = 0; component < 4U; ++component) {
        normalised[component] = raw[component] / norm;
        state[6U + component] = normalised[component];
    }

    std::array<double, 16U> projection{};
    for (std::size_t row = 0; row < 4U; ++row) {
        for (std::size_t column = 0; column < 4U; ++column) {
            projection[row * 4U + column] =
                ((row == column ? 1.0 : 0.0)
                 - normalised[row] * normalised[column])
                / norm;
        }
    }

    const auto raw_transition = transition;
    const auto raw_control = control_sensitivity;
    for (std::size_t output = 0; output < 4U; ++output) {
        for (std::size_t column = 0; column < 14U; ++column) {
            double value{0.0};
            for (std::size_t inner = 0; inner < 4U; ++inner) {
                value += projection[output * 4U + inner]
                         * raw_transition[(6U + inner) * 14U + column];
            }
            transition[(6U + output) * 14U + column] = value;
        }
        for (std::size_t column = 0; column < 7U; ++column) {
            double value{0.0};
            for (std::size_t inner = 0; inner < 4U; ++inner) {
                value += projection[output * 4U + inner]
                         * raw_control[(6U + inner) * 7U + column];
            }
            control_sensitivity[(6U + output) * 7U + column] = value;
        }
    }
}

}  // namespace spacepdhcg::dynamics
