#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spacepdhcg::dynamics {

inline constexpr std::size_t hcw_state_dimension = 6U;
inline constexpr std::size_t hcw_control_dimension = 3U;

using HcwState = std::array<double, hcw_state_dimension>;
using HcwControl = std::array<double, hcw_control_dimension>;
using HcwStateMatrix = std::array<double, hcw_state_dimension * hcw_state_dimension>;
using HcwControlMatrix = std::array<double, hcw_state_dimension * hcw_control_dimension>;

struct HcwDiscreteDynamics {
    HcwStateMatrix state{};
    HcwControlMatrix control{};
};

/// Exact zero-order-hold Hill-Clohessy-Wiltshire transition for constant LVLH acceleration.
[[nodiscard]] inline HcwDiscreteDynamics discretise_hcw(
    double mean_motion,
    double step_seconds
) {
    if (!std::isfinite(mean_motion) || mean_motion <= 0.0) {
        throw std::invalid_argument("HCW mean motion must be finite and positive");
    }
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("HCW step duration must be finite and positive");
    }

    const auto n = mean_motion;
    const auto t = step_seconds;
    const auto nt = n * t;
    const auto c = std::cos(nt);
    const auto s = std::sin(nt);
    const auto inv_n = 1.0 / n;
    const auto inv_n2 = inv_n * inv_n;

    HcwDiscreteDynamics result{};
    const auto a = [&result](std::size_t row, std::size_t column) -> double& {
        return result.state[row * hcw_state_dimension + column];
    };
    const auto b = [&result](std::size_t row, std::size_t column) -> double& {
        return result.control[row * hcw_control_dimension + column];
    };

    a(0U, 0U) = 4.0 - 3.0 * c;
    a(0U, 3U) = s * inv_n;
    a(0U, 4U) = 2.0 * (1.0 - c) * inv_n;

    a(1U, 0U) = 6.0 * (s - nt);
    a(1U, 1U) = 1.0;
    a(1U, 3U) = -2.0 * (1.0 - c) * inv_n;
    a(1U, 4U) = (4.0 * s - 3.0 * nt) * inv_n;

    a(2U, 2U) = c;
    a(2U, 5U) = s * inv_n;

    a(3U, 0U) = 3.0 * n * s;
    a(3U, 3U) = c;
    a(3U, 4U) = 2.0 * s;

    a(4U, 0U) = -6.0 * n * (1.0 - c);
    a(4U, 3U) = -2.0 * s;
    a(4U, 4U) = 4.0 * c - 3.0;

    a(5U, 2U) = -n * s;
    a(5U, 5U) = c;

    b(0U, 0U) = (1.0 - c) * inv_n2;
    b(0U, 1U) = 2.0 * (nt - s) * inv_n2;

    b(1U, 0U) = 2.0 * (s - nt) * inv_n2;
    b(1U, 1U) = 4.0 * (1.0 - c) * inv_n2 - 1.5 * t * t;

    b(2U, 2U) = (1.0 - c) * inv_n2;

    b(3U, 0U) = s * inv_n;
    b(3U, 1U) = 2.0 * (1.0 - c) * inv_n;

    b(4U, 0U) = -2.0 * (1.0 - c) * inv_n;
    b(4U, 1U) = 4.0 * s * inv_n - 3.0 * t;

    b(5U, 2U) = s * inv_n;
    return result;
}

[[nodiscard]] inline HcwState hcw_step(
    const HcwDiscreteDynamics& dynamics,
    const HcwState& state,
    const HcwControl& control
) {
    HcwState next{};
    for (std::size_t row = 0; row < hcw_state_dimension; ++row) {
        for (std::size_t column = 0; column < hcw_state_dimension; ++column) {
            next[row] += dynamics.state[row * hcw_state_dimension + column] * state[column];
        }
        for (std::size_t column = 0; column < hcw_control_dimension; ++column) {
            next[row] += dynamics.control[row * hcw_control_dimension + column]
                         * control[column];
        }
    }
    return next;
}

}  // namespace spacepdhcg::dynamics
