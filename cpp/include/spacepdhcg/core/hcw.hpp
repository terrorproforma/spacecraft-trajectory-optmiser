#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spacepdhcg::core {

inline constexpr std::size_t hcw_state_dimension = 6;
inline constexpr std::size_t hcw_control_dimension = 3;
using HCWStateMatrix = std::array<double, hcw_state_dimension * hcw_state_dimension>;
using HCWControlMatrix = std::array<double, hcw_state_dimension * hcw_control_dimension>;
using HCWState = std::array<double, hcw_state_dimension>;
using HCWControl = std::array<double, hcw_control_dimension>;

struct HCWDiscretisation {
    HCWStateMatrix state{};
    HCWControlMatrix control{};
};

[[nodiscard]] constexpr std::size_t hcw_state_index(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row * hcw_state_dimension + column;
}

[[nodiscard]] constexpr std::size_t hcw_control_index(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row * hcw_control_dimension + column;
}

/// Exact zero-order-hold HCW discretisation for constant acceleration over one interval.
[[nodiscard]] inline HCWDiscretisation discretise_hcw(
    const double mean_motion,
    const double step_seconds
) {
    if (!std::isfinite(mean_motion) || mean_motion <= 0.0) {
        throw std::invalid_argument("HCW mean motion must be finite and positive");
    }
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("HCW step duration must be finite and positive");
    }

    const double nt = mean_motion * step_seconds;
    const double sine = std::sin(nt);
    const double cosine = std::cos(nt);
    const double one_minus_cosine = 1.0 - cosine;
    const double inverse_motion = 1.0 / mean_motion;
    const double inverse_motion_squared = inverse_motion * inverse_motion;

    HCWDiscretisation result{};
    auto& a = result.state;
    a[hcw_state_index(0, 0)] = 4.0 - 3.0 * cosine;
    a[hcw_state_index(0, 3)] = sine * inverse_motion;
    a[hcw_state_index(0, 4)] = 2.0 * one_minus_cosine * inverse_motion;

    a[hcw_state_index(1, 0)] = 6.0 * (sine - nt);
    a[hcw_state_index(1, 1)] = 1.0;
    a[hcw_state_index(1, 3)] = -2.0 * one_minus_cosine * inverse_motion;
    a[hcw_state_index(1, 4)] = (4.0 * sine - 3.0 * nt) * inverse_motion;

    a[hcw_state_index(2, 2)] = cosine;
    a[hcw_state_index(2, 5)] = sine * inverse_motion;

    a[hcw_state_index(3, 0)] = 3.0 * mean_motion * sine;
    a[hcw_state_index(3, 3)] = cosine;
    a[hcw_state_index(3, 4)] = 2.0 * sine;

    a[hcw_state_index(4, 0)] = -6.0 * mean_motion * one_minus_cosine;
    a[hcw_state_index(4, 3)] = -2.0 * sine;
    a[hcw_state_index(4, 4)] = 4.0 * cosine - 3.0;

    a[hcw_state_index(5, 2)] = -mean_motion * sine;
    a[hcw_state_index(5, 5)] = cosine;

    auto& b = result.control;
    b[hcw_control_index(0, 0)] = one_minus_cosine * inverse_motion_squared;
    b[hcw_control_index(0, 1)] = 2.0 * (nt - sine) * inverse_motion_squared;

    b[hcw_control_index(1, 0)] = 2.0 * (sine - nt) * inverse_motion_squared;
    b[hcw_control_index(1, 1)] =
        (4.0 * one_minus_cosine - 1.5 * nt * nt) * inverse_motion_squared;

    b[hcw_control_index(2, 2)] = one_minus_cosine * inverse_motion_squared;

    b[hcw_control_index(3, 0)] = sine * inverse_motion;
    b[hcw_control_index(3, 1)] = 2.0 * one_minus_cosine * inverse_motion;

    b[hcw_control_index(4, 0)] = -2.0 * one_minus_cosine * inverse_motion;
    b[hcw_control_index(4, 1)] = (4.0 * sine - 3.0 * nt) * inverse_motion;

    b[hcw_control_index(5, 2)] = sine * inverse_motion;
    return result;
}

[[nodiscard]] inline HCWState propagate_hcw(
    const HCWDiscretisation& dynamics,
    const HCWState& state,
    const HCWControl& control
) noexcept {
    HCWState next{};
    for (std::size_t row = 0; row < hcw_state_dimension; ++row) {
        double value = 0.0;
        for (std::size_t column = 0; column < hcw_state_dimension; ++column) {
            value += dynamics.state[hcw_state_index(row, column)] * state[column];
        }
        for (std::size_t column = 0; column < hcw_control_dimension; ++column) {
            value += dynamics.control[hcw_control_index(row, column)] * control[column];
        }
        next[row] = value;
    }
    return next;
}

[[nodiscard]] inline HCWStateMatrix multiply_hcw_state_matrices(
    const HCWStateMatrix& left,
    const HCWStateMatrix& right
) noexcept {
    HCWStateMatrix product{};
    for (std::size_t row = 0; row < hcw_state_dimension; ++row) {
        for (std::size_t column = 0; column < hcw_state_dimension; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < hcw_state_dimension; ++inner) {
                value += left[hcw_state_index(row, inner)] * right[hcw_state_index(inner, column)];
            }
            product[hcw_state_index(row, column)] = value;
        }
    }
    return product;
}

[[nodiscard]] inline HCWControlMatrix compose_hcw_control_matrices(
    const HCWStateMatrix& later_state,
    const HCWControlMatrix& earlier_control,
    const HCWControlMatrix& later_control
) noexcept {
    HCWControlMatrix composed{};
    for (std::size_t row = 0; row < hcw_state_dimension; ++row) {
        for (std::size_t column = 0; column < hcw_control_dimension; ++column) {
            double value = later_control[hcw_control_index(row, column)];
            for (std::size_t inner = 0; inner < hcw_state_dimension; ++inner) {
                value += later_state[hcw_state_index(row, inner)] *
                    earlier_control[hcw_control_index(inner, column)];
            }
            composed[hcw_control_index(row, column)] = value;
        }
    }
    return composed;
}

}  // namespace spacepdhcg::core
