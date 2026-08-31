#include "spacepdhcg/native/cw.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spacepdhcg::native {
namespace {

template <std::size_t Columns, typename Matrix>
double& entry(Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

template <std::size_t Columns, typename Matrix>
const double& entry(const Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

[[nodiscard]] double one_minus_cosine(double angle) {
    const double half_sine = std::sin(0.5 * angle);
    return 2.0 * half_sine * half_sine;
}

[[nodiscard]] double four_one_minus_cosine_minus_three_halves_square(double angle) {
    const double magnitude = std::abs(angle);
    if (magnitude < 1.0e-3) {
        const double square = angle * angle;
        return square * (0.5 + square * (-1.0 / 6.0 + square / 180.0));
    }
    return 4.0 * one_minus_cosine(angle) - 1.5 * angle * angle;
}

}  // namespace

CwDiscreteDynamics discretise_cw(double mean_motion, double step_seconds) {
    if (!std::isfinite(mean_motion) || mean_motion <= 0.0) {
        throw std::invalid_argument("mean motion must be finite and positive");
    }
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("step duration must be finite and positive");
    }

    const double n = mean_motion;
    const double time = step_seconds;
    const double angle = n * time;
    const double sine = std::sin(angle);
    const double cosine = std::cos(angle);
    const double one_minus_cosine_value = one_minus_cosine(angle);
    const double inverse_n = 1.0 / n;
    const double inverse_n_squared = inverse_n * inverse_n;

    CwDiscreteDynamics result{};
    auto& state = result.state;
    auto& control = result.control;

    entry<cw_state_dimension>(state, 0, 0) = 4.0 - 3.0 * cosine;
    entry<cw_state_dimension>(state, 0, 3) = sine * inverse_n;
    entry<cw_state_dimension>(state, 0, 4) = 2.0 * one_minus_cosine_value * inverse_n;

    entry<cw_state_dimension>(state, 1, 0) = 6.0 * (sine - angle);
    entry<cw_state_dimension>(state, 1, 1) = 1.0;
    entry<cw_state_dimension>(state, 1, 3) = -2.0 * one_minus_cosine_value * inverse_n;
    entry<cw_state_dimension>(state, 1, 4) = (4.0 * sine - 3.0 * angle) * inverse_n;

    entry<cw_state_dimension>(state, 2, 2) = cosine;
    entry<cw_state_dimension>(state, 2, 5) = sine * inverse_n;

    entry<cw_state_dimension>(state, 3, 0) = 3.0 * n * sine;
    entry<cw_state_dimension>(state, 3, 3) = cosine;
    entry<cw_state_dimension>(state, 3, 4) = 2.0 * sine;

    entry<cw_state_dimension>(state, 4, 0) = -6.0 * n * one_minus_cosine_value;
    entry<cw_state_dimension>(state, 4, 3) = -2.0 * sine;
    entry<cw_state_dimension>(state, 4, 4) = 4.0 * cosine - 3.0;

    entry<cw_state_dimension>(state, 5, 2) = -n * sine;
    entry<cw_state_dimension>(state, 5, 5) = cosine;

    entry<cw_control_dimension>(control, 0, 0) =
        one_minus_cosine_value * inverse_n_squared;
    entry<cw_control_dimension>(control, 0, 1) =
        2.0 * (angle - sine) * inverse_n_squared;

    entry<cw_control_dimension>(control, 1, 0) =
        2.0 * (sine - angle) * inverse_n_squared;
    entry<cw_control_dimension>(control, 1, 1) =
        four_one_minus_cosine_minus_three_halves_square(angle) * inverse_n_squared;

    entry<cw_control_dimension>(control, 2, 2) =
        one_minus_cosine_value * inverse_n_squared;

    entry<cw_control_dimension>(control, 3, 0) = sine * inverse_n;
    entry<cw_control_dimension>(control, 3, 1) =
        2.0 * one_minus_cosine_value * inverse_n;

    entry<cw_control_dimension>(control, 4, 0) =
        -2.0 * one_minus_cosine_value * inverse_n;
    entry<cw_control_dimension>(control, 4, 1) = (4.0 * sine - 3.0 * angle) * inverse_n;

    entry<cw_control_dimension>(control, 5, 2) = sine * inverse_n;

    return result;
}

CwState propagate_cw(
    const CwDiscreteDynamics& dynamics,
    std::span<const double, cw_state_dimension> state,
    std::span<const double, cw_control_dimension> control
) {
    CwState result{};
    for (std::size_t row = 0; row < cw_state_dimension; ++row) {
        for (std::size_t column = 0; column < cw_state_dimension; ++column) {
            result[row] += entry<cw_state_dimension>(dynamics.state, row, column) * state[column];
        }
        for (std::size_t column = 0; column < cw_control_dimension; ++column) {
            result[row] +=
                entry<cw_control_dimension>(dynamics.control, row, column) * control[column];
        }
    }
    return result;
}

CwStateMatrix multiply_state_matrices(
    const CwStateMatrix& left,
    const CwStateMatrix& right
) {
    CwStateMatrix result{};
    for (std::size_t row = 0; row < cw_state_dimension; ++row) {
        for (std::size_t column = 0; column < cw_state_dimension; ++column) {
            for (std::size_t inner = 0; inner < cw_state_dimension; ++inner) {
                entry<cw_state_dimension>(result, row, column) +=
                    entry<cw_state_dimension>(left, row, inner) *
                    entry<cw_state_dimension>(right, inner, column);
            }
        }
    }
    return result;
}

CwControlMatrix compose_control_matrices(
    const CwStateMatrix& later_state,
    const CwControlMatrix& earlier_control,
    const CwControlMatrix& later_control
) {
    CwControlMatrix result = later_control;
    for (std::size_t row = 0; row < cw_state_dimension; ++row) {
        for (std::size_t column = 0; column < cw_control_dimension; ++column) {
            for (std::size_t inner = 0; inner < cw_state_dimension; ++inner) {
                entry<cw_control_dimension>(result, row, column) +=
                    entry<cw_state_dimension>(later_state, row, inner) *
                    entry<cw_control_dimension>(earlier_control, inner, column);
            }
        }
    }
    return result;
}

double maximum_absolute_difference(
    std::span<const double> left,
    std::span<const double> right
) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("difference operands must have equal length");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace spacepdhcg::native
