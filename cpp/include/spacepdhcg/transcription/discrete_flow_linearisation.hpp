#pragma once

#include "spacepdhcg/transcription/discretisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace spacepdhcg::transcription {

template <std::size_t StateDimension, std::size_t ControlDimension>
struct DiscreteAffineLinearisation {
    std::array<double, StateDimension * StateDimension> state{};
    std::array<double, StateDimension * ControlDimension> control{};
    std::array<double, StateDimension> offset{};
};

namespace discrete_flow_detail {

template <typename Model, typename State, typename Control>
[[nodiscard]] State forward_euler_step(
    const Model& model,
    const State& state,
    const Control& control,
    double step_seconds
) {
    const auto derivative = model.dynamics(state, control);
    State next{};
    for (std::size_t component = 0; component < next.size(); ++component) {
        next[component] = state[component] + step_seconds * derivative[component];
    }
    return next;
}

template <typename Model, typename State, typename Control>
[[nodiscard]] State discrete_step(
    const Model& model,
    const State& state,
    const Control& control,
    double step_seconds,
    DiscretisationMethod method
) {
    switch (method) {
        case DiscretisationMethod::forward_euler:
            return forward_euler_step(model, state, control, step_seconds);
        case DiscretisationMethod::rk4_finite_difference:
            return model.rk4_step(state, control, step_seconds);
    }
    throw std::invalid_argument("unsupported discrete-flow method");
}

inline double perturbation(double value, double relative_step) {
    return relative_step * std::max(1.0, std::abs(value));
}

template <typename Vector>
void require_finite_vector(const Vector& values, const char* message) {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(message);
        }
    }
}

template <typename Vector, typename Output, typename Evaluator>
[[nodiscard]] Output finite_difference_column(
    const Vector& reference_vector,
    const Output& reference_output,
    std::size_t column,
    double delta,
    Evaluator&& evaluate
) {
    auto plus = reference_vector;
    auto minus = reference_vector;
    plus[column] += delta;
    minus[column] -= delta;

    std::optional<Output> plus_output{};
    std::optional<Output> minus_output{};
    try {
        plus_output = evaluate(plus);
    } catch (const std::invalid_argument&) {
    }
    try {
        minus_output = evaluate(minus);
    } catch (const std::invalid_argument&) {
    }

    Output derivative{};
    if (plus_output.has_value() && minus_output.has_value()) {
        for (std::size_t row = 0; row < derivative.size(); ++row) {
            derivative[row] = ((*plus_output)[row] - (*minus_output)[row]) / (2.0 * delta);
        }
        return derivative;
    }
    if (plus_output.has_value()) {
        for (std::size_t row = 0; row < derivative.size(); ++row) {
            derivative[row] = ((*plus_output)[row] - reference_output[row]) / delta;
        }
        return derivative;
    }
    if (minus_output.has_value()) {
        for (std::size_t row = 0; row < derivative.size(); ++row) {
            derivative[row] = (reference_output[row] - (*minus_output)[row]) / delta;
        }
        return derivative;
    }
    throw std::invalid_argument(
        "finite-difference perturbations are infeasible in both directions"
    );
}

}  // namespace discrete_flow_detail

/// Linearise the selected one-step map directly, preserving the fixed CQP topology.
///
/// The affine model is `x_next = A x + B u + d`. Central differences are used in the
/// interior. When a perturbation crosses a physical domain boundary, such as nonnegative
/// mass or thrust epigraph, the valid one-sided derivative is used. The offset is formed from
/// the exact selected step at the reference, so the affine model reproduces that reference
/// step to roundoff even when derivative columns are approximate.
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] DiscreteAffineLinearisation<StateDimension, ControlDimension>
linearise_discrete_flow(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    double step_seconds,
    DiscretisationMethod method,
    double relative_step = 1.0e-6
) {
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("discrete-flow step duration must be finite and positive");
    }
    if (!std::isfinite(relative_step) || relative_step <= 0.0) {
        throw std::invalid_argument("finite-difference relative step must be finite and positive");
    }
    discrete_flow_detail::require_finite_vector(state, "discrete-flow state must be finite");
    discrete_flow_detail::require_finite_vector(control, "discrete-flow control must be finite");

    const auto reference = discrete_flow_detail::discrete_step(
        model,
        state,
        control,
        step_seconds,
        method
    );
    DiscreteAffineLinearisation<StateDimension, ControlDimension> result{};

    for (std::size_t column = 0; column < StateDimension; ++column) {
        const auto delta = discrete_flow_detail::perturbation(state[column], relative_step);
        const auto derivative = discrete_flow_detail::finite_difference_column(
            state,
            reference,
            column,
            delta,
            [&](const auto& candidate) {
                return discrete_flow_detail::discrete_step(
                    model,
                    candidate,
                    control,
                    step_seconds,
                    method
                );
            }
        );
        for (std::size_t row = 0; row < StateDimension; ++row) {
            result.state[row * StateDimension + column] = derivative[row];
        }
    }

    for (std::size_t column = 0; column < ControlDimension; ++column) {
        const auto delta = discrete_flow_detail::perturbation(control[column], relative_step);
        const auto derivative = discrete_flow_detail::finite_difference_column(
            control,
            reference,
            column,
            delta,
            [&](const auto& candidate) {
                return discrete_flow_detail::discrete_step(
                    model,
                    state,
                    candidate,
                    step_seconds,
                    method
                );
            }
        );
        for (std::size_t row = 0; row < StateDimension; ++row) {
            result.control[row * ControlDimension + column] = derivative[row];
        }
    }

    result.offset = reference;
    for (std::size_t row = 0; row < StateDimension; ++row) {
        for (std::size_t column = 0; column < StateDimension; ++column) {
            result.offset[row] -= result.state[row * StateDimension + column] * state[column];
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            result.offset[row] -=
                result.control[row * ControlDimension + column] * control[column];
        }
    }
    return result;
}

}  // namespace spacepdhcg::transcription
