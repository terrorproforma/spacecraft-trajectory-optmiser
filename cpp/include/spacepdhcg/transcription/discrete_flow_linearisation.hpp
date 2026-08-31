#pragma once

#include "spacepdhcg/transcription/discretisation.hpp"
#include "spacepdhcg/transcription/linearisation_types.hpp"
#include "spacepdhcg/transcription/variational_rk4.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace spacepdhcg::transcription {

namespace discrete_flow_detail {

template <typename Model, typename State, typename Control>
[[nodiscard]] State forward_euler_step(
    const Model& model,
    const State& state,
    const Control& control,
    const double step_seconds
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
    const double step_seconds,
    const DiscretisationMethod method
) {
    switch (method) {
        case DiscretisationMethod::forward_euler:
            return forward_euler_step(model, state, control, step_seconds);
        case DiscretisationMethod::rk4_finite_difference:
        case DiscretisationMethod::rk4_variational:
            return model.rk4_step(state, control, step_seconds);
    }
    throw std::invalid_argument("unsupported discrete-flow method");
}

inline double perturbation(const double value, const double relative_step) {
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
    const std::size_t column,
    const double delta,
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

template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] DiscreteAffineLinearisation<StateDimension, ControlDimension>
linearise_by_finite_difference(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double step_seconds,
    const DiscretisationMethod method,
    const double relative_step
) {
    const auto reference = discrete_step(
        model,
        state,
        control,
        step_seconds,
        method
    );
    DiscreteAffineLinearisation<StateDimension, ControlDimension> result{};

    for (std::size_t column = 0; column < StateDimension; ++column) {
        const auto delta = perturbation(state[column], relative_step);
        const auto derivative = finite_difference_column(
            state,
            reference,
            column,
            delta,
            [&](const auto& candidate) {
                return discrete_step(
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
        const auto delta = perturbation(control[column], relative_step);
        const auto derivative = finite_difference_column(
            control,
            reference,
            column,
            delta,
            [&](const auto& candidate) {
                return discrete_step(
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

}  // namespace discrete_flow_detail

/// Linearise the selected one-step map while preserving fixed CQP topology.
///
/// - `forward_euler` and `rk4_finite_difference` use domain-aware finite differences.
/// - `rk4_variational` integrates analytic state-transition and constant-control sensitivity
///   equations with the same RK4 stages as the state. A model-specific post-step projection
///   Jacobian is applied when exposed by the model.
///
/// The affine model is `x_next = A x + B u + d`. Finite differences use central columns in the
/// interior and a valid one-sided column when a perturbation crosses a physical domain boundary.
/// Every mode forms the offset from the exact selected step at the reference, so the affine model
/// reproduces the implemented reference step to roundoff.
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] DiscreteAffineLinearisation<StateDimension, ControlDimension>
linearise_discrete_flow(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double step_seconds,
    const DiscretisationMethod method,
    const double relative_step = 1.0e-6
) {
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("discrete-flow step duration must be finite and positive");
    }
    discrete_flow_detail::require_finite_vector(
        state,
        "discrete-flow state must be finite"
    );
    discrete_flow_detail::require_finite_vector(
        control,
        "discrete-flow control must be finite"
    );
    if (method == DiscretisationMethod::rk4_variational) {
        return linearise_rk4_variational<StateDimension, ControlDimension>(
            model,
            state,
            control,
            step_seconds
        );
    }
    if (!std::isfinite(relative_step) || relative_step <= 0.0) {
        throw std::invalid_argument(
            "finite-difference relative step must be finite and positive"
        );
    }
    return discrete_flow_detail::linearise_by_finite_difference<
        StateDimension,
        ControlDimension
    >(
        model,
        state,
        control,
        step_seconds,
        method,
        relative_step
    );
}

}  // namespace spacepdhcg::transcription
