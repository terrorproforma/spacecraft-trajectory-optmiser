#pragma once

#include "spacepdhcg/transcription/discrete_flow_linearisation.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spacepdhcg::transcription {

namespace variational_rk4_detail {

template <std::size_t StateDimension, std::size_t ControlDimension>
struct AugmentedState {
    std::array<double, StateDimension> state{};
    std::array<double, StateDimension * StateDimension> transition{};
    std::array<double, StateDimension * ControlDimension> control_sensitivity{};
};

template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] AugmentedState<StateDimension, ControlDimension> derivative(
    const Model& model,
    const AugmentedState<StateDimension, ControlDimension>& augmented,
    const std::array<double, ControlDimension>& control
) {
    const auto dynamics = model.dynamics(augmented.state, control);
    const auto jacobians = model.jacobians(augmented.state, control);
    AugmentedState<StateDimension, ControlDimension> result{};
    result.state = dynamics;

    for (std::size_t row = 0; row < StateDimension; ++row) {
        for (std::size_t column = 0; column < StateDimension; ++column) {
            double value{0.0};
            for (std::size_t inner = 0; inner < StateDimension; ++inner) {
                value += jacobians.state[row * StateDimension + inner]
                         * augmented.transition[inner * StateDimension + column];
            }
            result.transition[row * StateDimension + column] = value;
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            double value = jacobians.control[row * ControlDimension + column];
            for (std::size_t inner = 0; inner < StateDimension; ++inner) {
                value += jacobians.state[row * StateDimension + inner]
                         * augmented.control_sensitivity[
                             inner * ControlDimension + column
                         ];
            }
            result.control_sensitivity[row * ControlDimension + column] = value;
        }
    }
    return result;
}

template <std::size_t StateDimension, std::size_t ControlDimension>
[[nodiscard]] AugmentedState<StateDimension, ControlDimension> add_scaled(
    const AugmentedState<StateDimension, ControlDimension>& base,
    const AugmentedState<StateDimension, ControlDimension>& increment,
    double scale
) {
    AugmentedState<StateDimension, ControlDimension> result = base;
    for (std::size_t index = 0; index < StateDimension; ++index) {
        result.state[index] += scale * increment.state[index];
    }
    for (std::size_t index = 0; index < StateDimension * StateDimension; ++index) {
        result.transition[index] += scale * increment.transition[index];
    }
    for (std::size_t index = 0; index < StateDimension * ControlDimension; ++index) {
        result.control_sensitivity[index] += scale * increment.control_sensitivity[index];
    }
    return result;
}

}  // namespace variational_rk4_detail

/// Integrate the state-transition and constant-control sensitivity equations with RK4.
///
/// This avoids one dynamics rollout per Jacobian column for models whose state lives in a
/// Euclidean coordinate chart and whose `jacobians()` method is analytic. Models that apply a
/// post-step manifold projection, such as quaternion normalisation, must additionally apply the
/// projection Jacobian or retain finite-difference discrete-flow linearisation.
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] DiscreteAffineLinearisation<StateDimension, ControlDimension>
linearise_rk4_variational(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    double step_seconds
) {
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("variational RK4 step must be finite and positive");
    }
    discrete_flow_detail::require_finite_vector(state, "variational RK4 state must be finite");
    discrete_flow_detail::require_finite_vector(
        control,
        "variational RK4 control must be finite"
    );

    using Augmented = variational_rk4_detail::AugmentedState<
        StateDimension,
        ControlDimension
    >;
    Augmented initial{};
    initial.state = state;
    for (std::size_t index = 0; index < StateDimension; ++index) {
        initial.transition[index * StateDimension + index] = 1.0;
    }

    const auto k1 = variational_rk4_detail::derivative(model, initial, control);
    const auto k2 = variational_rk4_detail::derivative(
        model,
        variational_rk4_detail::add_scaled(initial, k1, 0.5 * step_seconds),
        control
    );
    const auto k3 = variational_rk4_detail::derivative(
        model,
        variational_rk4_detail::add_scaled(initial, k2, 0.5 * step_seconds),
        control
    );
    const auto k4 = variational_rk4_detail::derivative(
        model,
        variational_rk4_detail::add_scaled(initial, k3, step_seconds),
        control
    );

    Augmented integrated = initial;
    const auto accumulate = [step_seconds](
                                double initial_value,
                                double first,
                                double second,
                                double third,
                                double fourth
                            ) {
        return initial_value
               + step_seconds * (first + 2.0 * second + 2.0 * third + fourth) / 6.0;
    };
    for (std::size_t index = 0; index < StateDimension; ++index) {
        integrated.state[index] = accumulate(
            initial.state[index],
            k1.state[index],
            k2.state[index],
            k3.state[index],
            k4.state[index]
        );
    }
    for (std::size_t index = 0; index < StateDimension * StateDimension; ++index) {
        integrated.transition[index] = accumulate(
            initial.transition[index],
            k1.transition[index],
            k2.transition[index],
            k3.transition[index],
            k4.transition[index]
        );
    }
    for (std::size_t index = 0; index < StateDimension * ControlDimension; ++index) {
        integrated.control_sensitivity[index] = accumulate(
            initial.control_sensitivity[index],
            k1.control_sensitivity[index],
            k2.control_sensitivity[index],
            k3.control_sensitivity[index],
            k4.control_sensitivity[index]
        );
    }

    DiscreteAffineLinearisation<StateDimension, ControlDimension> result{};
    result.state = integrated.transition;
    result.control = integrated.control_sensitivity;
    result.offset = integrated.state;
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
