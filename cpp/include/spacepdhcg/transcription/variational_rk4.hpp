#pragma once

#include "spacepdhcg/dynamics/powered_descent_6dof_variational.hpp"
#include "spacepdhcg/transcription/linearisation_types.hpp"

#include <algorithm>
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

template <typename Vector>
void require_finite(const Vector& values, const char* message) {
    if (!std::all_of(values.begin(), values.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(message);
    }
}

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
    require_finite(result.state, "variational RK4 state derivative is non-finite");
    require_finite(
        result.transition,
        "variational RK4 state-transition derivative is non-finite"
    );
    require_finite(
        result.control_sensitivity,
        "variational RK4 control-sensitivity derivative is non-finite"
    );
    return result;
}

template <std::size_t StateDimension, std::size_t ControlDimension>
[[nodiscard]] AugmentedState<StateDimension, ControlDimension> add_scaled(
    const AugmentedState<StateDimension, ControlDimension>& base,
    const AugmentedState<StateDimension, ControlDimension>& increment,
    const double scale
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

template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
void apply_post_step_projection(
    const Model& model,
    AugmentedState<StateDimension, ControlDimension>& integrated
) {
    if constexpr (requires {
                      project_rk4_variational(
                          model,
                          integrated.state,
                          integrated.transition,
                          integrated.control_sensitivity
                      );
                  }) {
        // Deliberately unqualified: argument-dependent lookup selects the model's manifold
        // projection derivative without coupling Euclidean models to a projection interface.
        project_rk4_variational(
            model,
            integrated.state,
            integrated.transition,
            integrated.control_sensitivity
        );
    } else if constexpr (requires {
                             model.project_rk4_variational(
                                 integrated.state,
                                 integrated.transition,
                                 integrated.control_sensitivity
                             );
                         }) {
        model.project_rk4_variational(
            integrated.state,
            integrated.transition,
            integrated.control_sensitivity
        );
    }
}

}  // namespace variational_rk4_detail

/// Integrate the one-step map and its exact RK4 algorithmic sensitivities.
///
/// For constant control over one interval, the augmented variational equations are
///
///     x_dot     = f(x,u),
///     Phi_dot   = f_x(x,u) Phi,             Phi(0) = I,
///     Gamma_dot = f_x(x,u) Gamma + f_u(x,u), Gamma(0) = 0.
///
/// The same four RK4 stages are applied to all three blocks. This gives derivatives of the
/// implemented RK4 map, not merely of the continuous flow. A model may expose an ADL-visible
/// `project_rk4_variational(model,state,Phi,Gamma)` or an equivalent member to differentiate a
/// deterministic post-step manifold projection. The 6-DoF powered-descent model uses this hook
/// for quaternion normalisation; Euclidean models need no hook.
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] DiscreteAffineLinearisation<StateDimension, ControlDimension>
linearise_rk4_variational(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double step_seconds
) {
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("variational RK4 step must be finite and positive");
    }
    variational_rk4_detail::require_finite(
        state,
        "variational RK4 state must be finite"
    );
    variational_rk4_detail::require_finite(
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

    const auto accumulate = [step_seconds](
                                const double initial_value,
                                const double first,
                                const double second,
                                const double third,
                                const double fourth
                            ) {
        return initial_value
               + step_seconds * (first + 2.0 * second + 2.0 * third + fourth) / 6.0;
    };
    Augmented integrated = initial;
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

    variational_rk4_detail::apply_post_step_projection(model, integrated);
    variational_rk4_detail::require_finite(
        integrated.transition,
        "variational RK4 state-transition matrix is non-finite"
    );
    variational_rk4_detail::require_finite(
        integrated.control_sensitivity,
        "variational RK4 control-sensitivity matrix is non-finite"
    );

    // The model's public step is authoritative for the affine intercept. This guarantees that
    // the linear model reproduces the exact implemented reference step to roundoff, including
    // any model-specific post-step projection.
    const auto reference = model.rk4_step(state, control, step_seconds);
    DiscreteAffineLinearisation<StateDimension, ControlDimension> result{};
    result.state = integrated.transition;
    result.control = integrated.control_sensitivity;
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
    variational_rk4_detail::require_finite(
        result.offset,
        "variational RK4 affine offset is non-finite"
    );
    return result;
}

}  // namespace spacepdhcg::transcription
