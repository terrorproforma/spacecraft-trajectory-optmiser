#pragma once

#include "spacepdhcg/transcription/linearisation_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spacepdhcg::transcription {

/// Discrete affine model of a free-final-time (time-dilated) RK4 step.
///
/// With the normalised time `tau` on `[0, 1]` and the dilation `sigma = t_f`, the dynamics read
/// `dx/dtau = sigma f(x, u)`.  One interval of length `d_tau` is integrated with RK4 (constant
/// control) and linearised about the reference `(x_k, u_k, sigma)` as
///
///     x_{k+1} = A x_k + B u_k + S sigma + z,
///
/// where `S = dF/dsigma` is the algorithmic sensitivity of the implemented RK4 map to the
/// dilation.  `offset` (`z`) is formed from the exact reference step so that the affine model
/// reproduces `propagated` at the reference to roundoff (affine reconstruction identity).
template <std::size_t StateDimension, std::size_t ControlDimension>
struct TimeDilatedLinearisation {
    std::array<double, StateDimension * StateDimension> state{};
    std::array<double, StateDimension * ControlDimension> control{};
    std::array<double, StateDimension> sigma{};
    std::array<double, StateDimension> offset{};
    std::array<double, StateDimension> propagated{};
};

namespace time_dilated_detail {

template <std::size_t StateDimension, std::size_t ControlDimension>
struct Augmented {
    std::array<double, StateDimension> state{};
    std::array<double, StateDimension * StateDimension> transition{};
    std::array<double, StateDimension * ControlDimension> control_sensitivity{};
    std::array<double, StateDimension> sigma_sensitivity{};
};

template <typename Vector>
void require_finite(const Vector& values, const char* message) {
    if (!std::all_of(values.begin(), values.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(message);
    }
}

/// Augmented derivative of the time-dilated variational system:
///
///     x'     = sigma f,
///     Phi'   = sigma f_x Phi,
///     Gamma' = sigma (f_x Gamma + f_u),
///     S'     = f + sigma f_x S.
///
/// Applying the same RK4 stages to all four blocks yields the exact derivatives of the
/// implemented RK4 map (forward-mode differentiation of an explicit Runge-Kutta scheme).
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] Augmented<StateDimension, ControlDimension> derivative(
    const Model& model,
    const Augmented<StateDimension, ControlDimension>& augmented,
    const std::array<double, ControlDimension>& control,
    const double sigma
) {
    const auto f = model.dynamics(augmented.state, control);
    const auto jacobians = model.jacobians(augmented.state, control);
    Augmented<StateDimension, ControlDimension> result{};
    for (std::size_t row = 0; row < StateDimension; ++row) {
        result.state[row] = sigma * f[row];
        double sigma_value = f[row];
        for (std::size_t inner = 0; inner < StateDimension; ++inner) {
            sigma_value += sigma * jacobians.state[row * StateDimension + inner]
                           * augmented.sigma_sensitivity[inner];
        }
        result.sigma_sensitivity[row] = sigma_value;
        for (std::size_t column = 0; column < StateDimension; ++column) {
            double value{0.0};
            for (std::size_t inner = 0; inner < StateDimension; ++inner) {
                value += jacobians.state[row * StateDimension + inner]
                         * augmented.transition[inner * StateDimension + column];
            }
            result.transition[row * StateDimension + column] = sigma * value;
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            double value = jacobians.control[row * ControlDimension + column];
            for (std::size_t inner = 0; inner < StateDimension; ++inner) {
                value += jacobians.state[row * StateDimension + inner]
                         * augmented.control_sensitivity[inner * ControlDimension + column];
            }
            result.control_sensitivity[row * ControlDimension + column] = sigma * value;
        }
    }
    require_finite(result.state, "time-dilated variational state derivative is non-finite");
    require_finite(
        result.transition,
        "time-dilated variational transition derivative is non-finite"
    );
    require_finite(
        result.control_sensitivity,
        "time-dilated variational control-sensitivity derivative is non-finite"
    );
    require_finite(
        result.sigma_sensitivity,
        "time-dilated variational dilation-sensitivity derivative is non-finite"
    );
    return result;
}

template <std::size_t StateDimension, std::size_t ControlDimension>
[[nodiscard]] Augmented<StateDimension, ControlDimension> add_scaled(
    const Augmented<StateDimension, ControlDimension>& base,
    const Augmented<StateDimension, ControlDimension>& increment,
    const double scale
) {
    Augmented<StateDimension, ControlDimension> result = base;
    for (std::size_t index = 0; index < StateDimension; ++index) {
        result.state[index] += scale * increment.state[index];
        result.sigma_sensitivity[index] += scale * increment.sigma_sensitivity[index];
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
constexpr bool has_quaternion_projection = requires(
    const Model& model,
    std::array<double, StateDimension>& state,
    std::array<double, StateDimension * StateDimension>& transition,
    std::array<double, StateDimension * ControlDimension>& control_sensitivity
) {
    project_rk4_variational(model, state, transition, control_sensitivity);
};

/// Differentiate the model's post-step quaternion normalisation through every sensitivity
/// block, including the dilation column.  The quaternion occupies rows 6..9 for the 6-DoF
/// powered-descent model; the projection `(I - q_hat q_hat^T)/||q||` is the exact Jacobian of
/// `q -> q/||q||`, so the projected columns are tangent to S^3 at the normalised output
/// (quaternion tangent rule).  The model's own hook then normalises the state and projects the
/// transition and control blocks with the identical matrix.
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
void apply_projection(
    const Model& model,
    Augmented<StateDimension, ControlDimension>& integrated
) {
    if constexpr (has_quaternion_projection<StateDimension, ControlDimension, Model>) {
        static_assert(StateDimension == 14U, "quaternion projection assumes the 6-DoF layout");
        constexpr std::size_t quaternion_offset = 6U;
        std::array<double, 4U> raw{};
        double norm_squared{0.0};
        for (std::size_t component = 0; component < 4U; ++component) {
            raw[component] = integrated.state[quaternion_offset + component];
            norm_squared += raw[component] * raw[component];
        }
        const auto norm = std::sqrt(norm_squared);
        if (!std::isfinite(norm) || norm <= 0.0) {
            throw std::runtime_error("time-dilated quaternion normalisation has a singular output");
        }
        std::array<double, 4U> projected{};
        for (std::size_t row = 0; row < 4U; ++row) {
            double value{0.0};
            for (std::size_t inner = 0; inner < 4U; ++inner) {
                const auto entry =
                    ((row == inner ? 1.0 : 0.0) - raw[row] * raw[inner] / norm_squared) / norm;
                value += entry * integrated.sigma_sensitivity[quaternion_offset + inner];
            }
            projected[row] = value;
        }
        for (std::size_t row = 0; row < 4U; ++row) {
            integrated.sigma_sensitivity[quaternion_offset + row] = projected[row];
        }
        project_rk4_variational(
            model,
            integrated.state,
            integrated.transition,
            integrated.control_sensitivity
        );
    }
}

}  // namespace time_dilated_detail

/// Exact reference map of one time-dilated interval: `substeps` RK4 steps of `sigma d_tau /
/// substeps` seconds through the model's public `rk4_step` (which includes any manifold
/// projection).  This is the map the linearisation differentiates.
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] std::array<double, StateDimension> time_dilated_step(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double sigma,
    const double d_tau,
    const std::size_t substeps = 1U
) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw std::invalid_argument("time dilation must be finite and positive");
    }
    if (!std::isfinite(d_tau) || d_tau <= 0.0) {
        throw std::invalid_argument("normalised interval must be finite and positive");
    }
    if (substeps == 0U) {
        throw std::invalid_argument("time-dilated step needs at least one substep");
    }
    const auto step_seconds = sigma * d_tau / static_cast<double>(substeps);
    auto next = state;
    for (std::size_t substep = 0; substep < substeps; ++substep) {
        next = model.rk4_step(next, control, step_seconds);
    }
    return next;
}

/// Linearise the time-dilated RK4 map about `(state, control, sigma)`.
///
/// Returns `A = dF/dx`, `B = dF/du`, `S = dF/dsigma`, the reference `propagated = F(x,u,sigma)`
/// and the affine offset `z = propagated - A x - B u - S sigma`.  All sensitivities are exact
/// algorithmic derivatives of `time_dilated_step` (same RK4 stages, same per-substep quaternion
/// normalisation for the 6-DoF model).
template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
[[nodiscard]] TimeDilatedLinearisation<StateDimension, ControlDimension>
linearise_time_dilated_flow(
    const Model& model,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double sigma,
    const double d_tau,
    const std::size_t substeps = 1U
) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw std::invalid_argument("time dilation must be finite and positive");
    }
    if (!std::isfinite(d_tau) || d_tau <= 0.0) {
        throw std::invalid_argument("normalised interval must be finite and positive");
    }
    if (substeps == 0U) {
        throw std::invalid_argument("time-dilated linearisation needs at least one substep");
    }
    time_dilated_detail::require_finite(state, "time-dilated state must be finite");
    time_dilated_detail::require_finite(control, "time-dilated control must be finite");

    using Augmented = time_dilated_detail::Augmented<StateDimension, ControlDimension>;
    Augmented current{};
    current.state = state;
    for (std::size_t index = 0; index < StateDimension; ++index) {
        current.transition[index * StateDimension + index] = 1.0;
    }
    const auto h = d_tau / static_cast<double>(substeps);
    for (std::size_t substep = 0; substep < substeps; ++substep) {
        const auto k1 = time_dilated_detail::derivative(model, current, control, sigma);
        const auto k2 = time_dilated_detail::derivative(
            model,
            time_dilated_detail::add_scaled(current, k1, 0.5 * h),
            control,
            sigma
        );
        const auto k3 = time_dilated_detail::derivative(
            model,
            time_dilated_detail::add_scaled(current, k2, 0.5 * h),
            control,
            sigma
        );
        const auto k4 = time_dilated_detail::derivative(
            model,
            time_dilated_detail::add_scaled(current, k3, h),
            control,
            sigma
        );
        const auto accumulate = [h](
                                    const double base,
                                    const double first,
                                    const double second,
                                    const double third,
                                    const double fourth
                                ) {
            return base + h * (first + 2.0 * second + 2.0 * third + fourth) / 6.0;
        };
        Augmented integrated = current;
        for (std::size_t index = 0; index < StateDimension; ++index) {
            integrated.state[index] = accumulate(
                current.state[index],
                k1.state[index],
                k2.state[index],
                k3.state[index],
                k4.state[index]
            );
            integrated.sigma_sensitivity[index] = accumulate(
                current.sigma_sensitivity[index],
                k1.sigma_sensitivity[index],
                k2.sigma_sensitivity[index],
                k3.sigma_sensitivity[index],
                k4.sigma_sensitivity[index]
            );
        }
        for (std::size_t index = 0; index < StateDimension * StateDimension; ++index) {
            integrated.transition[index] = accumulate(
                current.transition[index],
                k1.transition[index],
                k2.transition[index],
                k3.transition[index],
                k4.transition[index]
            );
        }
        for (std::size_t index = 0; index < StateDimension * ControlDimension; ++index) {
            integrated.control_sensitivity[index] = accumulate(
                current.control_sensitivity[index],
                k1.control_sensitivity[index],
                k2.control_sensitivity[index],
                k3.control_sensitivity[index],
                k4.control_sensitivity[index]
            );
        }
        time_dilated_detail::apply_projection(model, integrated);
        current = integrated;
    }

    TimeDilatedLinearisation<StateDimension, ControlDimension> result{};
    result.state = current.transition;
    result.control = current.control_sensitivity;
    result.sigma = current.sigma_sensitivity;
    // The model's public step is authoritative for the intercept (affine reconstruction).
    result.propagated = time_dilated_step<StateDimension, ControlDimension>(
        model, state, control, sigma, d_tau, substeps
    );
    result.offset = result.propagated;
    for (std::size_t row = 0; row < StateDimension; ++row) {
        for (std::size_t column = 0; column < StateDimension; ++column) {
            result.offset[row] -= result.state[row * StateDimension + column] * state[column];
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            result.offset[row] -= result.control[row * ControlDimension + column] * control[column];
        }
        result.offset[row] -= result.sigma[row] * sigma;
    }
    time_dilated_detail::require_finite(result.state, "time-dilated transition is non-finite");
    time_dilated_detail::require_finite(
        result.control,
        "time-dilated control sensitivity is non-finite"
    );
    time_dilated_detail::require_finite(result.sigma, "time-dilated sigma sensitivity is non-finite");
    time_dilated_detail::require_finite(result.offset, "time-dilated affine offset is non-finite");
    return result;
}

}  // namespace spacepdhcg::transcription
