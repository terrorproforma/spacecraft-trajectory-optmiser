#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::dynamics {

inline constexpr std::size_t powered_descent_state_dimension = 7U;
inline constexpr std::size_t powered_descent_control_dimension = 4U;

using PoweredDescentState = std::array<double, powered_descent_state_dimension>;
using PoweredDescentControl = std::array<double, powered_descent_control_dimension>;
using ThrustVector = std::array<double, 3U>;

template <std::size_t Rows, std::size_t Columns>
using StaticMatrix = std::array<double, Rows * Columns>;

struct PoweredDescent3DofConfig {
    ThrustVector gravity{0.0, 0.0, -3.711};
    double mass_flow_coefficient{4.6e-4};
    double minimum_mass{1'000.0};
    double maximum_thrust{15'000.0};
    double minimum_sigma{0.0};
    double maximum_tilt_radians{0.5235987755982988};
    double glide_slope_radians{1.0471975511965976};

    void validate() const {
        for (const auto component : gravity) {
            require_finite(component, "gravity components must be finite");
        }
        require_positive(mass_flow_coefficient, "mass-flow coefficient must be positive");
        require_positive(minimum_mass, "minimum mass must be positive");
        require_positive(maximum_thrust, "maximum thrust must be positive");
        require_finite(minimum_sigma, "minimum sigma must be finite");
        if (minimum_sigma < 0.0 || minimum_sigma > maximum_thrust) {
            throw std::invalid_argument("minimum sigma must lie in [0, maximum thrust]");
        }
        constexpr double half_pi = 1.5707963267948966;
        if (!(maximum_tilt_radians > 0.0 && maximum_tilt_radians < half_pi)) {
            throw std::invalid_argument("maximum tilt must lie in (0, pi/2)");
        }
        if (!(glide_slope_radians > 0.0 && glide_slope_radians < half_pi)) {
            throw std::invalid_argument("glide slope must lie in (0, pi/2)");
        }
    }

    [[nodiscard]] double tilt_cosine() const noexcept {
        return std::cos(maximum_tilt_radians);
    }

    [[nodiscard]] double glide_slope_tangent() const noexcept {
        return std::tan(glide_slope_radians);
    }

  private:
    static void require_finite(double value, const char* message) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(message);
        }
    }

    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
};

struct PoweredDescentJacobians {
    StaticMatrix<powered_descent_state_dimension, powered_descent_state_dimension> state{};
    StaticMatrix<powered_descent_state_dimension, powered_descent_control_dimension> control{};
};

struct PoweredDescentAffineLinearisation {
    PoweredDescentJacobians jacobians{};
    PoweredDescentState offset{};
};

struct PoweredDescentEulerLinearisation {
    StaticMatrix<powered_descent_state_dimension, powered_descent_state_dimension> state{};
    StaticMatrix<powered_descent_state_dimension, powered_descent_control_dimension> control{};
    PoweredDescentState offset{};
};

struct PoweredDescentPathDiagnostics {
    double thrust_epigraph{0.0};
    double throttle_lower{0.0};
    double throttle_upper{0.0};
    double tilt{0.0};
    double minimum_mass{0.0};
    double altitude{0.0};
    double glide_slope{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max(
            {thrust_epigraph,
             throttle_lower,
             throttle_upper,
             tilt,
             minimum_mass,
             altitude,
             glide_slope}
        );
    }

    [[nodiscard]] bool feasible(double tolerance = 1.0e-8) const {
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument("feasibility tolerance must be finite and non-negative");
        }
        return maximum_violation() <= tolerance;
    }
};

class PoweredDescent3DofModel {
  public:
    explicit PoweredDescent3DofModel(PoweredDescent3DofConfig config = {})
        : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const PoweredDescent3DofConfig& config() const noexcept { return config_; }

    [[nodiscard]] PoweredDescentState dynamics(
        const PoweredDescentState& state,
        const PoweredDescentControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        const auto mass = state[6U];
        PoweredDescentState derivative{};
        derivative[0U] = state[3U];
        derivative[1U] = state[4U];
        derivative[2U] = state[5U];
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            derivative[3U + axis] = control[axis] / mass + config_.gravity[axis];
        }
        derivative[6U] = -config_.mass_flow_coefficient * control[3U];
        return derivative;
    }

    [[nodiscard]] PoweredDescentJacobians jacobians(
        const PoweredDescentState& state,
        const PoweredDescentControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        const auto mass = state[6U];
        PoweredDescentJacobians result{};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            result.state[index<7U, 7U>(axis, 3U + axis)] = 1.0;
            result.state[index<7U, 7U>(3U + axis, 6U)] = -control[axis] / (mass * mass);
            result.control[index<7U, 4U>(3U + axis, axis)] = 1.0 / mass;
        }
        result.control[index<7U, 4U>(6U, 3U)] = -config_.mass_flow_coefficient;
        return result;
    }

    [[nodiscard]] PoweredDescentAffineLinearisation affine_linearisation(
        const PoweredDescentState& state,
        const PoweredDescentControl& control
    ) const {
        const auto derivative = dynamics(state, control);
        const auto derivatives = jacobians(state, control);
        PoweredDescentAffineLinearisation result{derivatives, derivative};
        for (std::size_t row = 0; row < 7U; ++row) {
            for (std::size_t column = 0; column < 7U; ++column) {
                result.offset[row] -= derivatives.state[index<7U, 7U>(row, column)]
                                      * state[column];
            }
            for (std::size_t column = 0; column < 4U; ++column) {
                result.offset[row] -= derivatives.control[index<7U, 4U>(row, column)]
                                      * control[column];
            }
        }
        return result;
    }

    [[nodiscard]] PoweredDescentEulerLinearisation linearised_euler_dynamics(
        const PoweredDescentState& state,
        const PoweredDescentControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto continuous = affine_linearisation(state, control);
        PoweredDescentEulerLinearisation result{};
        for (std::size_t row = 0; row < 7U; ++row) {
            for (std::size_t column = 0; column < 7U; ++column) {
                result.state[index<7U, 7U>(row, column)] =
                    (row == column ? 1.0 : 0.0)
                    + step_seconds
                          * continuous.jacobians.state[index<7U, 7U>(row, column)];
            }
            for (std::size_t column = 0; column < 4U; ++column) {
                result.control[index<7U, 4U>(row, column)] =
                    step_seconds
                    * continuous.jacobians.control[index<7U, 4U>(row, column)];
            }
            result.offset[row] = step_seconds * continuous.offset[row];
        }
        return result;
    }

    [[nodiscard]] PoweredDescentState euler_step(
        const PoweredDescentState& state,
        const PoweredDescentControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto derivative = dynamics(state, control);
        PoweredDescentState next{};
        for (std::size_t index_value = 0; index_value < next.size(); ++index_value) {
            next[index_value] = state[index_value] + step_seconds * derivative[index_value];
        }
        validate_state(next);
        return next;
    }

    [[nodiscard]] PoweredDescentState rk4_step(
        const PoweredDescentState& state,
        const PoweredDescentControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto k1 = dynamics(state, control);
        const auto k2 = dynamics(add_scaled(state, k1, 0.5 * step_seconds), control);
        const auto k3 = dynamics(add_scaled(state, k2, 0.5 * step_seconds), control);
        const auto k4 = dynamics(add_scaled(state, k3, step_seconds), control);
        PoweredDescentState next{};
        for (std::size_t index_value = 0; index_value < next.size(); ++index_value) {
            next[index_value] = state[index_value]
                                + step_seconds
                                      * (k1[index_value] + 2.0 * k2[index_value]
                                         + 2.0 * k3[index_value] + k4[index_value])
                                      / 6.0;
        }
        validate_state(next);
        return next;
    }

    [[nodiscard]] std::vector<PoweredDescentState> rollout(
        const PoweredDescentState& initial_state,
        const std::vector<PoweredDescentControl>& controls,
        double step_seconds,
        bool use_rk4 = true
    ) const {
        validate_state(initial_state);
        require_step(step_seconds);
        std::vector<PoweredDescentState> states(controls.size() + 1U);
        states.front() = initial_state;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            states[interval + 1U] = use_rk4
                                       ? rk4_step(states[interval], controls[interval], step_seconds)
                                       : euler_step(states[interval], controls[interval], step_seconds);
        }
        return states;
    }

    [[nodiscard]] ThrustVector project_thrust(const ThrustVector& requested) const {
        for (const auto component : requested) {
            if (!std::isfinite(component)) {
                throw std::invalid_argument("requested thrust must be finite");
            }
        }
        auto thrust = requested;
        if (thrust[2U] <= 0.0) {
            thrust = ThrustVector{0.0, 0.0, std::max(config_.minimum_sigma, 1.0)};
        }
        const auto horizontal = std::hypot(thrust[0U], thrust[1U]);
        const auto maximum_horizontal = thrust[2U] * std::tan(config_.maximum_tilt_radians);
        if (horizontal > maximum_horizontal && horizontal > 0.0) {
            const auto scale = maximum_horizontal / horizontal;
            thrust[0U] *= scale;
            thrust[1U] *= scale;
        }
        auto magnitude = norm3(thrust);
        if (magnitude > config_.maximum_thrust) {
            const auto scale = config_.maximum_thrust / magnitude;
            for (auto& component : thrust) {
                component *= scale;
            }
            magnitude = config_.maximum_thrust;
        }
        if (magnitude < config_.minimum_sigma) {
            thrust = ThrustVector{0.0, 0.0, config_.minimum_sigma};
        }
        return thrust;
    }

    [[nodiscard]] PoweredDescentPathDiagnostics path_diagnostics(
        const std::vector<PoweredDescentState>& states,
        const std::vector<PoweredDescentControl>& controls
    ) const {
        if (states.size() != controls.size() + 1U || states.empty()) {
            throw std::invalid_argument("path diagnostics require N controls and N+1 states");
        }
        PoweredDescentPathDiagnostics result{};
        for (const auto& state : states) {
            validate_state(state);
            result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[6U]);
            result.altitude = std::max(result.altitude, -state[2U]);
            result.glide_slope = std::max(
                result.glide_slope,
                std::hypot(state[0U], state[1U])
                    - state[2U] * config_.glide_slope_tangent()
            );
        }
        for (const auto& control : controls) {
            validate_control(control);
            const ThrustVector thrust{control[0U], control[1U], control[2U]};
            const auto magnitude = norm3(thrust);
            const auto sigma = control[3U];
            result.thrust_epigraph = std::max(result.thrust_epigraph, magnitude - sigma);
            result.throttle_lower = std::max(result.throttle_lower, config_.minimum_sigma - sigma);
            result.throttle_upper = std::max(result.throttle_upper, sigma - config_.maximum_thrust);
            result.tilt = std::max(result.tilt, sigma * config_.tilt_cosine() - thrust[2U]);
        }
        result.thrust_epigraph = std::max(result.thrust_epigraph, 0.0);
        result.throttle_lower = std::max(result.throttle_lower, 0.0);
        result.throttle_upper = std::max(result.throttle_upper, 0.0);
        result.tilt = std::max(result.tilt, 0.0);
        result.minimum_mass = std::max(result.minimum_mass, 0.0);
        result.altitude = std::max(result.altitude, 0.0);
        result.glide_slope = std::max(result.glide_slope, 0.0);
        return result;
    }

  private:
    PoweredDescent3DofConfig config_{};

    template <std::size_t Rows, std::size_t Columns>
    static constexpr std::size_t index(std::size_t row, std::size_t column) noexcept {
        return row * Columns + column;
    }

    static void require_step(double step_seconds) {
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            throw std::invalid_argument("step duration must be finite and positive");
        }
    }

    static void validate_state(const PoweredDescentState& state) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("powered-descent state must be finite");
            }
        }
        if (state[6U] <= 0.0) {
            throw std::invalid_argument("powered-descent mass must be positive");
        }
    }

    static void validate_control(const PoweredDescentControl& control) {
        for (const auto value : control) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("powered-descent control must be finite");
            }
        }
    }

    static PoweredDescentState add_scaled(
        const PoweredDescentState& state,
        const PoweredDescentState& derivative,
        double scale
    ) {
        PoweredDescentState result{};
        for (std::size_t index_value = 0; index_value < state.size(); ++index_value) {
            result[index_value] = state[index_value] + scale * derivative[index_value];
        }
        return result;
    }

    static double norm3(const ThrustVector& vector) noexcept {
        return std::sqrt(
            vector[0U] * vector[0U] + vector[1U] * vector[1U]
            + vector[2U] * vector[2U]
        );
    }
};

}  // namespace spacepdhcg::dynamics
