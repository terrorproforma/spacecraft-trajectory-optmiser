#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::dynamics {

inline constexpr std::size_t low_thrust_state_dimension = 7U;
inline constexpr std::size_t low_thrust_control_dimension = 4U;

using LowThrustState = std::array<double, low_thrust_state_dimension>;
using LowThrustControl = std::array<double, low_thrust_control_dimension>;
using LowThrustVector3 = std::array<double, 3U>;

template <std::size_t Rows, std::size_t Columns>
using LowThrustMatrix = std::array<double, Rows * Columns>;

struct LowThrustTwoBodyConfig {
    double gravitational_parameter{398'600.4418};
    double thrust_to_acceleration{1.0e-3};
    double mass_flow_coefficient{3.4e-5};
    double minimum_mass{200.0};
    double maximum_thrust{1.0};
    double minimum_radius{6'500.0};

    void validate() const {
        require_positive(gravitational_parameter, "gravitational parameter must be positive");
        require_positive(thrust_to_acceleration, "thrust acceleration scale must be positive");
        require_positive(mass_flow_coefficient, "mass-flow coefficient must be positive");
        require_positive(minimum_mass, "minimum mass must be positive");
        require_positive(maximum_thrust, "maximum thrust must be positive");
        require_positive(minimum_radius, "minimum radius must be positive");
    }

  private:
    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
};

struct LowThrustJacobians {
    LowThrustMatrix<7U, 7U> state{};
    LowThrustMatrix<7U, 4U> control{};
};

struct LowThrustAffineLinearisation {
    LowThrustJacobians jacobians{};
    LowThrustState offset{};
};

struct LowThrustEulerLinearisation {
    LowThrustMatrix<7U, 7U> state{};
    LowThrustMatrix<7U, 4U> control{};
    LowThrustState offset{};
};

struct LowThrustPathDiagnostics {
    double thrust_epigraph{0.0};
    double throttle_upper{0.0};
    double minimum_mass{0.0};
    double minimum_radius{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max(
            {thrust_epigraph, throttle_upper, minimum_mass, minimum_radius}
        );
    }
};

class LowThrustTwoBodyModel {
  public:
    explicit LowThrustTwoBodyModel(LowThrustTwoBodyConfig config = {}) : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const LowThrustTwoBodyConfig& config() const noexcept { return config_; }

    [[nodiscard]] LowThrustState dynamics(
        const LowThrustState& state,
        const LowThrustControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        const auto radius = position_norm(state);
        const auto inverse_radius_cubed = 1.0 / (radius * radius * radius);
        const auto mass = state[6U];
        LowThrustState derivative{};
        derivative[0U] = state[3U];
        derivative[1U] = state[4U];
        derivative[2U] = state[5U];
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            derivative[3U + axis] =
                -config_.gravitational_parameter * state[axis] * inverse_radius_cubed
                + config_.thrust_to_acceleration * control[axis] / mass;
        }
        derivative[6U] = -config_.mass_flow_coefficient * control[3U];
        return derivative;
    }

    [[nodiscard]] LowThrustJacobians jacobians(
        const LowThrustState& state,
        const LowThrustControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        const auto radius = position_norm(state);
        const auto radius_squared = radius * radius;
        const auto inverse_radius_cubed = 1.0 / (radius_squared * radius);
        const auto inverse_radius_fifth = inverse_radius_cubed / radius_squared;
        const auto mass = state[6U];
        LowThrustJacobians result{};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            result.state[index<7U, 7U>(axis, 3U + axis)] = 1.0;
            for (std::size_t column = 0; column < 3U; ++column) {
                const auto identity = axis == column ? 1.0 : 0.0;
                result.state[index<7U, 7U>(3U + axis, column)] =
                    config_.gravitational_parameter
                    * (3.0 * state[axis] * state[column] * inverse_radius_fifth
                       - identity * inverse_radius_cubed);
            }
            result.state[index<7U, 7U>(3U + axis, 6U)] =
                -config_.thrust_to_acceleration * control[axis] / (mass * mass);
            result.control[index<7U, 4U>(3U + axis, axis)] =
                config_.thrust_to_acceleration / mass;
        }
        result.control[index<7U, 4U>(6U, 3U)] = -config_.mass_flow_coefficient;
        return result;
    }

    [[nodiscard]] LowThrustAffineLinearisation affine_linearisation(
        const LowThrustState& state,
        const LowThrustControl& control
    ) const {
        const auto derivative = dynamics(state, control);
        const auto derivatives = jacobians(state, control);
        LowThrustAffineLinearisation result{derivatives, derivative};
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

    [[nodiscard]] LowThrustEulerLinearisation linearised_euler_dynamics(
        const LowThrustState& state,
        const LowThrustControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto continuous = affine_linearisation(state, control);
        LowThrustEulerLinearisation result{};
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

    [[nodiscard]] LowThrustState euler_step(
        const LowThrustState& state,
        const LowThrustControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto derivative = dynamics(state, control);
        LowThrustState next{};
        for (std::size_t component = 0; component < next.size(); ++component) {
            next[component] = state[component] + step_seconds * derivative[component];
        }
        validate_state(next);
        return next;
    }

    [[nodiscard]] LowThrustState rk4_step(
        const LowThrustState& state,
        const LowThrustControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto k1 = dynamics(state, control);
        const auto k2 = dynamics(add_scaled(state, k1, 0.5 * step_seconds), control);
        const auto k3 = dynamics(add_scaled(state, k2, 0.5 * step_seconds), control);
        const auto k4 = dynamics(add_scaled(state, k3, step_seconds), control);
        LowThrustState next{};
        for (std::size_t component = 0; component < next.size(); ++component) {
            next[component] = state[component]
                              + step_seconds
                                    * (k1[component] + 2.0 * k2[component]
                                       + 2.0 * k3[component] + k4[component])
                                    / 6.0;
        }
        validate_state(next);
        return next;
    }

    [[nodiscard]] std::vector<LowThrustState> rollout(
        const LowThrustState& initial,
        const std::vector<LowThrustControl>& controls,
        double step_seconds,
        bool use_rk4 = true
    ) const {
        validate_state(initial);
        std::vector<LowThrustState> states(controls.size() + 1U);
        states.front() = initial;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            states[interval + 1U] = use_rk4
                                       ? rk4_step(states[interval], controls[interval], step_seconds)
                                       : euler_step(states[interval], controls[interval], step_seconds);
        }
        return states;
    }

    [[nodiscard]] LowThrustPathDiagnostics path_diagnostics(
        const std::vector<LowThrustState>& states,
        const std::vector<LowThrustControl>& controls
    ) const {
        if (states.empty() || states.size() != controls.size() + 1U) {
            throw std::invalid_argument("low-thrust diagnostics require N controls and N+1 states");
        }
        LowThrustPathDiagnostics result{};
        for (const auto& state : states) {
            validate_state(state);
            result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[6U]);
            result.minimum_radius = std::max(
                result.minimum_radius,
                config_.minimum_radius - position_norm(state)
            );
        }
        for (const auto& control : controls) {
            validate_control(control);
            const auto thrust_norm = std::sqrt(
                control[0U] * control[0U] + control[1U] * control[1U]
                + control[2U] * control[2U]
            );
            result.thrust_epigraph = std::max(
                result.thrust_epigraph,
                thrust_norm - control[3U]
            );
            result.throttle_upper = std::max(
                result.throttle_upper,
                control[3U] - config_.maximum_thrust
            );
        }
        result.thrust_epigraph = std::max(result.thrust_epigraph, 0.0);
        result.throttle_upper = std::max(result.throttle_upper, 0.0);
        result.minimum_mass = std::max(result.minimum_mass, 0.0);
        result.minimum_radius = std::max(result.minimum_radius, 0.0);
        return result;
    }

  private:
    LowThrustTwoBodyConfig config_{};

    template <std::size_t Rows, std::size_t Columns>
    static constexpr std::size_t index(std::size_t row, std::size_t column) noexcept {
        return row * Columns + column;
    }

    static void require_step(double step_seconds) {
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            throw std::invalid_argument("low-thrust step duration must be finite and positive");
        }
    }

    static void validate_control(const LowThrustControl& control) {
        for (const auto value : control) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("low-thrust control must be finite");
            }
        }
        if (control[3U] < 0.0) {
            throw std::invalid_argument("low-thrust magnitude epigraph must be non-negative");
        }
    }

    static void validate_state(const LowThrustState& state) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("low-thrust state must be finite");
            }
        }
        if (state[6U] <= 0.0 || position_norm(state) <= 0.0) {
            throw std::invalid_argument("low-thrust state must have positive mass and radius");
        }
    }

    static double position_norm(const LowThrustState& state) noexcept {
        return std::sqrt(
            state[0U] * state[0U] + state[1U] * state[1U] + state[2U] * state[2U]
        );
    }

    static LowThrustState add_scaled(
        const LowThrustState& state,
        const LowThrustState& derivative,
        double scale
    ) {
        LowThrustState result{};
        for (std::size_t component = 0; component < state.size(); ++component) {
            result[component] = state[component] + scale * derivative[component];
        }
        return result;
    }
};

}  // namespace spacepdhcg::dynamics
