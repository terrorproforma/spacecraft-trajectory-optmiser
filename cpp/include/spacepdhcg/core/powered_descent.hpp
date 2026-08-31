#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::core {

inline constexpr std::size_t powered_descent_state_dimension = 7;
inline constexpr std::size_t powered_descent_control_dimension = 4;
using PoweredDescentState = std::array<double, powered_descent_state_dimension>;
using PoweredDescentControl = std::array<double, powered_descent_control_dimension>;
using PoweredDescentStateJacobian =
    std::array<double, powered_descent_state_dimension * powered_descent_state_dimension>;
using PoweredDescentControlJacobian =
    std::array<double, powered_descent_state_dimension * powered_descent_control_dimension>;

[[nodiscard]] constexpr std::size_t powered_descent_state_index(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row * powered_descent_state_dimension + column;
}

[[nodiscard]] constexpr std::size_t powered_descent_control_index(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row * powered_descent_control_dimension + column;
}

struct PoweredDescentConfig {
    std::array<double, 3> gravity{0.0, 0.0, -3.711};
    double mass_flow_coefficient{4.6e-4};
    double minimum_mass{1'000.0};
    double maximum_thrust{15'000.0};
    double minimum_sigma{0.0};
    double maximum_tilt_radians{0.5235987755982988};
    double glide_slope_radians{1.0471975511965976};

    void validate() const {
        for (const auto value : gravity) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("gravity components must be finite");
            }
        }
        if (!std::isfinite(mass_flow_coefficient) || mass_flow_coefficient <= 0.0) {
            throw std::invalid_argument("mass-flow coefficient must be finite and positive");
        }
        if (!std::isfinite(minimum_mass) || minimum_mass <= 0.0) {
            throw std::invalid_argument("minimum mass must be finite and positive");
        }
        if (!std::isfinite(maximum_thrust) || maximum_thrust <= 0.0) {
            throw std::invalid_argument("maximum thrust must be finite and positive");
        }
        if (!std::isfinite(minimum_sigma) || minimum_sigma < 0.0 ||
            minimum_sigma > maximum_thrust) {
            throw std::invalid_argument("minimum sigma must lie between zero and maximum thrust");
        }
        constexpr double half_pi = 1.5707963267948966;
        if (!std::isfinite(maximum_tilt_radians) || maximum_tilt_radians <= 0.0 ||
            maximum_tilt_radians >= half_pi) {
            throw std::invalid_argument("maximum tilt must lie strictly between zero and pi/2");
        }
        if (!std::isfinite(glide_slope_radians) || glide_slope_radians <= 0.0 ||
            glide_slope_radians >= half_pi) {
            throw std::invalid_argument("glide slope must lie strictly between zero and pi/2");
        }
    }

    [[nodiscard]] double tilt_cosine() const noexcept {
        return std::cos(maximum_tilt_radians);
    }

    [[nodiscard]] double glide_slope_tangent() const noexcept {
        return std::tan(glide_slope_radians);
    }
};

struct PoweredDescentJacobians {
    PoweredDescentStateJacobian state{};
    PoweredDescentControlJacobian control{};
};

struct PoweredDescentAffineLinearisation {
    PoweredDescentStateJacobian state{};
    PoweredDescentControlJacobian control{};
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

    [[nodiscard]] bool feasible(const double tolerance = 1.0e-8) const {
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument("path-feasibility tolerance must be non-negative");
        }
        return maximum_violation() <= tolerance;
    }
};

class PoweredDescent3DOF {
  public:
    explicit PoweredDescent3DOF(PoweredDescentConfig config = {}) : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const PoweredDescentConfig& config() const noexcept { return config_; }

    [[nodiscard]] PoweredDescentState dynamics(
        const PoweredDescentState& state,
        const PoweredDescentControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        const double mass = state[6];
        if (mass <= 0.0) {
            throw std::invalid_argument("powered-descent mass must be positive");
        }
        PoweredDescentState derivative{};
        derivative[0] = state[3];
        derivative[1] = state[4];
        derivative[2] = state[5];
        derivative[3] = control[0] / mass + config_.gravity[0];
        derivative[4] = control[1] / mass + config_.gravity[1];
        derivative[5] = control[2] / mass + config_.gravity[2];
        derivative[6] = -config_.mass_flow_coefficient * control[3];
        return derivative;
    }

    [[nodiscard]] PoweredDescentJacobians jacobians(
        const PoweredDescentState& state,
        const PoweredDescentControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        const double mass = state[6];
        if (mass <= 0.0) {
            throw std::invalid_argument("powered-descent mass must be positive");
        }
        PoweredDescentJacobians result{};
        result.state[powered_descent_state_index(0, 3)] = 1.0;
        result.state[powered_descent_state_index(1, 4)] = 1.0;
        result.state[powered_descent_state_index(2, 5)] = 1.0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            result.state[powered_descent_state_index(3 + axis, 6)] =
                -control[axis] / (mass * mass);
            result.control[powered_descent_control_index(3 + axis, axis)] = 1.0 / mass;
        }
        result.control[powered_descent_control_index(6, 3)] =
            -config_.mass_flow_coefficient;
        return result;
    }

    [[nodiscard]] PoweredDescentAffineLinearisation affine_linearisation(
        const PoweredDescentState& state,
        const PoweredDescentControl& control
    ) const {
        const auto derivative = dynamics(state, control);
        const auto derivatives = jacobians(state, control);
        PoweredDescentAffineLinearisation result{};
        result.state = derivatives.state;
        result.control = derivatives.control;
        for (std::size_t row = 0; row < powered_descent_state_dimension; ++row) {
            double value = derivative[row];
            for (std::size_t column = 0; column < powered_descent_state_dimension; ++column) {
                value -= derivatives.state[powered_descent_state_index(row, column)] * state[column];
            }
            for (std::size_t column = 0; column < powered_descent_control_dimension; ++column) {
                value -= derivatives.control[powered_descent_control_index(row, column)] *
                    control[column];
            }
            result.offset[row] = value;
        }
        return result;
    }

    [[nodiscard]] PoweredDescentState euler_step(
        const PoweredDescentState& state,
        const PoweredDescentControl& control,
        const double step_seconds
    ) const {
        validate_step(step_seconds);
        const auto derivative = dynamics(state, control);
        PoweredDescentState next{};
        for (std::size_t index = 0; index < powered_descent_state_dimension; ++index) {
            next[index] = state[index] + step_seconds * derivative[index];
        }
        if (next[6] <= 0.0) {
            throw std::invalid_argument("powered-descent Euler step produced non-positive mass");
        }
        return next;
    }

    [[nodiscard]] PoweredDescentState rk4_step(
        const PoweredDescentState& state,
        const PoweredDescentControl& control,
        const double step_seconds
    ) const {
        validate_step(step_seconds);
        const auto k1 = dynamics(state, control);
        const auto state2 = add_scaled(state, k1, 0.5 * step_seconds);
        const auto k2 = dynamics(state2, control);
        const auto state3 = add_scaled(state, k2, 0.5 * step_seconds);
        const auto k3 = dynamics(state3, control);
        const auto state4 = add_scaled(state, k3, step_seconds);
        const auto k4 = dynamics(state4, control);
        PoweredDescentState next{};
        for (std::size_t index = 0; index < powered_descent_state_dimension; ++index) {
            next[index] = state[index] +
                step_seconds * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) /
                    6.0;
        }
        if (next[6] <= 0.0) {
            throw std::invalid_argument("powered-descent RK4 step produced non-positive mass");
        }
        return next;
    }

    [[nodiscard]] std::vector<PoweredDescentState> rollout(
        const PoweredDescentState& initial_state,
        const std::vector<PoweredDescentControl>& controls,
        const double step_seconds,
        const bool use_rk4 = false
    ) const {
        validate_state(initial_state);
        validate_step(step_seconds);
        std::vector<PoweredDescentState> states(controls.size() + 1U);
        states.front() = initial_state;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            states[interval + 1U] = use_rk4
                ? rk4_step(states[interval], controls[interval], step_seconds)
                : euler_step(states[interval], controls[interval], step_seconds);
        }
        return states;
    }

    [[nodiscard]] PoweredDescentPathDiagnostics path_diagnostics(
        const std::vector<PoweredDescentState>& states,
        const std::vector<PoweredDescentControl>& controls
    ) const {
        if (states.size() != controls.size() + 1U || states.empty()) {
            throw std::invalid_argument(
                "powered-descent states must contain exactly one more node than controls"
            );
        }
        PoweredDescentPathDiagnostics result{};
        for (const auto& state : states) {
            validate_state(state);
            const double horizontal = std::hypot(state[0], state[1]);
            result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[6]);
            result.altitude = std::max(result.altitude, -state[2]);
            result.glide_slope = std::max(
                result.glide_slope,
                horizontal - config_.glide_slope_tangent() * state[2]
            );
        }
        for (const auto& control : controls) {
            validate_control(control);
            const double thrust_norm =
                std::sqrt(control[0] * control[0] + control[1] * control[1] +
                          control[2] * control[2]);
            result.thrust_epigraph =
                std::max(result.thrust_epigraph, thrust_norm - control[3]);
            result.throttle_lower =
                std::max(result.throttle_lower, config_.minimum_sigma - control[3]);
            result.throttle_upper =
                std::max(result.throttle_upper, control[3] - config_.maximum_thrust);
            result.tilt = std::max(
                result.tilt,
                config_.tilt_cosine() * control[3] - control[2]
            );
        }
        result.thrust_epigraph = std::max(0.0, result.thrust_epigraph);
        result.throttle_lower = std::max(0.0, result.throttle_lower);
        result.throttle_upper = std::max(0.0, result.throttle_upper);
        result.tilt = std::max(0.0, result.tilt);
        result.minimum_mass = std::max(0.0, result.minimum_mass);
        result.altitude = std::max(0.0, result.altitude);
        result.glide_slope = std::max(0.0, result.glide_slope);
        return result;
    }

  private:
    static void validate_state(const PoweredDescentState& state) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("powered-descent state must be finite");
            }
        }
    }

    static void validate_control(const PoweredDescentControl& control) {
        for (const auto value : control) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("powered-descent control must be finite");
            }
        }
    }

    static void validate_step(const double step_seconds) {
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            throw std::invalid_argument("powered-descent step duration must be positive");
        }
    }

    [[nodiscard]] static PoweredDescentState add_scaled(
        const PoweredDescentState& state,
        const PoweredDescentState& derivative,
        const double scale
    ) noexcept {
        PoweredDescentState result{};
        for (std::size_t index = 0; index < powered_descent_state_dimension; ++index) {
            result[index] = state[index] + scale * derivative[index];
        }
        return result;
    }

    PoweredDescentConfig config_;
};

}  // namespace spacepdhcg::core
