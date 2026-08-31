#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::core {

inline constexpr std::size_t powered_descent_6dof_state_dimension = 14;
inline constexpr std::size_t powered_descent_6dof_control_dimension = 7;
using PoweredDescent6DOFState = std::array<double, powered_descent_6dof_state_dimension>;
using PoweredDescent6DOFControl = std::array<double, powered_descent_6dof_control_dimension>;
using PoweredDescent6DOFStateJacobian = std::array<
    double,
    powered_descent_6dof_state_dimension * powered_descent_6dof_state_dimension>;
using PoweredDescent6DOFControlJacobian = std::array<
    double,
    powered_descent_6dof_state_dimension * powered_descent_6dof_control_dimension>;
using Quaternion = std::array<double, 4>;
using Matrix3 = std::array<double, 9>;

[[nodiscard]] constexpr std::size_t powered_descent_6dof_state_index(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row * powered_descent_6dof_state_dimension + column;
}

[[nodiscard]] constexpr std::size_t powered_descent_6dof_control_index(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row * powered_descent_6dof_control_dimension + column;
}

struct PoweredDescent6DOFConfig {
    std::array<double, 3> gravity{0.0, 0.0, -3.711};
    std::array<double, 3> inertia{1'200.0, 1'000.0, 800.0};
    std::array<double, 3> maximum_torque{2'000.0, 2'000.0, 2'000.0};
    double mass_flow_coefficient{4.6e-4};
    double minimum_mass{1'000.0};
    double minimum_sigma{0.0};
    double maximum_thrust{15'000.0};
    double maximum_angular_rate{1.0};
    double quaternion_norm_tolerance{1.0e-8};

    void validate() const {
        for (const auto value : gravity) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("6-DoF gravity components must be finite");
            }
        }
        for (const auto value : inertia) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("6-DoF principal inertias must be positive");
            }
        }
        for (const auto value : maximum_torque) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("6-DoF maximum torques must be positive");
            }
        }
        if (!std::isfinite(mass_flow_coefficient) || mass_flow_coefficient <= 0.0 ||
            !std::isfinite(minimum_mass) || minimum_mass <= 0.0 ||
            !std::isfinite(maximum_thrust) || maximum_thrust <= 0.0 ||
            !std::isfinite(maximum_angular_rate) || maximum_angular_rate <= 0.0) {
            throw std::invalid_argument("6-DoF physical bounds must be finite and positive");
        }
        if (!std::isfinite(minimum_sigma) || minimum_sigma < 0.0 ||
            minimum_sigma > maximum_thrust) {
            throw std::invalid_argument("6-DoF minimum sigma is invalid");
        }
        if (!std::isfinite(quaternion_norm_tolerance) || quaternion_norm_tolerance <= 0.0) {
            throw std::invalid_argument("quaternion norm tolerance must be positive");
        }
    }
};

struct PoweredDescent6DOFJacobians {
    PoweredDescent6DOFStateJacobian state{};
    PoweredDescent6DOFControlJacobian control{};
};

struct PoweredDescent6DOFPathDiagnostics {
    double thrust_epigraph{0.0};
    double throttle_lower{0.0};
    double throttle_upper{0.0};
    double torque{0.0};
    double angular_rate{0.0};
    double minimum_mass{0.0};
    double quaternion_norm{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max(
            {thrust_epigraph,
             throttle_lower,
             throttle_upper,
             torque,
             angular_rate,
             minimum_mass,
             quaternion_norm}
        );
    }
};

[[nodiscard]] inline double quaternion_norm(const Quaternion& quaternion) noexcept {
    return std::sqrt(
        quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
        quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]
    );
}

[[nodiscard]] inline Quaternion normalise_quaternion(const Quaternion& quaternion) {
    const double norm = quaternion_norm(quaternion);
    if (!std::isfinite(norm) || norm <= 0.0) {
        throw std::invalid_argument("quaternion must have a finite non-zero norm");
    }
    Quaternion result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = quaternion[index] / norm;
    }
    return result;
}

[[nodiscard]] inline Matrix3 quaternion_rotation_matrix(const Quaternion& raw_quaternion) {
    const auto quaternion = normalise_quaternion(raw_quaternion);
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    return Matrix3{
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    };
}

class PoweredDescent6DOF {
  public:
    explicit PoweredDescent6DOF(PoweredDescent6DOFConfig config = {}) : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const PoweredDescent6DOFConfig& config() const noexcept { return config_; }

    [[nodiscard]] PoweredDescent6DOFState dynamics(
        const PoweredDescent6DOFState& state,
        const PoweredDescent6DOFControl& control
    ) const {
        validate_state(state);
        validate_control(control);
        if (state[13] <= 0.0) {
            throw std::invalid_argument("6-DoF mass must be positive");
        }

        const Quaternion quaternion{state[6], state[7], state[8], state[9]};
        const auto unit_quaternion = normalise_quaternion(quaternion);
        const auto rotation = quaternion_rotation_matrix(unit_quaternion);
        const std::array<double, 3> thrust_body{control[0], control[1], control[2]};
        std::array<double, 3> thrust_inertial{};
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                thrust_inertial[row] += rotation[row * 3U + column] * thrust_body[column];
            }
        }

        const double wx = state[10];
        const double wy = state[11];
        const double wz = state[12];
        const double qw = unit_quaternion[0];
        const double qx = unit_quaternion[1];
        const double qy = unit_quaternion[2];
        const double qz = unit_quaternion[3];

        const std::array<double, 3> angular_momentum{
            config_.inertia[0] * wx,
            config_.inertia[1] * wy,
            config_.inertia[2] * wz,
        };
        const std::array<double, 3> gyroscopic{
            wy * angular_momentum[2] - wz * angular_momentum[1],
            wz * angular_momentum[0] - wx * angular_momentum[2],
            wx * angular_momentum[1] - wy * angular_momentum[0],
        };

        PoweredDescent6DOFState derivative{};
        derivative[0] = state[3];
        derivative[1] = state[4];
        derivative[2] = state[5];
        derivative[3] = thrust_inertial[0] / state[13] + config_.gravity[0];
        derivative[4] = thrust_inertial[1] / state[13] + config_.gravity[1];
        derivative[5] = thrust_inertial[2] / state[13] + config_.gravity[2];
        derivative[6] = 0.5 * (-qx * wx - qy * wy - qz * wz);
        derivative[7] = 0.5 * (qw * wx + qy * wz - qz * wy);
        derivative[8] = 0.5 * (qw * wy + qz * wx - qx * wz);
        derivative[9] = 0.5 * (qw * wz + qx * wy - qy * wx);
        derivative[10] = (control[3] - gyroscopic[0]) / config_.inertia[0];
        derivative[11] = (control[4] - gyroscopic[1]) / config_.inertia[1];
        derivative[12] = (control[5] - gyroscopic[2]) / config_.inertia[2];
        derivative[13] = -config_.mass_flow_coefficient * control[6];
        return derivative;
    }

    [[nodiscard]] PoweredDescent6DOFState rk4_step(
        const PoweredDescent6DOFState& state,
        const PoweredDescent6DOFControl& control,
        const double step_seconds
    ) const {
        validate_step(step_seconds);
        const auto k1 = dynamics(state, control);
        const auto k2 = dynamics(add_scaled(state, k1, 0.5 * step_seconds), control);
        const auto k3 = dynamics(add_scaled(state, k2, 0.5 * step_seconds), control);
        const auto k4 = dynamics(add_scaled(state, k3, step_seconds), control);
        PoweredDescent6DOFState next{};
        for (std::size_t index = 0; index < next.size(); ++index) {
            next[index] = state[index] +
                step_seconds * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) /
                    6.0;
        }
        const auto unit = normalise_quaternion(Quaternion{next[6], next[7], next[8], next[9]});
        for (std::size_t index = 0; index < 4; ++index) {
            next[6 + index] = unit[index];
        }
        if (next[13] <= 0.0) {
            throw std::invalid_argument("6-DoF integration produced non-positive mass");
        }
        return next;
    }

    [[nodiscard]] std::vector<PoweredDescent6DOFState> rollout(
        const PoweredDescent6DOFState& initial_state,
        const std::vector<PoweredDescent6DOFControl>& controls,
        const double step_seconds
    ) const {
        validate_state(initial_state);
        validate_step(step_seconds);
        std::vector<PoweredDescent6DOFState> states(controls.size() + 1U);
        states.front() = initial_state;
        const auto initial_quaternion = normalise_quaternion(
            Quaternion{states.front()[6], states.front()[7], states.front()[8], states.front()[9]}
        );
        for (std::size_t index = 0; index < 4; ++index) {
            states.front()[6 + index] = initial_quaternion[index];
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            states[interval + 1U] = rk4_step(states[interval], controls[interval], step_seconds);
        }
        return states;
    }

    [[nodiscard]] PoweredDescent6DOFJacobians numerical_jacobians(
        const PoweredDescent6DOFState& state,
        const PoweredDescent6DOFControl& control,
        const double epsilon = 1.0e-6
    ) const {
        if (!std::isfinite(epsilon) || epsilon <= 0.0) {
            throw std::invalid_argument("finite-difference epsilon must be positive");
        }
        PoweredDescent6DOFJacobians result{};
        for (std::size_t column = 0; column < state.size(); ++column) {
            auto plus = state;
            auto minus = state;
            plus[column] += epsilon;
            minus[column] -= epsilon;
            const auto forward = dynamics(plus, control);
            const auto backward = dynamics(minus, control);
            for (std::size_t row = 0; row < state.size(); ++row) {
                result.state[powered_descent_6dof_state_index(row, column)] =
                    (forward[row] - backward[row]) / (2.0 * epsilon);
            }
        }
        for (std::size_t column = 0; column < control.size(); ++column) {
            auto plus = control;
            auto minus = control;
            plus[column] += epsilon;
            minus[column] -= epsilon;
            const auto forward = dynamics(state, plus);
            const auto backward = dynamics(state, minus);
            for (std::size_t row = 0; row < state.size(); ++row) {
                result.control[powered_descent_6dof_control_index(row, column)] =
                    (forward[row] - backward[row]) / (2.0 * epsilon);
            }
        }
        return result;
    }

    [[nodiscard]] PoweredDescent6DOFPathDiagnostics path_diagnostics(
        const std::vector<PoweredDescent6DOFState>& states,
        const std::vector<PoweredDescent6DOFControl>& controls
    ) const {
        if (states.size() != controls.size() + 1U || states.empty()) {
            throw std::invalid_argument("6-DoF states must contain one more node than controls");
        }
        PoweredDescent6DOFPathDiagnostics result{};
        for (const auto& state : states) {
            validate_state(state);
            result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[13]);
            const double rate = std::sqrt(
                state[10] * state[10] + state[11] * state[11] + state[12] * state[12]
            );
            result.angular_rate = std::max(result.angular_rate, rate - config_.maximum_angular_rate);
            const double norm = quaternion_norm(Quaternion{state[6], state[7], state[8], state[9]});
            result.quaternion_norm = std::max(result.quaternion_norm, std::abs(norm - 1.0));
        }
        for (const auto& control : controls) {
            validate_control(control);
            const double thrust = std::sqrt(
                control[0] * control[0] + control[1] * control[1] + control[2] * control[2]
            );
            result.thrust_epigraph = std::max(result.thrust_epigraph, thrust - control[6]);
            result.throttle_lower = std::max(result.throttle_lower, config_.minimum_sigma - control[6]);
            result.throttle_upper = std::max(result.throttle_upper, control[6] - config_.maximum_thrust);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result.torque = std::max(
                    result.torque,
                    std::abs(control[3 + axis]) - config_.maximum_torque[axis]
                );
            }
        }
        result.thrust_epigraph = std::max(0.0, result.thrust_epigraph);
        result.throttle_lower = std::max(0.0, result.throttle_lower);
        result.throttle_upper = std::max(0.0, result.throttle_upper);
        result.torque = std::max(0.0, result.torque);
        result.angular_rate = std::max(0.0, result.angular_rate);
        result.minimum_mass = std::max(0.0, result.minimum_mass);
        result.quaternion_norm = std::max(0.0, result.quaternion_norm - config_.quaternion_norm_tolerance);
        return result;
    }

  private:
    static void validate_state(const PoweredDescent6DOFState& state) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("6-DoF state must be finite");
            }
        }
    }

    static void validate_control(const PoweredDescent6DOFControl& control) {
        for (const auto value : control) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("6-DoF control must be finite");
            }
        }
    }

    static void validate_step(const double step_seconds) {
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            throw std::invalid_argument("6-DoF integration step must be positive");
        }
    }

    [[nodiscard]] static PoweredDescent6DOFState add_scaled(
        const PoweredDescent6DOFState& state,
        const PoweredDescent6DOFState& derivative,
        const double scale
    ) noexcept {
        PoweredDescent6DOFState result{};
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = state[index] + scale * derivative[index];
        }
        return result;
    }

    PoweredDescent6DOFConfig config_;
};

}  // namespace spacepdhcg::core
