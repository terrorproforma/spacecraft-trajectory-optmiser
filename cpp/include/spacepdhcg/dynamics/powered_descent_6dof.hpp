#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::dynamics {

inline constexpr std::size_t powered_descent_6dof_state_dimension = 14U;
inline constexpr std::size_t powered_descent_6dof_control_dimension = 7U;

using PoweredDescent6DofState = std::array<double, powered_descent_6dof_state_dimension>;
using PoweredDescent6DofControl = std::array<double, powered_descent_6dof_control_dimension>;
using Vector3d = std::array<double, 3U>;
using Quaternion = std::array<double, 4U>;

template <std::size_t Rows, std::size_t Columns>
using Matrix = std::array<double, Rows * Columns>;

struct PoweredDescent6DofConfig {
    Vector3d gravity{0.0, 0.0, -3.711};
    Vector3d principal_inertia{2'500.0, 2'200.0, 1'800.0};
    double mass_flow_coefficient{4.6e-4};
    double minimum_mass{1'000.0};
    double maximum_thrust{15'000.0};
    double minimum_sigma{0.0};
    double maximum_torque{2'000.0};
    double maximum_angular_rate{1.0};
    double maximum_tilt_radians{0.5235987755982988};
    double glide_slope_radians{1.0471975511965976};

    void validate() const {
        validate_finite(gravity, "gravity must be finite");
        validate_positive(principal_inertia, "principal inertia must be positive");
        require_positive(mass_flow_coefficient, "mass-flow coefficient must be positive");
        require_positive(minimum_mass, "minimum mass must be positive");
        require_positive(maximum_thrust, "maximum thrust must be positive");
        if (!std::isfinite(minimum_sigma) || minimum_sigma < 0.0
            || minimum_sigma > maximum_thrust) {
            throw std::invalid_argument("minimum sigma must lie in [0, maximum thrust]");
        }
        require_positive(maximum_torque, "maximum torque must be positive");
        require_positive(maximum_angular_rate, "maximum angular rate must be positive");
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
    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }

    static void validate_finite(const Vector3d& values, const char* message) {
        for (const auto value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(message);
            }
        }
    }

    static void validate_positive(const Vector3d& values, const char* message) {
        for (const auto value : values) {
            require_positive(value, message);
        }
    }
};

struct PoweredDescent6DofJacobians {
    Matrix<14U, 14U> state{};
    Matrix<14U, 7U> control{};
};

struct PoweredDescent6DofPathDiagnostics {
    double thrust_epigraph{0.0};
    double throttle_lower{0.0};
    double throttle_upper{0.0};
    double torque{0.0};
    double angular_rate{0.0};
    double minimum_mass{0.0};
    double altitude{0.0};
    double quaternion_norm_error{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max(
            {thrust_epigraph,
             throttle_lower,
             throttle_upper,
             torque,
             angular_rate,
             minimum_mass,
             altitude,
             quaternion_norm_error}
        );
    }
};

class PoweredDescent6DofModel {
  public:
    explicit PoweredDescent6DofModel(PoweredDescent6DofConfig config = {})
        : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const PoweredDescent6DofConfig& config() const noexcept { return config_; }

    [[nodiscard]] PoweredDescent6DofState dynamics(
        const PoweredDescent6DofState& state,
        const PoweredDescent6DofControl& control
    ) const {
        validate_state(state, false);
        validate_control(control);
        const auto mass = state[13U];
        const Quaternion quaternion{state[6U], state[7U], state[8U], state[9U]};
        const Vector3d body_thrust{control[0U], control[1U], control[2U]};
        const Vector3d torque{control[3U], control[4U], control[5U]};
        const Vector3d omega{state[10U], state[11U], state[12U]};
        const auto inertial_thrust = rotate_body_to_inertial(quaternion, body_thrust);
        const auto quaternion_rate = quaternion_derivative(quaternion, omega);

        PoweredDescent6DofState derivative{};
        derivative[0U] = state[3U];
        derivative[1U] = state[4U];
        derivative[2U] = state[5U];
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            derivative[3U + axis] = inertial_thrust[axis] / mass + config_.gravity[axis];
            derivative[6U + axis] = quaternion_rate[axis];
        }
        derivative[9U] = quaternion_rate[3U];

        const auto ix = config_.principal_inertia[0U];
        const auto iy = config_.principal_inertia[1U];
        const auto iz = config_.principal_inertia[2U];
        derivative[10U] = (torque[0U] - (iz - iy) * omega[1U] * omega[2U]) / ix;
        derivative[11U] = (torque[1U] - (ix - iz) * omega[0U] * omega[2U]) / iy;
        derivative[12U] = (torque[2U] - (iy - ix) * omega[0U] * omega[1U]) / iz;
        derivative[13U] = -config_.mass_flow_coefficient * control[6U];
        return derivative;
    }

    [[nodiscard]] PoweredDescent6DofJacobians jacobians(
        const PoweredDescent6DofState& state,
        const PoweredDescent6DofControl& control
    ) const {
        validate_state(state, false);
        validate_control(control);
        const auto mass = state[13U];
        const Quaternion quaternion{state[6U], state[7U], state[8U], state[9U]};
        const Vector3d thrust{control[0U], control[1U], control[2U]};
        const Vector3d omega{state[10U], state[11U], state[12U]};
        const auto rotation = rotation_matrix(quaternion);
        const auto rotation_derivatives = rotation_matrix_derivatives(quaternion);
        const auto inertial_thrust = matrix_vector(rotation, thrust);

        PoweredDescent6DofJacobians result{};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            result.state[index<14U, 14U>(axis, 3U + axis)] = 1.0;
            result.state[index<14U, 14U>(3U + axis, 13U)] =
                -inertial_thrust[axis] / (mass * mass);
            for (std::size_t body_axis = 0; body_axis < 3U; ++body_axis) {
                result.control[index<14U, 7U>(3U + axis, body_axis)] =
                    rotation[index<3U, 3U>(axis, body_axis)] / mass;
            }
            for (std::size_t quaternion_component = 0; quaternion_component < 4U;
                 ++quaternion_component) {
                const auto derivative_vector = matrix_vector(
                    rotation_derivatives[quaternion_component],
                    thrust
                );
                result.state[index<14U, 14U>(
                    3U + axis,
                    6U + quaternion_component
                )] = derivative_vector[axis] / mass;
            }
        }

        fill_quaternion_jacobians(result, quaternion, omega);
        fill_angular_jacobians(result, omega);
        result.control[index<14U, 7U>(13U, 6U)] = -config_.mass_flow_coefficient;
        return result;
    }

    [[nodiscard]] PoweredDescent6DofState euler_step(
        const PoweredDescent6DofState& state,
        const PoweredDescent6DofControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto derivative = dynamics(state, control);
        PoweredDescent6DofState next{};
        for (std::size_t component = 0; component < next.size(); ++component) {
            next[component] = state[component] + step_seconds * derivative[component];
        }
        normalise_quaternion(next);
        validate_state(next, true);
        return next;
    }

    [[nodiscard]] PoweredDescent6DofState rk4_step(
        const PoweredDescent6DofState& state,
        const PoweredDescent6DofControl& control,
        double step_seconds
    ) const {
        require_step(step_seconds);
        const auto k1 = dynamics(state, control);
        const auto k2 = dynamics(add_scaled(state, k1, 0.5 * step_seconds), control);
        const auto k3 = dynamics(add_scaled(state, k2, 0.5 * step_seconds), control);
        const auto k4 = dynamics(add_scaled(state, k3, step_seconds), control);
        PoweredDescent6DofState next{};
        for (std::size_t component = 0; component < next.size(); ++component) {
            next[component] = state[component]
                              + step_seconds
                                    * (k1[component] + 2.0 * k2[component]
                                       + 2.0 * k3[component] + k4[component])
                                    / 6.0;
        }
        normalise_quaternion(next);
        validate_state(next, true);
        return next;
    }

    [[nodiscard]] std::vector<PoweredDescent6DofState> rollout(
        const PoweredDescent6DofState& initial,
        const std::vector<PoweredDescent6DofControl>& controls,
        double step_seconds
    ) const {
        validate_state(initial, true);
        std::vector<PoweredDescent6DofState> states(controls.size() + 1U);
        states.front() = initial;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            states[interval + 1U] = rk4_step(states[interval], controls[interval], step_seconds);
        }
        return states;
    }

    [[nodiscard]] PoweredDescent6DofPathDiagnostics path_diagnostics(
        const std::vector<PoweredDescent6DofState>& states,
        const std::vector<PoweredDescent6DofControl>& controls
    ) const {
        if (states.empty() || states.size() != controls.size() + 1U) {
            throw std::invalid_argument("6-DoF diagnostics require N controls and N+1 states");
        }
        PoweredDescent6DofPathDiagnostics result{};
        for (const auto& state : states) {
            validate_state(state, false);
            result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[13U]);
            result.altitude = std::max(result.altitude, -state[2U]);
            const Quaternion quaternion{state[6U], state[7U], state[8U], state[9U]};
            result.quaternion_norm_error = std::max(
                result.quaternion_norm_error,
                std::abs(quaternion_norm(quaternion) - 1.0)
            );
            const Vector3d omega{state[10U], state[11U], state[12U]};
            result.angular_rate = std::max(
                result.angular_rate,
                vector_norm(omega) - config_.maximum_angular_rate
            );
        }
        for (const auto& control : controls) {
            validate_control(control);
            const Vector3d thrust{control[0U], control[1U], control[2U]};
            const Vector3d torque{control[3U], control[4U], control[5U]};
            const auto sigma = control[6U];
            result.thrust_epigraph = std::max(
                result.thrust_epigraph,
                vector_norm(thrust) - sigma
            );
            result.throttle_lower = std::max(
                result.throttle_lower,
                config_.minimum_sigma - sigma
            );
            result.throttle_upper = std::max(
                result.throttle_upper,
                sigma - config_.maximum_thrust
            );
            result.torque = std::max(
                result.torque,
                vector_norm(torque) - config_.maximum_torque
            );
        }
        result.thrust_epigraph = std::max(result.thrust_epigraph, 0.0);
        result.throttle_lower = std::max(result.throttle_lower, 0.0);
        result.throttle_upper = std::max(result.throttle_upper, 0.0);
        result.torque = std::max(result.torque, 0.0);
        result.angular_rate = std::max(result.angular_rate, 0.0);
        result.minimum_mass = std::max(result.minimum_mass, 0.0);
        result.altitude = std::max(result.altitude, 0.0);
        return result;
    }

    [[nodiscard]] static Matrix<3U, 3U> rotation_matrix(const Quaternion& quaternion) {
        validate_quaternion_values(quaternion);
        const auto w = quaternion[0U];
        const auto x = quaternion[1U];
        const auto y = quaternion[2U];
        const auto z = quaternion[3U];
        return Matrix<3U, 3U>{
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

    [[nodiscard]] static Vector3d rotate_body_to_inertial(
        const Quaternion& quaternion,
        const Vector3d& vector
    ) {
        return matrix_vector(rotation_matrix(quaternion), vector);
    }

  private:
    PoweredDescent6DofConfig config_{};

    template <std::size_t Rows, std::size_t Columns>
    static constexpr std::size_t index(std::size_t row, std::size_t column) noexcept {
        return row * Columns + column;
    }

    static void require_step(double step_seconds) {
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            throw std::invalid_argument("6-DoF step duration must be finite and positive");
        }
    }

    static void validate_state(const PoweredDescent6DofState& state, bool require_unit_quaternion) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("6-DoF state must be finite");
            }
        }
        if (state[13U] <= 0.0) {
            throw std::invalid_argument("6-DoF mass must be positive");
        }
        if (require_unit_quaternion) {
            const Quaternion quaternion{state[6U], state[7U], state[8U], state[9U]};
            if (std::abs(quaternion_norm(quaternion) - 1.0) > 1.0e-8) {
                throw std::invalid_argument("6-DoF quaternion must have unit norm");
            }
        }
    }

    static void validate_control(const PoweredDescent6DofControl& control) {
        for (const auto value : control) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("6-DoF control must be finite");
            }
        }
    }

    static void validate_quaternion_values(const Quaternion& quaternion) {
        for (const auto value : quaternion) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("quaternion must be finite");
            }
        }
    }

    static double quaternion_norm(const Quaternion& quaternion) noexcept {
        return std::sqrt(
            quaternion[0U] * quaternion[0U] + quaternion[1U] * quaternion[1U]
            + quaternion[2U] * quaternion[2U] + quaternion[3U] * quaternion[3U]
        );
    }

    static double vector_norm(const Vector3d& vector) noexcept {
        return std::sqrt(
            vector[0U] * vector[0U] + vector[1U] * vector[1U]
            + vector[2U] * vector[2U]
        );
    }

    static void normalise_quaternion(PoweredDescent6DofState& state) {
        Quaternion quaternion{state[6U], state[7U], state[8U], state[9U]};
        const auto norm = quaternion_norm(quaternion);
        if (!std::isfinite(norm) || norm <= 0.0) {
            throw std::runtime_error("6-DoF quaternion normalisation failed");
        }
        for (std::size_t component = 0; component < 4U; ++component) {
            state[6U + component] /= norm;
        }
    }

    static PoweredDescent6DofState add_scaled(
        const PoweredDescent6DofState& state,
        const PoweredDescent6DofState& derivative,
        double scale
    ) {
        PoweredDescent6DofState result{};
        for (std::size_t component = 0; component < state.size(); ++component) {
            result[component] = state[component] + scale * derivative[component];
        }
        return result;
    }

    static Quaternion quaternion_derivative(
        const Quaternion& quaternion,
        const Vector3d& omega
    ) noexcept {
        const auto w = quaternion[0U];
        const auto x = quaternion[1U];
        const auto y = quaternion[2U];
        const auto z = quaternion[3U];
        const auto wx = omega[0U];
        const auto wy = omega[1U];
        const auto wz = omega[2U];
        return Quaternion{
            -0.5 * (x * wx + y * wy + z * wz),
            0.5 * (w * wx + y * wz - z * wy),
            0.5 * (w * wy + z * wx - x * wz),
            0.5 * (w * wz + x * wy - y * wx),
        };
    }

    static Vector3d matrix_vector(const Matrix<3U, 3U>& matrix, const Vector3d& vector) noexcept {
        Vector3d result{};
        for (std::size_t row = 0; row < 3U; ++row) {
            for (std::size_t column = 0; column < 3U; ++column) {
                result[row] += matrix[index<3U, 3U>(row, column)] * vector[column];
            }
        }
        return result;
    }

    static std::array<Matrix<3U, 3U>, 4U> rotation_matrix_derivatives(
        const Quaternion& quaternion
    ) {
        validate_quaternion_values(quaternion);
        const auto w = quaternion[0U];
        const auto x = quaternion[1U];
        const auto y = quaternion[2U];
        const auto z = quaternion[3U];
        return {
            Matrix<3U, 3U>{
                0.0, -2.0 * z, 2.0 * y,
                2.0 * z, 0.0, -2.0 * x,
                -2.0 * y, 2.0 * x, 0.0,
            },
            Matrix<3U, 3U>{
                0.0, 2.0 * y, 2.0 * z,
                2.0 * y, -4.0 * x, -2.0 * w,
                2.0 * z, 2.0 * w, -4.0 * x,
            },
            Matrix<3U, 3U>{
                -4.0 * y, 2.0 * x, 2.0 * w,
                2.0 * x, 0.0, 2.0 * z,
                -2.0 * w, 2.0 * z, -4.0 * y,
            },
            Matrix<3U, 3U>{
                -4.0 * z, -2.0 * w, 2.0 * x,
                2.0 * w, -4.0 * z, 2.0 * y,
                2.0 * x, 2.0 * y, 0.0,
            },
        };
    }

    static void fill_quaternion_jacobians(
        PoweredDescent6DofJacobians& result,
        const Quaternion& quaternion,
        const Vector3d& omega
    ) noexcept {
        const auto w = quaternion[0U];
        const auto x = quaternion[1U];
        const auto y = quaternion[2U];
        const auto z = quaternion[3U];
        const auto wx = omega[0U];
        const auto wy = omega[1U];
        const auto wz = omega[2U];
        const std::array<double, 16U> dqdot_dq{
            0.0, -0.5 * wx, -0.5 * wy, -0.5 * wz,
            0.5 * wx, 0.0, 0.5 * wz, -0.5 * wy,
            0.5 * wy, -0.5 * wz, 0.0, 0.5 * wx,
            0.5 * wz, 0.5 * wy, -0.5 * wx, 0.0,
        };
        const std::array<double, 12U> dqdot_domega{
            -0.5 * x, -0.5 * y, -0.5 * z,
            0.5 * w, -0.5 * z, 0.5 * y,
            0.5 * z, 0.5 * w, -0.5 * x,
            -0.5 * y, 0.5 * x, 0.5 * w,
        };
        for (std::size_t row = 0; row < 4U; ++row) {
            for (std::size_t column = 0; column < 4U; ++column) {
                result.state[index<14U, 14U>(6U + row, 6U + column)] =
                    dqdot_dq[row * 4U + column];
            }
            for (std::size_t column = 0; column < 3U; ++column) {
                result.state[index<14U, 14U>(6U + row, 10U + column)] =
                    dqdot_domega[row * 3U + column];
            }
        }
    }

    void fill_angular_jacobians(
        PoweredDescent6DofJacobians& result,
        const Vector3d& omega
    ) const noexcept {
        const auto ix = config_.principal_inertia[0U];
        const auto iy = config_.principal_inertia[1U];
        const auto iz = config_.principal_inertia[2U];
        result.state[index<14U, 14U>(10U, 11U)] = -(iz - iy) * omega[2U] / ix;
        result.state[index<14U, 14U>(10U, 12U)] = -(iz - iy) * omega[1U] / ix;
        result.state[index<14U, 14U>(11U, 10U)] = -(ix - iz) * omega[2U] / iy;
        result.state[index<14U, 14U>(11U, 12U)] = -(ix - iz) * omega[0U] / iy;
        result.state[index<14U, 14U>(12U, 10U)] = -(iy - ix) * omega[1U] / iz;
        result.state[index<14U, 14U>(12U, 11U)] = -(iy - ix) * omega[0U] / iz;
        result.control[index<14U, 7U>(10U, 3U)] = 1.0 / ix;
        result.control[index<14U, 7U>(11U, 4U)] = 1.0 / iy;
        result.control[index<14U, 7U>(12U, 5U)] = 1.0 / iz;
    }
};

}  // namespace spacepdhcg::dynamics
