#include "spacepdhcg/native/powered_descent_6dof.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace spacepdhcg::native {
namespace {

template <std::size_t Columns, typename Matrix>
double& entry(Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

template <std::size_t Columns, typename Matrix>
const double& entry(const Matrix& matrix, std::size_t row, std::size_t column) {
    return matrix[row * Columns + column];
}

void require_finite(std::span<const double> values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }
}

[[nodiscard]] std::array<double, 3> rotate(
    const RotationMatrix3& rotation,
    std::span<const double, 3> vector
) {
    std::array<double, 3> result{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            result[row] += entry<3>(rotation, row, column) * vector[column];
        }
    }
    return result;
}

[[nodiscard]] std::array<RotationMatrix3, 4> rotation_derivatives(
    std::span<const double, 4> quaternion
) {
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    std::array<RotationMatrix3, 4> derivatives{};

    auto& dw = derivatives[0];
    entry<3>(dw, 0, 1) = -2.0 * z;
    entry<3>(dw, 0, 2) = 2.0 * y;
    entry<3>(dw, 1, 0) = 2.0 * z;
    entry<3>(dw, 1, 2) = -2.0 * x;
    entry<3>(dw, 2, 0) = -2.0 * y;
    entry<3>(dw, 2, 1) = 2.0 * x;

    auto& dx = derivatives[1];
    entry<3>(dx, 0, 1) = 2.0 * y;
    entry<3>(dx, 0, 2) = 2.0 * z;
    entry<3>(dx, 1, 0) = 2.0 * y;
    entry<3>(dx, 1, 1) = -4.0 * x;
    entry<3>(dx, 1, 2) = -2.0 * w;
    entry<3>(dx, 2, 0) = 2.0 * z;
    entry<3>(dx, 2, 1) = 2.0 * w;
    entry<3>(dx, 2, 2) = -4.0 * x;

    auto& dy = derivatives[2];
    entry<3>(dy, 0, 0) = -4.0 * y;
    entry<3>(dy, 0, 1) = 2.0 * x;
    entry<3>(dy, 0, 2) = 2.0 * w;
    entry<3>(dy, 1, 0) = 2.0 * x;
    entry<3>(dy, 1, 2) = 2.0 * z;
    entry<3>(dy, 2, 0) = -2.0 * w;
    entry<3>(dy, 2, 1) = 2.0 * z;
    entry<3>(dy, 2, 2) = -4.0 * y;

    auto& dz = derivatives[3];
    entry<3>(dz, 0, 0) = -4.0 * z;
    entry<3>(dz, 0, 1) = -2.0 * w;
    entry<3>(dz, 0, 2) = 2.0 * x;
    entry<3>(dz, 1, 0) = 2.0 * w;
    entry<3>(dz, 1, 1) = -4.0 * z;
    entry<3>(dz, 1, 2) = 2.0 * y;
    entry<3>(dz, 2, 0) = 2.0 * x;
    entry<3>(dz, 2, 1) = 2.0 * y;

    return derivatives;
}

[[nodiscard]] PoweredDescent6DofState add_scaled(
    std::span<const double, powered_descent_6dof_state_dimension> state,
    std::span<const double, powered_descent_6dof_state_dimension> derivative,
    double scale
) {
    PoweredDescent6DofState result{};
    for (std::size_t index = 0; index < powered_descent_6dof_state_dimension; ++index) {
        result[index] = state[index] + scale * derivative[index];
    }
    return result;
}

}  // namespace

void PoweredDescent6DofConfig::validate() const {
    require_finite(gravity, "six-DoF gravity");
    require_finite(principal_inertia, "principal inertia");
    if (std::ranges::any_of(principal_inertia, [](double value) { return value <= 0.0; })) {
        throw std::invalid_argument("principal inertia components must be positive");
    }
    for (const auto& item : std::array{
             std::pair{"mass_flow_coefficient", mass_flow_coefficient},
             std::pair{"minimum_mass", minimum_mass},
             std::pair{"maximum_thrust", maximum_thrust},
             std::pair{"maximum_torque", maximum_torque},
             std::pair{"maximum_angular_rate", maximum_angular_rate},
         }) {
        if (!std::isfinite(item.second) || item.second <= 0.0) {
            throw std::invalid_argument(std::string(item.first) + " must be finite and positive");
        }
    }
    if (!std::isfinite(minimum_sigma) || minimum_sigma < 0.0 ||
        minimum_sigma > maximum_thrust) {
        throw std::invalid_argument("minimum sigma must lie in [0, maximum thrust]");
    }
    const double half_pi = 0.5 * std::numbers::pi_v<double>;
    if (!(maximum_tilt_radians > 0.0 && maximum_tilt_radians < half_pi) ||
        !(glide_slope_radians > 0.0 && glide_slope_radians < half_pi)) {
        throw std::invalid_argument("six-DoF tilt and glide angles must lie in (0, pi/2)");
    }
}

double PoweredDescent6DofConfig::tilt_cosine() const {
    return std::cos(maximum_tilt_radians);
}

double PoweredDescent6DofConfig::glide_slope_tangent() const {
    return std::tan(glide_slope_radians);
}

double PoweredDescent6DofPathDiagnostics::maximum_violation() const noexcept {
    return std::max({
        thrust_epigraph,
        throttle_lower,
        throttle_upper,
        torque,
        tilt,
        angular_rate,
        minimum_mass,
        altitude,
        glide_slope,
        quaternion_norm,
    });
}

RotationMatrix3 quaternion_rotation_matrix(std::span<const double, 4> quaternion) {
    require_finite(quaternion, "quaternion");
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];

    RotationMatrix3 rotation{};
    entry<3>(rotation, 0, 0) = 1.0 - 2.0 * (y * y + z * z);
    entry<3>(rotation, 0, 1) = 2.0 * (x * y - w * z);
    entry<3>(rotation, 0, 2) = 2.0 * (x * z + w * y);
    entry<3>(rotation, 1, 0) = 2.0 * (x * y + w * z);
    entry<3>(rotation, 1, 1) = 1.0 - 2.0 * (x * x + z * z);
    entry<3>(rotation, 1, 2) = 2.0 * (y * z - w * x);
    entry<3>(rotation, 2, 0) = 2.0 * (x * z - w * y);
    entry<3>(rotation, 2, 1) = 2.0 * (y * z + w * x);
    entry<3>(rotation, 2, 2) = 1.0 - 2.0 * (x * x + y * y);
    return rotation;
}

std::array<double, 4> normalise_quaternion(std::span<const double, 4> quaternion) {
    require_finite(quaternion, "quaternion");
    double norm_squared = 0.0;
    for (double value : quaternion) {
        norm_squared += value * value;
    }
    const double norm = std::sqrt(norm_squared);
    if (norm <= 1.0e-15) {
        throw std::invalid_argument("quaternion norm is too small to normalise");
    }
    return {
        quaternion[0] / norm,
        quaternion[1] / norm,
        quaternion[2] / norm,
        quaternion[3] / norm,
    };
}

PoweredDescent6DofModel::PoweredDescent6DofModel(PoweredDescent6DofConfig config)
    : config_(std::move(config)) {
    config_.validate();
}

void PoweredDescent6DofModel::require_state(
    std::span<const double, powered_descent_6dof_state_dimension> state
) {
    require_finite(state, "six-DoF state");
    if (state[13] <= 0.0) {
        throw std::invalid_argument("six-DoF mass must be positive");
    }
    double quaternion_norm_squared = 0.0;
    for (std::size_t index = 6; index < 10; ++index) {
        quaternion_norm_squared += state[index] * state[index];
    }
    if (quaternion_norm_squared <= 1.0e-30) {
        throw std::invalid_argument("six-DoF quaternion norm is too small");
    }
}

void PoweredDescent6DofModel::require_control(
    std::span<const double, powered_descent_6dof_control_dimension> control
) {
    require_finite(control, "six-DoF control");
}

PoweredDescent6DofState PoweredDescent6DofModel::dynamics(
    std::span<const double, powered_descent_6dof_state_dimension> state,
    std::span<const double, powered_descent_6dof_control_dimension> control
) const {
    require_state(state);
    require_control(control);
    const std::span<const double, 4> quaternion{state.data() + 6, 4};
    const std::span<const double, 3> body_thrust{control.data(), 3};
    const auto inertial_thrust = rotate(quaternion_rotation_matrix(quaternion), body_thrust);
    const double mass = state[13];

    PoweredDescent6DofState derivative{};
    derivative[0] = state[3];
    derivative[1] = state[4];
    derivative[2] = state[5];
    for (std::size_t axis = 0; axis < 3; ++axis) {
        derivative[3 + axis] = inertial_thrust[axis] / mass + config_.gravity[axis];
    }

    const double w = state[6];
    const double x = state[7];
    const double y = state[8];
    const double z = state[9];
    const double wx = state[10];
    const double wy = state[11];
    const double wz = state[12];
    derivative[6] = -0.5 * (x * wx + y * wy + z * wz);
    derivative[7] = 0.5 * (w * wx + y * wz - z * wy);
    derivative[8] = 0.5 * (w * wy + z * wx - x * wz);
    derivative[9] = 0.5 * (w * wz + x * wy - y * wx);

    const double ix = config_.principal_inertia[0];
    const double iy = config_.principal_inertia[1];
    const double iz = config_.principal_inertia[2];
    derivative[10] = ((iy - iz) * wy * wz + control[3]) / ix;
    derivative[11] = ((iz - ix) * wz * wx + control[4]) / iy;
    derivative[12] = ((ix - iy) * wx * wy + control[5]) / iz;
    derivative[13] = -config_.mass_flow_coefficient * control[6];
    return derivative;
}

PoweredDescent6DofLinearisation PoweredDescent6DofModel::linearise(
    std::span<const double, powered_descent_6dof_state_dimension> state,
    std::span<const double, powered_descent_6dof_control_dimension> control
) const {
    require_state(state);
    require_control(control);
    PoweredDescent6DofLinearisation result{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        entry<powered_descent_6dof_state_dimension>(
            result.state_jacobian,
            axis,
            3 + axis
        ) = 1.0;
    }

    const std::span<const double, 4> quaternion{state.data() + 6, 4};
    const std::span<const double, 3> body_thrust{control.data(), 3};
    const auto rotation = quaternion_rotation_matrix(quaternion);
    const auto derivatives = rotation_derivatives(quaternion);
    const auto inertial_thrust = rotate(rotation, body_thrust);
    const double mass = state[13];
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t quaternion_component = 0;
             quaternion_component < 4;
             ++quaternion_component) {
            double value = 0.0;
            for (std::size_t column = 0; column < 3; ++column) {
                value += entry<3>(
                    derivatives[quaternion_component],
                    row,
                    column
                ) * body_thrust[column];
            }
            entry<powered_descent_6dof_state_dimension>(
                result.state_jacobian,
                3 + row,
                6 + quaternion_component
            ) = value / mass;
        }
        entry<powered_descent_6dof_state_dimension>(
            result.state_jacobian,
            3 + row,
            13
        ) = -inertial_thrust[row] / (mass * mass);
        for (std::size_t column = 0; column < 3; ++column) {
            entry<powered_descent_6dof_control_dimension>(
                result.control_jacobian,
                3 + row,
                column
            ) = entry<3>(rotation, row, column) / mass;
        }
    }

    const double w = state[6];
    const double x = state[7];
    const double y = state[8];
    const double z = state[9];
    const double wx = state[10];
    const double wy = state[11];
    const double wz = state[12];

    const std::array<double, 16> omega_matrix{
        0.0, -wx, -wy, -wz,
        wx, 0.0, wz, -wy,
        wy, -wz, 0.0, wx,
        wz, wy, -wx, 0.0,
    };
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            entry<powered_descent_6dof_state_dimension>(
                result.state_jacobian,
                6 + row,
                6 + column
            ) = 0.5 * entry<4>(omega_matrix, row, column);
        }
    }

    const std::array<double, 12> quaternion_rate_by_omega{
        -x, -y, -z,
        w, -z, y,
        z, w, -x,
        -y, x, w,
    };
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            entry<powered_descent_6dof_state_dimension>(
                result.state_jacobian,
                6 + row,
                10 + column
            ) = 0.5 * entry<3>(quaternion_rate_by_omega, row, column);
        }
    }

    const double ix = config_.principal_inertia[0];
    const double iy = config_.principal_inertia[1];
    const double iz = config_.principal_inertia[2];
    entry<powered_descent_6dof_state_dimension>(result.state_jacobian, 10, 11) =
        (iy - iz) * wz / ix;
    entry<powered_descent_6dof_state_dimension>(result.state_jacobian, 10, 12) =
        (iy - iz) * wy / ix;
    entry<powered_descent_6dof_state_dimension>(result.state_jacobian, 11, 10) =
        (iz - ix) * wz / iy;
    entry<powered_descent_6dof_state_dimension>(result.state_jacobian, 11, 12) =
        (iz - ix) * wx / iy;
    entry<powered_descent_6dof_state_dimension>(result.state_jacobian, 12, 10) =
        (ix - iy) * wy / iz;
    entry<powered_descent_6dof_state_dimension>(result.state_jacobian, 12, 11) =
        (ix - iy) * wx / iz;

    entry<powered_descent_6dof_control_dimension>(result.control_jacobian, 10, 3) =
        1.0 / ix;
    entry<powered_descent_6dof_control_dimension>(result.control_jacobian, 11, 4) =
        1.0 / iy;
    entry<powered_descent_6dof_control_dimension>(result.control_jacobian, 12, 5) =
        1.0 / iz;
    entry<powered_descent_6dof_control_dimension>(result.control_jacobian, 13, 6) =
        -config_.mass_flow_coefficient;

    const auto derivative = dynamics(state, control);
    for (std::size_t row = 0; row < powered_descent_6dof_state_dimension; ++row) {
        double linearised = 0.0;
        for (std::size_t column = 0;
             column < powered_descent_6dof_state_dimension;
             ++column) {
            linearised += entry<powered_descent_6dof_state_dimension>(
                result.state_jacobian,
                row,
                column
            ) * state[column];
        }
        for (std::size_t column = 0;
             column < powered_descent_6dof_control_dimension;
             ++column) {
            linearised += entry<powered_descent_6dof_control_dimension>(
                result.control_jacobian,
                row,
                column
            ) * control[column];
        }
        result.offset[row] = derivative[row] - linearised;
    }
    return result;
}

PoweredDescent6DofState PoweredDescent6DofModel::rk4_step(
    std::span<const double, powered_descent_6dof_state_dimension> state,
    std::span<const double, powered_descent_6dof_control_dimension> control,
    double step_seconds,
    bool renormalise_attitude
) const {
    require_state(state);
    require_control(control);
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("six-DoF integration step must be finite and positive");
    }
    const auto first = dynamics(state, control);
    const auto second_state = add_scaled(state, first, 0.5 * step_seconds);
    const auto second = dynamics(second_state, control);
    const auto third_state = add_scaled(state, second, 0.5 * step_seconds);
    const auto third = dynamics(third_state, control);
    const auto fourth_state = add_scaled(state, third, step_seconds);
    const auto fourth = dynamics(fourth_state, control);

    PoweredDescent6DofState result{};
    for (std::size_t index = 0; index < powered_descent_6dof_state_dimension; ++index) {
        result[index] = state[index] + step_seconds / 6.0 *
            (first[index] + 2.0 * second[index] + 2.0 * third[index] + fourth[index]);
    }
    if (renormalise_attitude) {
        const std::span<const double, 4> quaternion{result.data() + 6, 4};
        const auto normalised = normalise_quaternion(quaternion);
        std::copy(normalised.begin(), normalised.end(), result.begin() + 6);
    }
    require_state(result);
    return result;
}

std::vector<PoweredDescent6DofState> PoweredDescent6DofModel::rollout_rk4(
    std::span<const double, powered_descent_6dof_state_dimension> initial_state,
    std::span<const PoweredDescent6DofControl> controls,
    double step_seconds,
    bool renormalise_attitude
) const {
    require_state(initial_state);
    std::vector<PoweredDescent6DofState> result(controls.size() + 1U);
    std::copy(initial_state.begin(), initial_state.end(), result.front().begin());
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        result[interval + 1U] = rk4_step(
            result[interval],
            controls[interval],
            step_seconds,
            renormalise_attitude
        );
    }
    return result;
}

PoweredDescent6DofPathDiagnostics PoweredDescent6DofModel::path_diagnostics(
    std::span<const PoweredDescent6DofState> states,
    std::span<const PoweredDescent6DofControl> controls
) const {
    if (states.empty() || states.size() != controls.size() + 1U) {
        throw std::invalid_argument("six-DoF path requires one more state than control");
    }
    PoweredDescent6DofPathDiagnostics result{};
    for (const auto& state : states) {
        require_state(state);
        const double horizontal = std::hypot(state[0], state[1]);
        const double angular_rate = std::hypot(state[10], state[11], state[12]);
        const double quaternion_norm =
            std::sqrt(state[6] * state[6] + state[7] * state[7] +
                      state[8] * state[8] + state[9] * state[9]);
        result.angular_rate = std::max(
            result.angular_rate,
            angular_rate - config_.maximum_angular_rate
        );
        result.minimum_mass = std::max(result.minimum_mass, config_.minimum_mass - state[13]);
        result.altitude = std::max(result.altitude, -state[2]);
        result.glide_slope = std::max(
            result.glide_slope,
            horizontal - config_.glide_slope_tangent() * state[2]
        );
        result.quaternion_norm = std::max(
            result.quaternion_norm,
            std::abs(quaternion_norm - 1.0)
        );
    }
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        const auto& control = controls[interval];
        require_control(control);
        const double thrust_norm = std::hypot(control[0], control[1], control[2]);
        const double torque_norm = std::hypot(control[3], control[4], control[5]);
        const std::span<const double, 4> quaternion{states[interval].data() + 6, 4};
        const std::span<const double, 3> body_thrust{control.data(), 3};
        const auto inertial_thrust = rotate(
            quaternion_rotation_matrix(quaternion),
            body_thrust
        );
        result.thrust_epigraph = std::max(result.thrust_epigraph, thrust_norm - control[6]);
        result.throttle_lower = std::max(result.throttle_lower, config_.minimum_sigma - control[6]);
        result.throttle_upper = std::max(result.throttle_upper, control[6] - config_.maximum_thrust);
        result.torque = std::max(result.torque, torque_norm - config_.maximum_torque);
        result.tilt = std::max(
            result.tilt,
            config_.tilt_cosine() * control[6] - inertial_thrust[2]
        );
    }

    result.thrust_epigraph = std::max(result.thrust_epigraph, 0.0);
    result.throttle_lower = std::max(result.throttle_lower, 0.0);
    result.throttle_upper = std::max(result.throttle_upper, 0.0);
    result.torque = std::max(result.torque, 0.0);
    result.tilt = std::max(result.tilt, 0.0);
    result.angular_rate = std::max(result.angular_rate, 0.0);
    result.minimum_mass = std::max(result.minimum_mass, 0.0);
    result.altitude = std::max(result.altitude, 0.0);
    result.glide_slope = std::max(result.glide_slope, 0.0);
    return result;
}

}  // namespace spacepdhcg::native
