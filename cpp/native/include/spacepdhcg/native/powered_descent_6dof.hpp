#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace spacepdhcg::native {

inline constexpr std::size_t powered_descent_6dof_state_dimension = 14;
inline constexpr std::size_t powered_descent_6dof_control_dimension = 7;

using PoweredDescent6DofState = std::array<
    double,
    powered_descent_6dof_state_dimension
>;
using PoweredDescent6DofControl = std::array<
    double,
    powered_descent_6dof_control_dimension
>;
using PoweredDescent6DofStateMatrix = std::array<
    double,
    powered_descent_6dof_state_dimension * powered_descent_6dof_state_dimension
>;
using PoweredDescent6DofControlMatrix = std::array<
    double,
    powered_descent_6dof_state_dimension * powered_descent_6dof_control_dimension
>;
using RotationMatrix3 = std::array<double, 9>;

struct PoweredDescent6DofConfig {
    std::array<double, 3> gravity{0.0, 0.0, -3.711};
    std::array<double, 3> principal_inertia{900.0, 1'000.0, 800.0};
    double mass_flow_coefficient{4.6e-4};
    double minimum_mass{1'000.0};
    double maximum_thrust{15'000.0};
    double minimum_sigma{0.0};
    double maximum_torque{2'000.0};
    double maximum_angular_rate{0.75};
    double maximum_tilt_radians{0.52359877559829887308};
    double glide_slope_radians{1.04719755119659774615};

    void validate() const;
    [[nodiscard]] double tilt_cosine() const;
    [[nodiscard]] double glide_slope_tangent() const;
};

struct PoweredDescent6DofLinearisation {
    PoweredDescent6DofStateMatrix state_jacobian{};
    PoweredDescent6DofControlMatrix control_jacobian{};
    PoweredDescent6DofState offset{};
};

struct PoweredDescent6DofPathDiagnostics {
    double thrust_epigraph{0.0};
    double throttle_lower{0.0};
    double throttle_upper{0.0};
    double torque{0.0};
    double tilt{0.0};
    double angular_rate{0.0};
    double minimum_mass{0.0};
    double altitude{0.0};
    double glide_slope{0.0};
    double quaternion_norm{0.0};

    [[nodiscard]] double maximum_violation() const noexcept;
};

[[nodiscard]] RotationMatrix3 quaternion_rotation_matrix(
    std::span<const double, 4> quaternion
);

[[nodiscard]] std::array<double, 4> normalise_quaternion(
    std::span<const double, 4> quaternion
);

class PoweredDescent6DofModel {
  public:
    explicit PoweredDescent6DofModel(PoweredDescent6DofConfig config = {});

    [[nodiscard]] const PoweredDescent6DofConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] PoweredDescent6DofState dynamics(
        std::span<const double, powered_descent_6dof_state_dimension> state,
        std::span<const double, powered_descent_6dof_control_dimension> control
    ) const;

    [[nodiscard]] PoweredDescent6DofLinearisation linearise(
        std::span<const double, powered_descent_6dof_state_dimension> state,
        std::span<const double, powered_descent_6dof_control_dimension> control
    ) const;

    [[nodiscard]] PoweredDescent6DofState rk4_step(
        std::span<const double, powered_descent_6dof_state_dimension> state,
        std::span<const double, powered_descent_6dof_control_dimension> control,
        double step_seconds,
        bool renormalise_attitude = true
    ) const;

    [[nodiscard]] std::vector<PoweredDescent6DofState> rollout_rk4(
        std::span<const double, powered_descent_6dof_state_dimension> initial_state,
        std::span<const PoweredDescent6DofControl> controls,
        double step_seconds,
        bool renormalise_attitude = true
    ) const;

    [[nodiscard]] PoweredDescent6DofPathDiagnostics path_diagnostics(
        std::span<const PoweredDescent6DofState> states,
        std::span<const PoweredDescent6DofControl> controls
    ) const;

  private:
    PoweredDescent6DofConfig config_{};

    static void require_state(
        std::span<const double, powered_descent_6dof_state_dimension> state
    );
    static void require_control(
        std::span<const double, powered_descent_6dof_control_dimension> control
    );
};

}  // namespace spacepdhcg::native
