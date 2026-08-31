#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace spacepdhcg::native {

inline constexpr std::size_t powered_descent_state_dimension = 7;
inline constexpr std::size_t powered_descent_control_dimension = 4;

using PoweredDescentState = std::array<double, powered_descent_state_dimension>;
using PoweredDescentControl = std::array<double, powered_descent_control_dimension>;
using PoweredDescentStateMatrix = std::array<
    double,
    powered_descent_state_dimension * powered_descent_state_dimension
>;
using PoweredDescentControlMatrix = std::array<
    double,
    powered_descent_state_dimension * powered_descent_control_dimension
>;

struct PoweredDescent3DofConfig {
    std::array<double, 3> gravity{0.0, 0.0, -3.711};
    double mass_flow_coefficient{4.6e-4};
    double minimum_mass{1'000.0};
    double maximum_thrust{15'000.0};
    double minimum_sigma{0.0};
    double maximum_tilt_radians{0.52359877559829887308};
    double glide_slope_radians{1.04719755119659774615};

    void validate() const;
    [[nodiscard]] double tilt_cosine() const;
    [[nodiscard]] double glide_slope_tangent() const;
};

struct PoweredDescentLinearisation {
    PoweredDescentStateMatrix state_jacobian{};
    PoweredDescentControlMatrix control_jacobian{};
    PoweredDescentState offset{};
};

struct PoweredDescentDiscreteLinearisation {
    PoweredDescentStateMatrix state_matrix{};
    PoweredDescentControlMatrix control_matrix{};
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

    [[nodiscard]] double maximum_violation() const noexcept;
};

class PoweredDescent3DofModel {
  public:
    explicit PoweredDescent3DofModel(PoweredDescent3DofConfig config = {});

    [[nodiscard]] const PoweredDescent3DofConfig& config() const noexcept { return config_; }

    [[nodiscard]] PoweredDescentState dynamics(
        std::span<const double, powered_descent_state_dimension> state,
        std::span<const double, powered_descent_control_dimension> control
    ) const;

    [[nodiscard]] PoweredDescentLinearisation linearise(
        std::span<const double, powered_descent_state_dimension> state,
        std::span<const double, powered_descent_control_dimension> control
    ) const;

    [[nodiscard]] PoweredDescentDiscreteLinearisation linearised_euler(
        std::span<const double, powered_descent_state_dimension> state,
        std::span<const double, powered_descent_control_dimension> control,
        double step_seconds
    ) const;

    [[nodiscard]] PoweredDescentState euler_step(
        std::span<const double, powered_descent_state_dimension> state,
        std::span<const double, powered_descent_control_dimension> control,
        double step_seconds
    ) const;

    [[nodiscard]] PoweredDescentState rk4_step(
        std::span<const double, powered_descent_state_dimension> state,
        std::span<const double, powered_descent_control_dimension> control,
        double step_seconds
    ) const;

    [[nodiscard]] std::vector<PoweredDescentState> rollout_euler(
        std::span<const double, powered_descent_state_dimension> initial_state,
        std::span<const PoweredDescentControl> controls,
        double step_seconds
    ) const;

    [[nodiscard]] std::vector<PoweredDescentState> rollout_rk4(
        std::span<const double, powered_descent_state_dimension> initial_state,
        std::span<const PoweredDescentControl> controls,
        double step_seconds
    ) const;

    [[nodiscard]] PoweredDescentPathDiagnostics path_diagnostics(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls
    ) const;

  private:
    PoweredDescent3DofConfig config_{};

    static void require_state(std::span<const double, powered_descent_state_dimension> state);
    static void require_control(
        std::span<const double, powered_descent_control_dimension> control
    );
    static void require_step(double step_seconds);
};

}  // namespace spacepdhcg::native
