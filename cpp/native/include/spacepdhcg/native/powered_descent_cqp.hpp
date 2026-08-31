#pragma once

#include "spacepdhcg/native/cqp.hpp"
#include "spacepdhcg/native/powered_descent_3dof.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace spacepdhcg::native {

struct PoweredDescentCqpConfig {
    Index intervals{10};
    double step_seconds{2.0};
    double trust_radius{1.0};
    double virtual_l1_weight{1.0e5};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{1.0e-3};
    PoweredDescentState state_tracking_weights{
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-8,
    };
    PoweredDescentControl control_tracking_weights{
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
    };
    PoweredDescentState state_trust_scales{
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-3,
    };
    PoweredDescentControl control_trust_scales{
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 15'000.0,
    };

    void validate() const;
};

struct PoweredDescentCqpLayout {
    Index intervals{0};

    [[nodiscard]] Index state_count() const noexcept;
    [[nodiscard]] Index control_count() const noexcept;
    [[nodiscard]] Index virtual_count() const noexcept;
    [[nodiscard]] Index virtual_epigraph_count() const noexcept;
    [[nodiscard]] Index control_offset() const noexcept;
    [[nodiscard]] Index virtual_offset() const noexcept;
    [[nodiscard]] Index virtual_epigraph_offset() const noexcept;
    [[nodiscard]] Index variables() const noexcept;

    [[nodiscard]] Index initial_row() const noexcept { return 0; }
    [[nodiscard]] Index dynamics_row() const noexcept;
    [[nodiscard]] Index terminal_row() const noexcept;
    [[nodiscard]] Index virtual_epigraph_row() const noexcept;
    [[nodiscard]] Index tilt_row() const noexcept;
    [[nodiscard]] Index scalar_rows() const noexcept;

    [[nodiscard]] Index thrust_cone_row() const noexcept { return 0; }
    [[nodiscard]] Index glide_cone_row() const noexcept;
    [[nodiscard]] Index stage_trust_cone_row() const noexcept;
    [[nodiscard]] Index terminal_trust_cone_row() const noexcept;
    [[nodiscard]] Index affine_rows() const noexcept;

    [[nodiscard]] Index state_offset(Index node) const;
    [[nodiscard]] Index control_offset(Index interval) const;
    [[nodiscard]] Index virtual_offset(Index interval) const;
    [[nodiscard]] Index virtual_epigraph_offset(Index interval) const;
};

struct PoweredDescentDecision {
    std::vector<PoweredDescentState> states{};
    std::vector<PoweredDescentControl> controls{};
    std::vector<PoweredDescentState> virtual_controls{};
    std::vector<PoweredDescentState> virtual_epigraphs{};
};

struct PoweredDescentCqpDiagnostics {
    CqpDiagnostics convex{};
    double linearised_dynamics_defect{0.0};
    double nonlinear_dynamics_defect{0.0};
    double terminal_error{0.0};
    double virtual_control{0.0};

    [[nodiscard]] double maximum_convex_violation() const noexcept {
        return convex.maximum_violation();
    }
};

class PoweredDescentCqp {
  public:
    explicit PoweredDescentCqp(
        PoweredDescent3DofModel model = {},
        PoweredDescentCqpConfig config = {}
    );

    [[nodiscard]] const PoweredDescent3DofModel& model() const noexcept { return model_; }
    [[nodiscard]] const PoweredDescentCqpConfig& config() const noexcept { return config_; }
    [[nodiscard]] const PoweredDescentCqpLayout& layout() const noexcept { return layout_; }

    [[nodiscard]] OwnedCqp make_cqp(
        std::span<const PoweredDescentState> reference_states,
        std::span<const PoweredDescentControl> reference_controls,
        std::span<const double, powered_descent_state_dimension> initial_state,
        std::span<const double, 3> target_position,
        std::span<const double, 3> target_velocity,
        double trust_radius
    ) const;

    void update_numerical_values(
        OwnedCqp& problem,
        std::span<const PoweredDescentState> reference_states,
        std::span<const PoweredDescentControl> reference_controls,
        std::span<const double, powered_descent_state_dimension> initial_state,
        std::span<const double, 3> target_position,
        std::span<const double, 3> target_velocity,
        double trust_radius
    ) const;

    [[nodiscard]] PoweredDescentDecision decode(std::span<const double> decision) const;

    [[nodiscard]] PoweredDescentCqpDiagnostics diagnostics(
        std::span<const double> decision,
        const OwnedCqp& problem,
        std::span<const double, 3> target_position,
        std::span<const double, 3> target_velocity
    ) const;

  private:
    PoweredDescent3DofModel model_{};
    PoweredDescentCqpConfig config_{};
    PoweredDescentCqpLayout layout_{};
    OwnedCqp prototype_{};

    [[nodiscard]] OwnedCqp build_prototype() const;
    void assert_compatible(const OwnedCqp& problem) const;
};

}  // namespace spacepdhcg::native
