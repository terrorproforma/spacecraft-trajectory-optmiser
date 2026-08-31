#pragma once

#include "spacepdhcg/native/cqp.hpp"
#include "spacepdhcg/native/cw.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace spacepdhcg::native {

enum class CwThrustConstraint {
    box,
    second_order_cone,
};

struct CwRendezvousConfig {
    Index intervals{40};
    double step_seconds{20.0};
    double mean_motion{1.13e-3};
    double maximum_acceleration{5.0e-2};
    CwThrustConstraint thrust_constraint{CwThrustConstraint::second_order_cone};
    CwState state_weights{1.0e-4, 1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2, 1.0e-2};
    CwControl control_weights{1.0, 1.0, 1.0};

    void validate() const;
};

struct CwRendezvousLayout {
    Index intervals{0};
    CwThrustConstraint thrust_constraint{CwThrustConstraint::second_order_cone};

    [[nodiscard]] Index state_variables() const noexcept;
    [[nodiscard]] Index control_variables() const noexcept;
    [[nodiscard]] Index variables() const noexcept;
    [[nodiscard]] Index scalar_constraints() const noexcept;
    [[nodiscard]] Index affine_rows() const noexcept;
    [[nodiscard]] Index initial_row() const noexcept { return 0; }
    [[nodiscard]] Index dynamics_row() const noexcept;
    [[nodiscard]] Index terminal_row() const noexcept;
    [[nodiscard]] Index control_row() const noexcept;
    [[nodiscard]] Index state_offset(Index node) const;
    [[nodiscard]] Index control_offset(Index interval) const;
};

struct CwRendezvousDiagnostics {
    double initial_error{0.0};
    double terminal_error{0.0};
    double dynamics_defect{0.0};
    double control_violation{0.0};
    double maximum_component_acceleration{0.0};
    double maximum_acceleration_norm{0.0};

    [[nodiscard]] double maximum_violation() const noexcept;
};

class CwRendezvousProblem {
  public:
    explicit CwRendezvousProblem(CwRendezvousConfig config = {});

    [[nodiscard]] const CwRendezvousConfig& config() const noexcept { return config_; }
    [[nodiscard]] const CwRendezvousLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const CwDiscreteDynamics& dynamics() const noexcept { return dynamics_; }

    [[nodiscard]] OwnedCqp make_cqp(
        std::span<const double, cw_state_dimension> initial_state,
        std::span<const double, cw_state_dimension> target_state
    ) const;

    void update_numerical_values(
        OwnedCqp& problem,
        std::span<const double, cw_state_dimension> initial_state,
        std::span<const double, cw_state_dimension> target_state
    ) const;

    [[nodiscard]] CwRendezvousDiagnostics diagnostics(
        std::span<const double> decision,
        std::span<const double, cw_state_dimension> initial_state,
        std::span<const double, cw_state_dimension> target_state
    ) const;

  private:
    CwRendezvousConfig config_{};
    CwRendezvousLayout layout_{};
    CwDiscreteDynamics dynamics_{};
    OwnedCqp prototype_{};

    [[nodiscard]] OwnedCqp build_prototype() const;
    void assert_compatible(const OwnedCqp& problem) const;
};

}  // namespace spacepdhcg::native
