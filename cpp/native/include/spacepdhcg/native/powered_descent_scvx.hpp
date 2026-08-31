#pragma once

#include "spacepdhcg/native/backend.hpp"
#include "spacepdhcg/native/persistent_session.hpp"
#include "spacepdhcg/native/powered_descent_cqp.hpp"
#include "spacepdhcg/native/scvx_policy.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace spacepdhcg::native {

struct PoweredDescentOuterConfig {
    std::size_t maximum_iterations{15};
    std::size_t minimum_iterations{2};
    double convergence_tolerance{2.0e-4};
    double step_tolerance{2.0e-2};
    double acceptance_threshold{0.05};
    double feasibility_penalty{100.0};
    double virtual_penalty{100.0};
    double minimum_actual_reduction{1.0e-10};
    double minimum_predicted_reduction{1.0e-12};
    double restoration_reduction{0.9};
    std::size_t maximum_resolves_per_iteration{1};

    void validate() const;
};

struct PoweredDescentReference {
    std::vector<PoweredDescentState> states{};
    std::vector<PoweredDescentControl> controls{};
};

[[nodiscard]] PoweredDescentReference make_powered_descent_reference(
    const PoweredDescent3DofModel& model,
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    std::size_t intervals,
    double step_seconds
);

struct PoweredDescentScvxIteration {
    std::size_t iteration{0};
    SolvePhase phase{SolvePhase::repair};
    double requested_tolerance{0.0};
    double effective_tolerance{0.0};
    std::size_t solver_iterations{0};
    CqpSolveStatus solver_status{CqpSolveStatus::internal_error};
    double primal_residual{0.0};
    double dual_residual{0.0};
    double trust_radius_before{0.0};
    double trust_radius_after{0.0};
    TrustAction trust_action{TrustAction::keep};
    double step_fraction{0.0};
    double predicted_reduction{0.0};
    double actual_reduction{0.0};
    double agreement{0.0};
    bool accepted{false};
    bool restoration_accepted{false};
    bool re_solved{false};
    bool warm_started{false};
    double merit_before{0.0};
    double merit_after{0.0};
    OuterResidual residual{};
    PoweredDescentCqpDiagnostics convex{};
};

struct PoweredDescentScvxResult {
    std::string status{};
    std::vector<PoweredDescentState> states{};
    std::vector<PoweredDescentControl> controls{};
    double merit{0.0};
    OuterResidual residual{};
    PoweredDescentPathDiagnostics path{};
    std::vector<PoweredDescentScvxIteration> iterations{};
    std::size_t accepted_iterations{0};
    std::size_t workspace_updates{0};
    std::size_t workspace_solves{0};

    [[nodiscard]] bool converged() const noexcept { return status == "converged"; }
};

class PoweredDescentScvxDriver {
  public:
    PoweredDescentScvxDriver(
        PoweredDescentCqp transcription,
        CqpWorkspaceFactory backend_factory,
        PoweredDescentOuterConfig outer_config = {},
        ForcingRuleConfig forcing_config = {},
        TrustRegionConfig trust_config = {}
    );

    [[nodiscard]] PoweredDescentScvxResult solve(
        std::span<const double, powered_descent_state_dimension> initial_state,
        std::span<const double, 3> target_position,
        std::span<const double, 3> target_velocity,
        const PoweredDescentReference* initial_reference = nullptr
    ) const;

  private:
    PoweredDescentCqp transcription_;
    CqpWorkspaceFactory backend_factory_;
    PoweredDescentOuterConfig outer_config_{};
    ForcingRuleConfig forcing_config_{};
    TrustRegionConfig trust_config_{};
};

}  // namespace spacepdhcg::native
