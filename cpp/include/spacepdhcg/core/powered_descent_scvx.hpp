#pragma once

#include "spacepdhcg/core/host_pdhg.hpp"
#include "spacepdhcg/core/powered_descent_cqp.hpp"
#include "spacepdhcg/core/scvx_policy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

struct PoweredDescentHostSCvxConfig {
    std::size_t maximum_iterations{15};
    std::size_t minimum_iterations{1};
    double convergence_tolerance{2.0e-4};
    double step_tolerance{2.0e-2};
    double acceptance_threshold{0.05};
    double feasibility_penalty{100.0};
    double virtual_penalty{100.0};
    double minimum_actual_reduction{1.0e-10};
    double minimum_predicted_reduction{1.0e-12};
    double restoration_reduction{0.9};
    std::size_t residual_check_interval{25};
    std::size_t norm_iterations{30};

    void validate() const {
        if (maximum_iterations == 0U || minimum_iterations > maximum_iterations ||
            residual_check_interval == 0U || norm_iterations == 0U) {
            throw std::invalid_argument("host SCvx iteration counts are invalid");
        }
        const double positive[]{
            convergence_tolerance,
            step_tolerance,
            feasibility_penalty,
            virtual_penalty,
            minimum_actual_reduction,
            minimum_predicted_reduction,
        };
        for (const auto value : positive) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("host SCvx positive parameters are invalid");
            }
        }
        if (!std::isfinite(acceptance_threshold) || acceptance_threshold < 0.0 ||
            acceptance_threshold >= 1.0) {
            throw std::invalid_argument("host SCvx acceptance threshold must lie in [0,1)");
        }
        if (!std::isfinite(restoration_reduction) || restoration_reduction <= 0.0 ||
            restoration_reduction >= 1.0) {
            throw std::invalid_argument("host SCvx restoration factor must lie in (0,1)");
        }
    }
};

struct PoweredDescentHostSCvxIteration {
    std::size_t iteration{0};
    SolvePhase phase{SolvePhase::exploration};
    double requested_tolerance{0.0};
    std::size_t solver_iterations{0};
    std::string solver_status;
    double primal_residual{0.0};
    double dual_residual{0.0};
    double trust_radius_before{0.0};
    double trust_radius_after{0.0};
    RadiusAction trust_action{RadiusAction::keep};
    double step_fraction{0.0};
    double predicted_reduction{0.0};
    double actual_reduction{0.0};
    double agreement{0.0};
    bool accepted{false};
    bool restoration_accepted{false};
    OuterResidual residual{};
    PoweredDescentCQPDiagnostics convex_diagnostics{};
    double solve_seconds{0.0};
};

struct PoweredDescentHostSCvxResult {
    std::string status{"maximum_iterations"};
    std::vector<PoweredDescentState> states;
    std::vector<PoweredDescentControl> controls;
    OuterResidual residual{};
    PoweredDescentPathDiagnostics path{};
    std::vector<PoweredDescentHostSCvxIteration> iterations;
    std::size_t accepted_iterations{0};
    std::size_t workspace_updates{0};
    std::size_t warm_starts{0};
    std::size_t inner_solves{0};
    double total_solve_seconds{0.0};

    [[nodiscard]] bool converged() const noexcept { return status == "converged"; }
};

class PoweredDescentHostSCvx {
  public:
    explicit PoweredDescentHostSCvx(
        PoweredDescentCQP subproblem,
        PoweredDescentHostSCvxConfig outer_config = {},
        ForcingRuleConfig forcing_config = {},
        TrustRegionConfig trust_config = {}
    )
        : subproblem_(std::move(subproblem)),
          outer_config_(outer_config),
          forcing_(forcing_config),
          trust_(normalise_trust_config(trust_config, subproblem_.config().trust_radius)) {
        outer_config_.validate();
    }

    [[nodiscard]] PoweredDescentHostSCvxResult solve(
        const PoweredDescentState& initial_state,
        const std::array<double, 3>& target_position,
        const std::array<double, 3>& target_velocity,
        std::vector<PoweredDescentState> reference_states,
        std::vector<PoweredDescentControl> reference_controls
    ) {
        auto current_values = subproblem_.values(
            reference_states,
            reference_controls,
            initial_state,
            target_position,
            target_velocity,
            trust_.radius()
        );
        PersistentHostPDHG workspace{subproblem_.structure(), current_values};
        auto warm_primal = subproblem_.reference_decision(reference_states, reference_controls);
        workspace.warm_start(warm_primal);

        auto current_rollout = subproblem_.model().rollout(
            initial_state,
            reference_controls,
            subproblem_.config().step_seconds,
            false
        );
        double current_merit = actual_merit(
            current_rollout,
            reference_controls,
            target_position,
            target_velocity
        );
        OuterResidual current_residual = outer_residual(
            reference_states,
            reference_controls,
            current_rollout,
            target_position,
            target_velocity,
            1.0
        );

        PoweredDescentHostSCvxResult result;
        result.states = reference_states;
        result.controls = reference_controls;
        result.residual = current_residual;
        std::size_t accepted_streak = 0U;
        double previous_agreement = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> warm_dual;

        for (std::size_t iteration = 0; iteration < outer_config_.maximum_iterations;
             ++iteration) {
            const auto forcing = forcing_.request(
                iteration,
                current_residual,
                accepted_streak,
                previous_agreement
            );
            if (iteration > 0U) {
                current_values = subproblem_.values(
                    reference_states,
                    reference_controls,
                    initial_state,
                    target_position,
                    target_velocity,
                    trust_.radius()
                );
                workspace.update_values(std::move(current_values), false);
                workspace.warm_start(warm_primal, warm_dual);
            }

            HostPDHGOptions options;
            options.tolerance = forcing.tolerance;
            options.iteration_limit = forcing.iteration_limit;
            options.check_interval = outer_config_.residual_check_interval;
            options.norm_iterations = outer_config_.norm_iterations;
            const auto begin = std::chrono::steady_clock::now();
            const auto solution = workspace.solve(options);
            const double solve_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin
            ).count();

            if (solution.primal.size() != subproblem_.layout().variables()) {
                result.status = "invalid_solver_result";
                break;
            }
            const auto decoded = subproblem_.decode(solution.primal);
            std::vector<PoweredDescentState> rollout;
            bool rollout_valid = true;
            try {
                rollout = subproblem_.model().rollout(
                    initial_state,
                    decoded.controls,
                    subproblem_.config().step_seconds,
                    false
                );
            } catch (const std::invalid_argument&) {
                rollout_valid = false;
            }
            const auto convex = subproblem_.diagnostics(solution.primal, workspace.values());
            const double step_fraction = step_usage(
                decoded.states,
                decoded.controls,
                reference_states,
                reference_controls,
                trust_.radius()
            );
            const double model_value = model_merit(
                decoded.states,
                decoded.controls,
                decoded.virtual_controls,
                target_position,
                target_velocity
            );
            const double actual_value = rollout_valid
                ? actual_merit(rollout, decoded.controls, target_position, target_velocity)
                : std::numeric_limits<double>::infinity();
            const double predicted_reduction = current_merit - model_value;
            const double actual_reduction = current_merit - actual_value;
            const double agreement = predicted_reduction >
                    outer_config_.minimum_predicted_reduction
                ? actual_reduction / predicted_reduction
                : -std::numeric_limits<double>::infinity();
            const OuterResidual candidate_residual = rollout_valid
                ? outer_residual(
                      decoded.states,
                      decoded.controls,
                      rollout,
                      target_position,
                      target_velocity,
                      step_fraction
                  )
                : OuterResidual{
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      step_fraction,
                  };
            const bool converged_candidate = rollout_valid &&
                candidate_residual.maximum() <= outer_config_.convergence_tolerance &&
                step_fraction <= outer_config_.step_tolerance;
            const bool restoration = rollout_valid &&
                candidate_residual.maximum() <
                    outer_config_.restoration_reduction * current_residual.maximum();
            const bool accepted = solution.solved() && rollout_valid &&
                convex.convex_violation() <= std::max(1.0e-6, 10.0 * forcing.tolerance) &&
                (converged_candidate || restoration ||
                 (actual_reduction > outer_config_.minimum_actual_reduction &&
                  agreement >= outer_config_.acceptance_threshold));
            const auto trust_update = trust_.update(accepted, agreement, step_fraction);

            result.iterations.push_back(PoweredDescentHostSCvxIteration{
                iteration,
                forcing.phase,
                forcing.tolerance,
                solution.iterations,
                solution.status,
                solution.primal_residual,
                solution.dual_residual,
                trust_update.radius_before,
                trust_update.radius_after,
                trust_update.action,
                step_fraction,
                predicted_reduction,
                actual_reduction,
                agreement,
                accepted,
                accepted && restoration,
                candidate_residual,
                convex,
                solve_seconds,
            });
            result.total_solve_seconds += solve_seconds;

            if (!solution.solved()) {
                result.status = "solver_failed";
                break;
            }
            if (accepted && rollout_valid) {
                reference_states = rollout;
                reference_controls = decoded.controls;
                current_rollout = rollout;
                current_merit = actual_value;
                current_residual = candidate_residual;
                warm_primal = solution.primal;
                warm_dual = solution.dual;
                ++accepted_streak;
                ++result.accepted_iterations;
                previous_agreement = agreement;
                result.states = rollout;
                result.controls = decoded.controls;
                result.residual = candidate_residual;
                if (iteration + 1U >= outer_config_.minimum_iterations &&
                    converged_candidate) {
                    result.status = "converged";
                    break;
                }
            } else {
                accepted_streak = 0U;
                previous_agreement = std::numeric_limits<double>::quiet_NaN();
                if (trust_.exhausted()) {
                    result.status = "trust_region_exhausted";
                    break;
                }
            }
        }

        result.path = subproblem_.model().path_diagnostics(result.states, result.controls);
        result.workspace_updates = workspace.update_count();
        result.warm_starts = workspace.warm_start_count();
        result.inner_solves = workspace.solve_count();
        return result;
    }

  private:
    [[nodiscard]] static TrustRegionConfig normalise_trust_config(
        TrustRegionConfig config,
        const double initial_radius
    ) {
        config.initial_radius = initial_radius;
        config.maximum_radius = std::max(config.maximum_radius, initial_radius);
        config.minimum_radius = std::min(config.minimum_radius, initial_radius);
        return config;
    }

    [[nodiscard]] OuterResidual outer_residual(
        std::span<const PoweredDescentState> decision_states,
        std::span<const PoweredDescentControl> controls,
        std::span<const PoweredDescentState> rollout,
        const std::array<double, 3>& target_position,
        const std::array<double, 3>& target_velocity,
        const double step_fraction
    ) const {
        double dynamics = 0.0;
        for (std::size_t node = 0; node < decision_states.size(); ++node) {
            for (std::size_t component = 0; component < powered_descent_state_dimension;
                 ++component) {
                dynamics = std::max(
                    dynamics,
                    std::abs(decision_states[node][component] - rollout[node][component]) *
                        subproblem_.config().state_trust_scales[component]
                );
            }
        }
        double terminal = 0.0;
        for (std::size_t component = 0; component < 3U; ++component) {
            terminal = std::max(
                terminal,
                std::abs(rollout.back()[component] - target_position[component]) *
                    subproblem_.config().state_trust_scales[component]
            );
            terminal = std::max(
                terminal,
                std::abs(rollout.back()[3U + component] - target_velocity[component]) *
                    subproblem_.config().state_trust_scales[3U + component]
            );
        }
        return OuterResidual{
            dynamics,
            maximum_path_component(rollout, controls),
            terminal,
            step_fraction,
        };
    }

    [[nodiscard]] double actual_merit(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls,
        const std::array<double, 3>& target_position,
        const std::array<double, 3>& target_velocity
    ) const {
        double terminal = 0.0;
        for (std::size_t component = 0; component < 3U; ++component) {
            terminal += std::abs(states.back()[component] - target_position[component]) *
                subproblem_.config().state_trust_scales[component];
            terminal += std::abs(states.back()[3U + component] - target_velocity[component]) *
                subproblem_.config().state_trust_scales[3U + component];
        }
        return normalised_fuel(controls) + outer_config_.feasibility_penalty *
            (terminal + sum_path_components(states, controls));
    }

    [[nodiscard]] double model_merit(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls,
        std::span<const PoweredDescentState> virtual_controls,
        const std::array<double, 3>& target_position,
        const std::array<double, 3>& target_velocity
    ) const {
        double virtual_measure = 0.0;
        for (const auto& virtual_control : virtual_controls) {
            for (std::size_t component = 0; component < powered_descent_state_dimension;
                 ++component) {
                virtual_measure += std::abs(virtual_control[component]) *
                    subproblem_.config().state_trust_scales[component];
            }
        }
        virtual_measure /= static_cast<double>(
            std::max<std::size_t>(1U, virtual_controls.size())
        );
        return actual_merit(states, controls, target_position, target_velocity) +
            outer_config_.virtual_penalty * virtual_measure;
    }

    [[nodiscard]] double step_usage(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls,
        std::span<const PoweredDescentState> reference_states,
        std::span<const PoweredDescentControl> reference_controls,
        const double radius
    ) const {
        double maximum = 0.0;
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            double squared = 0.0;
            for (std::size_t component = 0; component < powered_descent_state_dimension;
                 ++component) {
                const double delta = (states[interval][component] -
                    reference_states[interval][component]) *
                    subproblem_.config().state_trust_scales[component];
                squared += delta * delta;
            }
            for (std::size_t component = 0; component < powered_descent_control_dimension;
                 ++component) {
                const double delta = (controls[interval][component] -
                    reference_controls[interval][component]) *
                    subproblem_.config().control_trust_scales[component];
                squared += delta * delta;
            }
            maximum = std::max(maximum, std::sqrt(squared) / radius);
        }
        double terminal_squared = 0.0;
        for (std::size_t component = 0; component < powered_descent_state_dimension;
             ++component) {
            const double delta = (states.back()[component] -
                reference_states.back()[component]) *
                subproblem_.config().state_trust_scales[component];
            terminal_squared += delta * delta;
        }
        return std::max(maximum, std::sqrt(terminal_squared) / radius);
    }

    [[nodiscard]] double normalised_fuel(
        std::span<const PoweredDescentControl> controls
    ) const {
        double sum = 0.0;
        for (const auto& control : controls) {
            sum += control[3];
        }
        return sum /
            (static_cast<double>(std::max<std::size_t>(1U, controls.size())) *
             subproblem_.model().config().maximum_thrust);
    }

    [[nodiscard]] double maximum_path_component(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls
    ) const {
        const auto components = path_components(states, controls);
        return *std::max_element(components.begin(), components.end());
    }

    [[nodiscard]] double sum_path_components(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls
    ) const {
        const auto components = path_components(states, controls);
        double sum = 0.0;
        for (const auto value : components) {
            sum += value;
        }
        return sum;
    }

    [[nodiscard]] std::array<double, 7> path_components(
        std::span<const PoweredDescentState> states,
        std::span<const PoweredDescentControl> controls
    ) const {
        const std::vector<PoweredDescentState> state_vector(states.begin(), states.end());
        const std::vector<PoweredDescentControl> control_vector(controls.begin(), controls.end());
        const auto diagnostics = subproblem_.model().path_diagnostics(
            state_vector,
            control_vector
        );
        const double thrust_scale = subproblem_.model().config().maximum_thrust;
        const double position_scale = *std::max_element(
            subproblem_.config().state_trust_scales.begin(),
            subproblem_.config().state_trust_scales.begin() + 3
        );
        const double mass_scale = subproblem_.config().state_trust_scales[6];
        return {
            diagnostics.thrust_epigraph / thrust_scale,
            diagnostics.throttle_lower / thrust_scale,
            diagnostics.throttle_upper / thrust_scale,
            diagnostics.tilt / thrust_scale,
            diagnostics.minimum_mass * mass_scale,
            diagnostics.altitude * position_scale,
            diagnostics.glide_slope * position_scale,
        };
    }

    PoweredDescentCQP subproblem_;
    PoweredDescentHostSCvxConfig outer_config_;
    AdaptiveForcingRule forcing_;
    TrustRegionController trust_;
};

}  // namespace spacepdhcg::core
