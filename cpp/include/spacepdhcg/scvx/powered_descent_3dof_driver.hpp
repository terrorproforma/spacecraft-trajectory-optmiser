#pragma once

#include "spacepdhcg/core/host_backend.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/scvx/policies.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace spacepdhcg::scvx {

using transcription::PoweredDescent3DofSubproblem;
using transcription::PoweredDescentDecodedDecision;

struct NativePoweredDescentOuterConfig {
    std::size_t maximum_iterations{15U};
    std::size_t minimum_iterations{2U};
    double convergence_tolerance{2.0e-4};
    double step_tolerance{2.0e-2};
    double acceptance_threshold{0.05};
    double feasibility_penalty{100.0};
    double virtual_penalty{100.0};
    double minimum_actual_reduction{1.0e-10};
    double minimum_predicted_reduction{1.0e-12};
    double restoration_reduction{0.9};
    std::size_t maximum_resolves_per_iteration{1U};

    void validate() const {
        if (maximum_iterations == 0U || minimum_iterations > maximum_iterations) {
            throw std::invalid_argument("native SCvx iteration limits are invalid");
        }
        require_positive(convergence_tolerance, "convergence tolerance must be positive");
        require_positive(step_tolerance, "step tolerance must be positive");
        require_positive(feasibility_penalty, "feasibility penalty must be positive");
        require_positive(virtual_penalty, "virtual penalty must be positive");
        require_positive(minimum_actual_reduction, "minimum actual reduction must be positive");
        require_positive(
            minimum_predicted_reduction,
            "minimum predicted reduction must be positive"
        );
        if (!std::isfinite(acceptance_threshold) || acceptance_threshold < 0.0
            || acceptance_threshold >= 1.0) {
            throw std::invalid_argument("acceptance threshold must lie in [0, 1)");
        }
        if (!std::isfinite(restoration_reduction) || restoration_reduction <= 0.0
            || restoration_reduction >= 1.0) {
            throw std::invalid_argument("restoration reduction must lie in (0, 1)");
        }
    }

  private:
    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
};

enum class NativeScvxStatus : std::uint8_t {
    converged,
    maximum_iterations,
    trust_region_exhausted,
    solver_failed,
};

struct NativeScvxIterationRecord {
    std::size_t iteration{0U};
    SolvePhase phase{SolvePhase::repair};
    double requested_tolerance{0.0};
    double effective_tolerance{0.0};
    std::size_t solver_iterations{0U};
    double primal_residual{0.0};
    double dual_residual{0.0};
    double setup_seconds{0.0};
    double update_seconds{0.0};
    double solve_seconds{0.0};
    double trust_radius_before{0.0};
    double trust_radius_after{0.0};
    TrustAction trust_action{TrustAction::retain};
    double step_fraction{0.0};
    double predicted_reduction{0.0};
    double actual_reduction{0.0};
    double agreement{0.0};
    bool accepted{false};
    bool restoration_accepted{false};
    bool re_solved{false};
    OuterResidual residual{};
    transcription::PoweredDescentConvexDiagnostics convex_diagnostics{};
};

struct NativePoweredDescentScvxResult {
    NativeScvxStatus status{NativeScvxStatus::maximum_iterations};
    std::vector<dynamics::PoweredDescentState> states{};
    std::vector<dynamics::PoweredDescentControl> controls{};
    double merit{0.0};
    OuterResidual residual{};
    dynamics::PoweredDescentPathDiagnostics path_diagnostics{};
    std::vector<NativeScvxIterationRecord> iterations{};
    std::size_t accepted_iterations{0U};
    std::size_t backend_creations{0U};
    std::size_t backend_updates{0U};

    [[nodiscard]] bool converged() const noexcept {
        return status == NativeScvxStatus::converged;
    }
};

using HostBackendFactory = std::function<core::HostBackendPointer(core::FixedCQP)>;

inline std::pair<
    std::vector<dynamics::PoweredDescentState>,
    std::vector<dynamics::PoweredDescentControl>>
make_native_powered_descent_reference(
    const dynamics::PoweredDescent3DofModel& model,
    const dynamics::PoweredDescentState& initial,
    const std::array<double, 3U>& target_position,
    const std::array<double, 3U>& target_velocity,
    std::size_t intervals,
    double step_seconds
) {
    if (intervals < 2U || !std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("native powered-descent reference grid is invalid");
    }
    if (initial[6U] <= model.config().minimum_mass) {
        throw std::invalid_argument("reference initial mass must exceed the reserve");
    }
    const auto first_moment = static_cast<double>(intervals * (intervals - 1U)) / 2.0;
    const auto second_moment = static_cast<double>(
        (intervals - 1U) * intervals * (2U * intervals - 1U)
    ) / 6.0;
    const auto a00 = static_cast<double>(intervals);
    const auto a01 = static_cast<double>(intervals * intervals);
    const auto a10 = step_seconds * first_moment;
    const auto a11 = step_seconds * second_moment;
    const auto determinant = a00 * a11 - a01 * a10;
    if (std::abs(determinant) <= 1.0e-14) {
        throw std::runtime_error("native powered-descent reference system is singular");
    }

    std::vector<std::array<double, 3U>> velocities(intervals + 1U);
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const auto rhs0 = target_velocity[axis] - initial[3U + axis];
        const auto rhs1 = target_position[axis] - initial[axis]
                          - step_seconds * static_cast<double>(intervals)
                                * initial[3U + axis];
        const auto linear = (rhs0 * a11 - a01 * rhs1) / determinant;
        const auto quadratic = (a00 * rhs1 - rhs0 * a10) / determinant;
        for (std::size_t node = 0; node <= intervals; ++node) {
            const auto index = static_cast<double>(node);
            velocities[node][axis] = initial[3U + axis] + linear * index
                                     + quadratic * index * index;
        }
    }

    std::vector<dynamics::PoweredDescentState> states(intervals + 1U);
    std::vector<dynamics::PoweredDescentControl> controls(intervals);
    states.front() = initial;
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        dynamics::ThrustVector requested{};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            const auto acceleration =
                (velocities[interval + 1U][axis] - velocities[interval][axis])
                / step_seconds;
            requested[axis] = states[interval][6U]
                              * (acceleration - model.config().gravity[axis]);
        }
        const auto thrust = model.project_thrust(requested);
        const auto sigma = std::sqrt(
            thrust[0U] * thrust[0U] + thrust[1U] * thrust[1U]
            + thrust[2U] * thrust[2U]
        );
        controls[interval] = dynamics::PoweredDescentControl{
            thrust[0U],
            thrust[1U],
            thrust[2U],
            sigma,
        };
        states[interval + 1U] = model.euler_step(
            states[interval],
            controls[interval],
            step_seconds
        );
        if (states[interval + 1U][6U] <= model.config().minimum_mass) {
            throw std::runtime_error("native reference consumes the propellant reserve");
        }
    }
    return {states, controls};
}

class NativePoweredDescentScvxDriver {
  public:
    NativePoweredDescentScvxDriver(
        PoweredDescent3DofSubproblem subproblem,
        HostBackendFactory backend_factory,
        NativePoweredDescentOuterConfig outer_config = {},
        ForcingRuleConfig forcing_config = {},
        TrustRegionConfig trust_config = {}
    )
        : subproblem_(std::move(subproblem)),
          backend_factory_(std::move(backend_factory)),
          outer_config_(outer_config),
          forcing_(forcing_config),
          trust_(normalise_trust_config(trust_config, subproblem_.config().trust_radius)) {
        if (!backend_factory_) {
            throw std::invalid_argument("native SCvx requires a backend factory");
        }
        outer_config_.validate();
    }

    [[nodiscard]] NativePoweredDescentScvxResult solve(
        const dynamics::PoweredDescentState& initial,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity,
        std::optional<std::pair<
            std::vector<dynamics::PoweredDescentState>,
            std::vector<dynamics::PoweredDescentControl>>> reference = std::nullopt
    ) {
        auto [current_states, current_controls] = reference.has_value()
                                                      ? std::move(*reference)
                                                      : make_native_powered_descent_reference(
                                                            subproblem_.model(),
                                                            initial,
                                                            target_position,
                                                            target_velocity,
                                                            subproblem_.layout().intervals,
                                                            subproblem_.config().step_seconds
                                                        );
        validate_reference(current_states, current_controls);
        auto current_merit = actual_merit(
            current_states,
            current_controls,
            target_position,
            target_velocity
        );
        auto current_residual = outer_residual(
            current_states,
            current_controls,
            current_states,
            target_position,
            target_velocity,
            1.0
        );

        core::HostBackendPointer backend{};
        core::HostWarmStart warm_start{};
        bool have_warm_start{false};
        std::vector<NativeScvxIterationRecord> records{};
        std::size_t accepted_streak{0U};
        std::size_t accepted_iterations{0U};
        std::size_t backend_creations{0U};
        std::optional<double> previous_agreement{};
        auto status = NativeScvxStatus::maximum_iterations;

        for (std::size_t iteration = 0; iteration < outer_config_.maximum_iterations;
             ++iteration) {
            const auto request = forcing_.request(
                iteration,
                current_residual,
                accepted_streak,
                previous_agreement.value_or(
                    std::numeric_limits<double>::quiet_NaN()
                )
            );
            auto values = subproblem_.values(
                current_states,
                current_controls,
                initial,
                target_position,
                target_velocity,
                trust_.radius()
            );
            if (!backend) {
                backend = backend_factory_(core::FixedCQP(subproblem_.structure(), values));
                ++backend_creations;
                if (!backend || backend->structure().fingerprint()
                                    != subproblem_.structure().fingerprint()) {
                    throw std::runtime_error("native backend factory returned incompatible topology");
                }
            } else {
                backend->update(values);
            }
            if (have_warm_start) {
                backend->warm_start(warm_start);
            }

            auto effective_tolerance = request.tolerance;
            auto solution = backend->solve(effective_tolerance, request.iteration_limit);
            auto candidate = evaluate_candidate(
                solution,
                values,
                current_states,
                current_controls,
                initial,
                target_position,
                target_velocity,
                current_merit,
                current_residual
            );
            bool re_solved{false};
            for (std::size_t resolve = 0;
                 resolve < outer_config_.maximum_resolves_per_iteration;
                 ++resolve) {
                if (!forcing_.should_resolve(
                        candidate.accepted,
                        solution.primal_residual,
                        solution.dual_residual,
                        effective_tolerance
                    )) {
                    break;
                }
                effective_tolerance = forcing_.refined_tolerance(effective_tolerance);
                solution = backend->solve(
                    effective_tolerance,
                    std::max(
                        request.iteration_limit,
                        forcing_.config().refinement_iteration_limit
                    )
                );
                candidate = evaluate_candidate(
                    solution,
                    values,
                    current_states,
                    current_controls,
                    initial,
                    target_position,
                    target_velocity,
                    current_merit,
                    current_residual
                );
                re_solved = true;
            }

            const auto trust_update = trust_.update(
                candidate.accepted,
                candidate.agreement,
                candidate.step_fraction
            );
            records.push_back(
                NativeScvxIterationRecord{
                    iteration,
                    request.phase,
                    request.tolerance,
                    effective_tolerance,
                    solution.outer_iterations + solution.inner_iterations,
                    solution.primal_residual,
                    solution.dual_residual,
                    solution.setup_seconds,
                    solution.update_seconds,
                    solution.solve_seconds,
                    trust_update.radius_before,
                    trust_update.radius_after,
                    trust_update.action,
                    candidate.step_fraction,
                    candidate.predicted_reduction,
                    candidate.actual_reduction,
                    candidate.agreement,
                    candidate.accepted,
                    candidate.restoration,
                    re_solved,
                    candidate.residual,
                    candidate.convex_diagnostics,
                }
            );

            if (!solution.solved()) {
                status = NativeScvxStatus::solver_failed;
                break;
            }
            if (candidate.accepted) {
                current_states = std::move(candidate.rollout);
                current_controls = std::move(candidate.decoded.controls);
                current_merit = candidate.actual_merit;
                current_residual = candidate.residual;
                warm_start = core::HostWarmStart{solution.primal, solution.dual};
                have_warm_start = true;
                ++accepted_streak;
                ++accepted_iterations;
                previous_agreement = candidate.agreement;
                if (iteration + 1U >= outer_config_.minimum_iterations
                    && current_residual.maximum() <= outer_config_.convergence_tolerance
                    && candidate.step_fraction <= outer_config_.step_tolerance) {
                    status = NativeScvxStatus::converged;
                    break;
                }
            } else {
                accepted_streak = 0U;
                previous_agreement.reset();
                if (trust_.exhausted()) {
                    status = NativeScvxStatus::trust_region_exhausted;
                    break;
                }
            }
        }

        return NativePoweredDescentScvxResult{
            status,
            current_states,
            current_controls,
            current_merit,
            current_residual,
            subproblem_.model().path_diagnostics(current_states, current_controls),
            records,
            accepted_iterations,
            backend_creations,
            backend ? backend->update_count() : 0U,
        };
    }

  private:
    struct Candidate {
        PoweredDescentDecodedDecision decoded{};
        std::vector<dynamics::PoweredDescentState> rollout{};
        transcription::PoweredDescentConvexDiagnostics convex_diagnostics{};
        OuterResidual residual{};
        double model_merit{0.0};
        double actual_merit{0.0};
        double step_fraction{0.0};
        double predicted_reduction{0.0};
        double actual_reduction{0.0};
        double agreement{-std::numeric_limits<double>::infinity()};
        bool accepted{false};
        bool restoration{false};
    };

    PoweredDescent3DofSubproblem subproblem_;
    HostBackendFactory backend_factory_;
    NativePoweredDescentOuterConfig outer_config_{};
    AdaptiveForcingRule forcing_{};
    TrustRegionController trust_{};

    static TrustRegionConfig normalise_trust_config(
        TrustRegionConfig config,
        double subproblem_radius
    ) {
        if (config.initial_radius == TrustRegionConfig{}.initial_radius
            && subproblem_radius != config.initial_radius) {
            config.initial_radius = subproblem_radius;
            config.maximum_radius = std::max(config.maximum_radius, subproblem_radius);
        }
        return config;
    }

    void validate_reference(
        const std::vector<dynamics::PoweredDescentState>& states,
        const std::vector<dynamics::PoweredDescentControl>& controls
    ) const {
        if (states.size() != subproblem_.layout().intervals + 1U
            || controls.size() != subproblem_.layout().intervals) {
            throw std::invalid_argument("native SCvx reference trajectory has the wrong horizon");
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            static_cast<void>(subproblem_.model().dynamics(states[interval], controls[interval]));
        }
    }

    [[nodiscard]] Candidate evaluate_candidate(
        const core::HostCqpSolution& solution,
        const core::NumericValues& values,
        const std::vector<dynamics::PoweredDescentState>& current_states,
        const std::vector<dynamics::PoweredDescentControl>& current_controls,
        const dynamics::PoweredDescentState& initial,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity,
        double current_merit,
        const OuterResidual& current_residual
    ) const {
        if (solution.primal.size() != subproblem_.layout().variables()) {
            throw std::runtime_error("native backend returned a primal with the wrong size");
        }
        if (!solution.dual.empty()
            && solution.dual.size() != static_cast<std::size_t>(subproblem_.structure().duals())) {
            throw std::runtime_error("native backend returned a dual with the wrong size");
        }
        Candidate candidate{};
        candidate.decoded = subproblem_.decode(solution.primal);
        candidate.convex_diagnostics = subproblem_.diagnostics(solution.primal, values);
        candidate.step_fraction = step_fraction(
            candidate.decoded.states,
            candidate.decoded.controls,
            current_states,
            current_controls,
            trust_.radius()
        );
        try {
            candidate.rollout = subproblem_.model().rollout(
                initial,
                candidate.decoded.controls,
                subproblem_.config().step_seconds,
                subproblem_.config().discretisation
                    == transcription::DiscretisationMethod::rk4_finite_difference
            );
        } catch (const std::exception&) {
            candidate.rollout.clear();
        }
        candidate.model_merit = model_merit(
            candidate.decoded,
            target_position,
            target_velocity
        );
        candidate.actual_merit = candidate.rollout.empty()
                                     ? std::numeric_limits<double>::infinity()
                                     : actual_merit(
                                           candidate.rollout,
                                           candidate.decoded.controls,
                                           target_position,
                                           target_velocity
                                       );
        candidate.predicted_reduction = current_merit - candidate.model_merit;
        candidate.actual_reduction = current_merit - candidate.actual_merit;
        if (candidate.predicted_reduction > outer_config_.minimum_predicted_reduction) {
            candidate.agreement =
                candidate.actual_reduction / candidate.predicted_reduction;
        }
        candidate.residual = candidate.rollout.empty()
                                 ? OuterResidual{
                                       std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity(),
                                       candidate.step_fraction,
                                   }
                                 : outer_residual(
                                       candidate.decoded.states,
                                       candidate.decoded.controls,
                                       candidate.rollout,
                                       target_position,
                                       target_velocity,
                                       candidate.step_fraction
                                   );
        candidate.restoration = !candidate.rollout.empty()
                                && candidate.residual.maximum()
                                       < outer_config_.restoration_reduction
                                             * current_residual.maximum();
        const auto convex_tolerance = std::max(
            1.0e-7,
            10.0 * std::max(solution.primal_residual, solution.dual_residual)
        );
        candidate.accepted = solution.solved() && !candidate.rollout.empty()
                             && candidate.convex_diagnostics.maximum_violation()
                                    <= convex_tolerance
                             && std::isfinite(candidate.actual_merit)
                             && ((candidate.actual_reduction
                                      > outer_config_.minimum_actual_reduction
                                  && candidate.agreement
                                         >= outer_config_.acceptance_threshold)
                                 || candidate.restoration);
        return candidate;
    }

    [[nodiscard]] OuterResidual outer_residual(
        const std::vector<dynamics::PoweredDescentState>& decision_states,
        const std::vector<dynamics::PoweredDescentControl>& controls,
        const std::vector<dynamics::PoweredDescentState>& rollout,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity,
        double step_fraction_value
    ) const {
        const auto& scales = subproblem_.config().state_trust_scales;
        double dynamics_residual{0.0};
        for (std::size_t node = 0; node < decision_states.size(); ++node) {
            for (std::size_t component = 0; component < 7U; ++component) {
                dynamics_residual = std::max(
                    dynamics_residual,
                    std::abs(
                        (decision_states[node][component] - rollout[node][component])
                        * scales[component]
                    )
                );
            }
        }
        double terminal{0.0};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            terminal = std::max(
                terminal,
                std::abs((rollout.back()[axis] - target_position[axis]) * scales[axis])
            );
            terminal = std::max(
                terminal,
                std::abs(
                    (rollout.back()[3U + axis] - target_velocity[axis])
                    * scales[3U + axis]
                )
            );
        }
        const auto path = maximum_path_component(rollout, controls);
        return OuterResidual{dynamics_residual, path, terminal, step_fraction_value};
    }

    [[nodiscard]] double actual_merit(
        const std::vector<dynamics::PoweredDescentState>& states,
        const std::vector<dynamics::PoweredDescentControl>& controls,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity
    ) const {
        const auto& scales = subproblem_.config().state_trust_scales;
        double terminal{0.0};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            terminal += std::abs((states.back()[axis] - target_position[axis]) * scales[axis]);
            terminal += std::abs(
                (states.back()[3U + axis] - target_velocity[axis]) * scales[3U + axis]
            );
        }
        return normalised_fuel(controls)
               + outer_config_.feasibility_penalty
                     * (terminal + sum_path_components(states, controls));
    }

    [[nodiscard]] double model_merit(
        const PoweredDescentDecodedDecision& decision,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity
    ) const {
        const auto& scales = subproblem_.config().state_trust_scales;
        double terminal{0.0};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            terminal += std::abs(
                (decision.states.back()[axis] - target_position[axis]) * scales[axis]
            );
            terminal += std::abs(
                (decision.states.back()[3U + axis] - target_velocity[axis])
                * scales[3U + axis]
            );
        }
        double virtual_measure{0.0};
        std::size_t virtual_components{0U};
        for (const auto& virtual_control : decision.virtual_controls) {
            for (std::size_t component = 0; component < 7U; ++component) {
                virtual_measure += std::abs(virtual_control[component] * scales[component]);
                ++virtual_components;
            }
        }
        if (virtual_components > 0U) {
            virtual_measure /= static_cast<double>(virtual_components);
        }
        return normalised_fuel(decision.controls)
               + outer_config_.feasibility_penalty
                     * (terminal + sum_path_components(decision.states, decision.controls))
               + outer_config_.virtual_penalty * virtual_measure;
    }

    [[nodiscard]] double maximum_path_component(
        const std::vector<dynamics::PoweredDescentState>& states,
        const std::vector<dynamics::PoweredDescentControl>& controls
    ) const {
        const auto components = path_components(states, controls);
        return *std::max_element(components.begin(), components.end());
    }

    [[nodiscard]] double sum_path_components(
        const std::vector<dynamics::PoweredDescentState>& states,
        const std::vector<dynamics::PoweredDescentControl>& controls
    ) const {
        const auto components = path_components(states, controls);
        double total{0.0};
        for (const auto component : components) {
            total += component;
        }
        return total;
    }

    [[nodiscard]] std::array<double, 7U> path_components(
        const std::vector<dynamics::PoweredDescentState>& states,
        const std::vector<dynamics::PoweredDescentControl>& controls
    ) const {
        const auto diagnostics = subproblem_.model().path_diagnostics(states, controls);
        const auto maximum_thrust = subproblem_.model().config().maximum_thrust;
        const auto position_scale = std::max(
            {subproblem_.config().state_trust_scales[0U],
             subproblem_.config().state_trust_scales[1U],
             subproblem_.config().state_trust_scales[2U]}
        );
        const auto mass_scale = subproblem_.config().state_trust_scales[6U];
        return {
            diagnostics.thrust_epigraph / maximum_thrust,
            diagnostics.throttle_lower / maximum_thrust,
            diagnostics.throttle_upper / maximum_thrust,
            diagnostics.tilt / maximum_thrust,
            diagnostics.minimum_mass * mass_scale,
            diagnostics.altitude * position_scale,
            diagnostics.glide_slope * position_scale,
        };
    }

    [[nodiscard]] double normalised_fuel(
        const std::vector<dynamics::PoweredDescentControl>& controls
    ) const {
        double total{0.0};
        for (const auto& control : controls) {
            total += control[3U];
        }
        return controls.empty()
                   ? 0.0
                   : total
                         / (static_cast<double>(controls.size())
                            * subproblem_.model().config().maximum_thrust);
    }

    [[nodiscard]] double step_fraction(
        const std::vector<dynamics::PoweredDescentState>& states,
        const std::vector<dynamics::PoweredDescentControl>& controls,
        const std::vector<dynamics::PoweredDescentState>& reference_states,
        const std::vector<dynamics::PoweredDescentControl>& reference_controls,
        double radius
    ) const {
        double maximum{0.0};
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            double norm_squared{0.0};
            for (std::size_t component = 0; component < 7U; ++component) {
                const auto value =
                    (states[interval][component] - reference_states[interval][component])
                    * subproblem_.config().state_trust_scales[component];
                norm_squared += value * value;
            }
            for (std::size_t component = 0; component < 4U; ++component) {
                const auto value =
                    (controls[interval][component] - reference_controls[interval][component])
                    * subproblem_.config().control_trust_scales[component];
                norm_squared += value * value;
            }
            maximum = std::max(maximum, std::sqrt(norm_squared) / radius);
        }
        double terminal_norm{0.0};
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto value =
                (states.back()[component] - reference_states.back()[component])
                * subproblem_.config().state_trust_scales[component];
            terminal_norm += value * value;
        }
        return std::max(maximum, std::sqrt(terminal_norm) / radius);
    }
};

inline std::string_view native_scvx_status_name(NativeScvxStatus status) noexcept {
    switch (status) {
        case NativeScvxStatus::converged:
            return "converged";
        case NativeScvxStatus::maximum_iterations:
            return "maximum_iterations";
        case NativeScvxStatus::trust_region_exhausted:
            return "trust_region_exhausted";
        case NativeScvxStatus::solver_failed:
            return "solver_failed";
    }
    return "unknown";
}

}  // namespace spacepdhcg::scvx
