#pragma once

#include "spacepdhcg/core/host_backend.hpp"
#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/scvx/policies.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace spacepdhcg::scvx {

struct NativeLowThrustOuterConfig {
    std::size_t maximum_iterations{20U};
    std::size_t minimum_iterations{2U};
    double convergence_tolerance{5.0e-4};
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
            throw std::invalid_argument("low-thrust SCvx iteration limits are invalid");
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
            throw std::invalid_argument("acceptance threshold must lie in [0,1)");
        }
        if (!std::isfinite(restoration_reduction) || restoration_reduction <= 0.0
            || restoration_reduction >= 1.0) {
            throw std::invalid_argument("restoration reduction must lie in (0,1)");
        }
    }

  private:
    static void require_positive(const double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
};

enum class NativeLowThrustStatus : std::uint8_t {
    converged,
    maximum_iterations,
    trust_region_exhausted,
    solver_failed,
};

struct NativeLowThrustIterationRecord {
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
    transcription::LowThrustConvexDiagnostics convex_diagnostics{};
};

struct NativeLowThrustScvxResult {
    NativeLowThrustStatus status{NativeLowThrustStatus::maximum_iterations};
    std::vector<dynamics::LowThrustState> states{};
    std::vector<dynamics::LowThrustControl> controls{};
    double merit{0.0};
    OuterResidual residual{};
    dynamics::LowThrustPathDiagnostics path_diagnostics{};
    std::vector<NativeLowThrustIterationRecord> iterations{};
    std::size_t accepted_iterations{0U};
    std::size_t backend_creations{0U};
    std::size_t backend_updates{0U};

    [[nodiscard]] bool converged() const noexcept {
        return status == NativeLowThrustStatus::converged;
    }
};

struct LowThrustDecodedDecision {
    std::vector<dynamics::LowThrustState> states{};
    std::vector<dynamics::LowThrustControl> controls{};
    std::vector<dynamics::LowThrustState> virtual_controls{};
};

using LowThrustReference = std::pair<
    std::vector<dynamics::LowThrustState>,
    std::vector<dynamics::LowThrustControl>>;
using LowThrustHostBackendFactory =
    std::function<core::HostBackendPointer(core::FixedCQP)>;

[[nodiscard]] inline LowThrustDecodedDecision decode_low_thrust_decision(
    const transcription::LowThrustSubproblem& subproblem,
    const std::vector<double>& decision
) {
    const auto& layout = subproblem.layout();
    if (decision.size() != layout.variables()) {
        throw std::invalid_argument("low-thrust decision vector has the wrong size");
    }
    LowThrustDecodedDecision result;
    result.states.resize(layout.intervals + 1U);
    result.controls.resize(layout.intervals);
    result.virtual_controls.resize(layout.intervals);
    for (std::size_t node = 0; node <= layout.intervals; ++node) {
        const auto range = layout.state(node);
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(range.start),
            range.size,
            result.states[node].begin()
        );
    }
    for (std::size_t interval = 0; interval < layout.intervals; ++interval) {
        const auto control = layout.control(interval);
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(control.start),
            control.size,
            result.controls[interval].begin()
        );
        const auto virtual_control = layout.virtual_control(interval);
        std::copy_n(
            decision.begin() + static_cast<std::ptrdiff_t>(virtual_control.start),
            virtual_control.size,
            result.virtual_controls[interval].begin()
        );
    }
    return result;
}

namespace low_thrust_driver_detail {

inline std::array<double, 3U> hermite_position(
    const dynamics::LowThrustState& initial,
    const dynamics::LowThrustState& target,
    const double duration,
    const double normalised_time
) {
    const auto s = normalised_time;
    const auto s2 = s * s;
    const auto s3 = s2 * s;
    const auto h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    const auto h10 = s3 - 2.0 * s2 + s;
    const auto h01 = -2.0 * s3 + 3.0 * s2;
    const auto h11 = s3 - s2;
    std::array<double, 3U> result{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        result[axis] = h00 * initial[axis] + h10 * duration * initial[3U + axis]
                       + h01 * target[axis] + h11 * duration * target[3U + axis];
    }
    return result;
}

inline std::array<double, 3U> hermite_acceleration(
    const dynamics::LowThrustState& initial,
    const dynamics::LowThrustState& target,
    const double duration,
    const double normalised_time
) {
    const auto s = normalised_time;
    const auto d2h00 = 12.0 * s - 6.0;
    const auto d2h10 = 6.0 * s - 4.0;
    const auto d2h01 = -12.0 * s + 6.0;
    const auto d2h11 = 6.0 * s - 2.0;
    std::array<double, 3U> result{};
    const auto inverse_duration_squared = 1.0 / (duration * duration);
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        result[axis] =
            (d2h00 * initial[axis] + d2h10 * duration * initial[3U + axis]
             + d2h01 * target[axis] + d2h11 * duration * target[3U + axis])
            * inverse_duration_squared;
    }
    return result;
}

inline double norm3(const std::array<double, 3U>& vector) noexcept {
    return std::sqrt(
        vector[0U] * vector[0U] + vector[1U] * vector[1U]
        + vector[2U] * vector[2U]
    );
}

}  // namespace low_thrust_driver_detail

/// Construct a physical, dynamically propagated initial reference.
///
/// A cubic Hermite boundary curve supplies desired midpoint accelerations. The corresponding
/// thrust is clipped to the physical limit, then the actual nonlinear model is propagated.
/// Virtual control in the first convex subproblem absorbs the remaining boundary mismatch.
[[nodiscard]] inline LowThrustReference make_native_low_thrust_reference(
    const dynamics::LowThrustTwoBodyModel& model,
    const dynamics::LowThrustState& initial,
    const dynamics::LowThrustState& target,
    const std::size_t intervals,
    const double step_seconds,
    const bool use_rk4 = true
) {
    if (intervals < 2U || !std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("low-thrust reference grid is invalid");
    }
    const dynamics::LowThrustControl zero_control{0.0, 0.0, 0.0, 0.0};
    static_cast<void>(model.dynamics(initial, zero_control));
    static_cast<void>(model.dynamics(target, zero_control));
    if (initial[6U] <= model.config().minimum_mass) {
        throw std::invalid_argument("low-thrust initial mass must exceed the reserve");
    }

    const auto duration = static_cast<double>(intervals) * step_seconds;
    std::vector<dynamics::LowThrustControl> controls(intervals);
    double reference_mass = initial[6U];
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        const auto midpoint =
            (static_cast<double>(interval) + 0.5) / static_cast<double>(intervals);
        const auto position = low_thrust_driver_detail::hermite_position(
            initial,
            target,
            duration,
            midpoint
        );
        const auto acceleration = low_thrust_driver_detail::hermite_acceleration(
            initial,
            target,
            duration,
            midpoint
        );
        const auto radius = low_thrust_driver_detail::norm3(position);
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::runtime_error("Hermite low-thrust reference crosses the central singularity");
        }
        const auto gravity_scale =
            -model.config().gravitational_parameter / (radius * radius * radius);
        std::array<double, 3U> requested{};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            requested[axis] = reference_mass
                              * (acceleration[axis] - gravity_scale * position[axis])
                              / model.config().thrust_to_acceleration;
        }
        auto magnitude = low_thrust_driver_detail::norm3(requested);
        if (magnitude > model.config().maximum_thrust) {
            const auto scale = model.config().maximum_thrust / magnitude;
            for (auto& component : requested) {
                component *= scale;
            }
            magnitude = model.config().maximum_thrust;
        }
        controls[interval] = dynamics::LowThrustControl{
            requested[0U],
            requested[1U],
            requested[2U],
            magnitude,
        };
        reference_mass -= model.config().mass_flow_coefficient * magnitude * step_seconds;
        if (reference_mass <= model.config().minimum_mass) {
            throw std::runtime_error("low-thrust reference consumes the mass reserve");
        }
    }
    auto states = model.rollout(initial, controls, step_seconds, use_rk4);
    return LowThrustReference{std::move(states), std::move(controls)};
}

class NativeLowThrustScvxDriver {
  public:
    NativeLowThrustScvxDriver(
        transcription::LowThrustSubproblem subproblem,
        LowThrustHostBackendFactory backend_factory,
        NativeLowThrustOuterConfig outer_config = {},
        ForcingRuleConfig forcing_config = {},
        TrustRegionConfig trust_config = {}
    )
        : subproblem_(std::move(subproblem)),
          backend_factory_(std::move(backend_factory)),
          outer_config_(outer_config),
          forcing_(forcing_config),
          trust_(normalise_trust_config(
              trust_config,
              subproblem_.config().trust_radius
          )) {
        if (!backend_factory_) {
            throw std::invalid_argument("low-thrust SCvx requires a backend factory");
        }
        outer_config_.validate();
    }

    [[nodiscard]] const transcription::LowThrustSubproblem& subproblem() const noexcept {
        return subproblem_;
    }

    [[nodiscard]] NativeLowThrustScvxResult solve(
        const dynamics::LowThrustState& initial,
        const dynamics::LowThrustState& target,
        std::optional<LowThrustReference> reference = std::nullopt
    ) {
        trust_.reset();
        auto [current_states, current_controls] = reference.has_value()
                                                      ? std::move(*reference)
                                                      : make_native_low_thrust_reference(
                                                            subproblem_.model(),
                                                            initial,
                                                            target,
                                                            subproblem_.layout().intervals,
                                                            subproblem_.config().step_seconds,
                                                            use_rk4()
                                                        );
        validate_reference(current_states, current_controls);
        auto current_merit = actual_merit(current_states, current_controls, target);
        auto current_residual = outer_residual(
            current_states,
            current_controls,
            current_states,
            target,
            1.0
        );

        core::HostBackendPointer backend{};
        core::HostWarmStart warm_start{};
        bool have_warm_start{false};
        std::vector<NativeLowThrustIterationRecord> records{};
        std::size_t accepted_streak{0U};
        std::size_t accepted_iterations{0U};
        std::size_t backend_creations{0U};
        std::optional<double> previous_agreement{};
        auto status = NativeLowThrustStatus::maximum_iterations;

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
                target,
                trust_.radius()
            );
            if (!backend) {
                backend = backend_factory_(core::FixedCQP(subproblem_.structure(), values));
                ++backend_creations;
                if (!backend || backend->structure().fingerprint()
                                    != subproblem_.structure().fingerprint()) {
                    throw std::runtime_error(
                        "low-thrust backend factory returned incompatible topology"
                    );
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
                target,
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
                    target,
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
            records.push_back(NativeLowThrustIterationRecord{
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
            });

            if (!solution.solved()) {
                status = NativeLowThrustStatus::solver_failed;
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
                    status = NativeLowThrustStatus::converged;
                    break;
                }
            } else {
                accepted_streak = 0U;
                previous_agreement.reset();
                if (trust_.exhausted()) {
                    status = NativeLowThrustStatus::trust_region_exhausted;
                    break;
                }
            }
        }

        return NativeLowThrustScvxResult{
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
        LowThrustDecodedDecision decoded{};
        std::vector<dynamics::LowThrustState> rollout{};
        transcription::LowThrustConvexDiagnostics convex_diagnostics{};
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

    transcription::LowThrustSubproblem subproblem_;
    LowThrustHostBackendFactory backend_factory_;
    NativeLowThrustOuterConfig outer_config_{};
    AdaptiveForcingRule forcing_{};
    TrustRegionController trust_{};

    static TrustRegionConfig normalise_trust_config(
        TrustRegionConfig config,
        const double subproblem_radius
    ) {
        if (config.initial_radius == TrustRegionConfig{}.initial_radius
            && subproblem_radius != config.initial_radius) {
            config.initial_radius = subproblem_radius;
            config.maximum_radius = std::max(config.maximum_radius, subproblem_radius);
        }
        return config;
    }

    [[nodiscard]] bool use_rk4() const noexcept {
        return subproblem_.config().discretisation
               == transcription::DiscretisationMethod::rk4_finite_difference;
    }

    void validate_reference(
        const std::vector<dynamics::LowThrustState>& states,
        const std::vector<dynamics::LowThrustControl>& controls
    ) const {
        if (states.size() != subproblem_.layout().intervals + 1U
            || controls.size() != subproblem_.layout().intervals) {
            throw std::invalid_argument("low-thrust reference trajectory has the wrong horizon");
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            static_cast<void>(subproblem_.model().dynamics(states[interval], controls[interval]));
        }
        const dynamics::LowThrustControl zero_control{0.0, 0.0, 0.0, 0.0};
        static_cast<void>(subproblem_.model().dynamics(states.back(), zero_control));
    }

    [[nodiscard]] Candidate evaluate_candidate(
        const core::HostCqpSolution& solution,
        const core::NumericValues& values,
        const std::vector<dynamics::LowThrustState>& current_states,
        const std::vector<dynamics::LowThrustControl>& current_controls,
        const dynamics::LowThrustState& initial,
        const dynamics::LowThrustState& target,
        const double current_merit,
        const OuterResidual& current_residual
    ) const {
        if (solution.primal.size() != subproblem_.layout().variables()) {
            throw std::runtime_error("low-thrust backend returned a primal with the wrong size");
        }
        if (!solution.dual.empty()
            && solution.dual.size() != static_cast<std::size_t>(subproblem_.structure().duals())) {
            throw std::runtime_error("low-thrust backend returned a dual with the wrong size");
        }
        Candidate candidate{};
        candidate.decoded = decode_low_thrust_decision(subproblem_, solution.primal);
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
                use_rk4()
            );
        } catch (const std::exception&) {
            candidate.rollout.clear();
        }
        candidate.model_merit = model_merit(candidate.decoded, target);
        candidate.actual_merit = candidate.rollout.empty()
                                     ? std::numeric_limits<double>::infinity()
                                     : actual_merit(
                                           candidate.rollout,
                                           candidate.decoded.controls,
                                           target
                                       );
        candidate.predicted_reduction = current_merit - candidate.model_merit;
        candidate.actual_reduction = current_merit - candidate.actual_merit;
        if (candidate.predicted_reduction > outer_config_.minimum_predicted_reduction) {
            candidate.agreement = candidate.actual_reduction / candidate.predicted_reduction;
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
                                       target,
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
        const std::vector<dynamics::LowThrustState>& decision_states,
        const std::vector<dynamics::LowThrustControl>& controls,
        const std::vector<dynamics::LowThrustState>& rollout,
        const dynamics::LowThrustState& target,
        const double step_fraction_value
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
        for (std::size_t component = 0; component < 6U; ++component) {
            terminal = std::max(
                terminal,
                std::abs((rollout.back()[component] - target[component]) * scales[component])
            );
        }
        return OuterResidual{
            dynamics_residual,
            maximum_path_component(rollout, controls),
            terminal,
            step_fraction_value,
        };
    }

    [[nodiscard]] double actual_merit(
        const std::vector<dynamics::LowThrustState>& states,
        const std::vector<dynamics::LowThrustControl>& controls,
        const dynamics::LowThrustState& target
    ) const {
        const auto& scales = subproblem_.config().state_trust_scales;
        double terminal{0.0};
        for (std::size_t component = 0; component < 6U; ++component) {
            terminal += std::abs(
                (states.back()[component] - target[component]) * scales[component]
            );
        }
        return normalised_fuel(controls)
               + outer_config_.feasibility_penalty
                     * (terminal + sum_path_components(states, controls));
    }

    [[nodiscard]] double model_merit(
        const LowThrustDecodedDecision& decision,
        const dynamics::LowThrustState& target
    ) const {
        const auto& scales = subproblem_.config().state_trust_scales;
        double terminal{0.0};
        for (std::size_t component = 0; component < 6U; ++component) {
            terminal += std::abs(
                (decision.states.back()[component] - target[component]) * scales[component]
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

    [[nodiscard]] std::array<double, 4U> path_components(
        const std::vector<dynamics::LowThrustState>& states,
        const std::vector<dynamics::LowThrustControl>& controls
    ) const {
        const auto diagnostics = subproblem_.model().path_diagnostics(states, controls);
        const auto maximum_thrust = subproblem_.model().config().maximum_thrust;
        const auto position_scale = std::max(
            {subproblem_.config().state_trust_scales[0U],
             subproblem_.config().state_trust_scales[1U],
             subproblem_.config().state_trust_scales[2U]}
        );
        return {
            diagnostics.thrust_epigraph / maximum_thrust,
            diagnostics.throttle_upper / maximum_thrust,
            diagnostics.minimum_mass * subproblem_.config().state_trust_scales[6U],
            diagnostics.minimum_radius * position_scale,
        };
    }

    [[nodiscard]] double maximum_path_component(
        const std::vector<dynamics::LowThrustState>& states,
        const std::vector<dynamics::LowThrustControl>& controls
    ) const {
        const auto components = path_components(states, controls);
        return *std::max_element(components.begin(), components.end());
    }

    [[nodiscard]] double sum_path_components(
        const std::vector<dynamics::LowThrustState>& states,
        const std::vector<dynamics::LowThrustControl>& controls
    ) const {
        const auto components = path_components(states, controls);
        double result{0.0};
        for (const auto component : components) {
            result += component;
        }
        return result;
    }

    [[nodiscard]] double normalised_fuel(
        const std::vector<dynamics::LowThrustControl>& controls
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
        const std::vector<dynamics::LowThrustState>& states,
        const std::vector<dynamics::LowThrustControl>& controls,
        const std::vector<dynamics::LowThrustState>& reference_states,
        const std::vector<dynamics::LowThrustControl>& reference_controls,
        const double radius
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
        double terminal_norm_squared{0.0};
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto value =
                (states.back()[component] - reference_states.back()[component])
                * subproblem_.config().state_trust_scales[component];
            terminal_norm_squared += value * value;
        }
        return std::max(maximum, std::sqrt(terminal_norm_squared) / radius);
    }
};

[[nodiscard]] inline std::string_view native_low_thrust_status_name(
    const NativeLowThrustStatus status
) noexcept {
    switch (status) {
        case NativeLowThrustStatus::converged:
            return "converged";
        case NativeLowThrustStatus::maximum_iterations:
            return "maximum_iterations";
        case NativeLowThrustStatus::trust_region_exhausted:
            return "trust_region_exhausted";
        case NativeLowThrustStatus::solver_failed:
            return "solver_failed";
    }
    return "unknown";
}

}  // namespace spacepdhcg::scvx
