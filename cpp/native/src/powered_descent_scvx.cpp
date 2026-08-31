#include "spacepdhcg/native/powered_descent_scvx.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace spacepdhcg::native {
namespace {

void require_finite(std::span<const double> values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }
}

[[nodiscard]] PoweredDescentControl project_thrust(
    const PoweredDescent3DofModel& model,
    std::array<double, 3> thrust
) {
    if (thrust[2] <= 0.0) {
        thrust = {0.0, 0.0, std::max(model.config().minimum_sigma, 1.0)};
    }
    const double horizontal = std::hypot(thrust[0], thrust[1]);
    const double maximum_horizontal =
        thrust[2] * std::tan(model.config().maximum_tilt_radians);
    if (horizontal > maximum_horizontal && horizontal > 0.0) {
        const double scale = maximum_horizontal / horizontal;
        thrust[0] *= scale;
        thrust[1] *= scale;
    }
    double norm = std::hypot(thrust[0], thrust[1], thrust[2]);
    if (norm > model.config().maximum_thrust) {
        const double scale = model.config().maximum_thrust / norm;
        for (double& value : thrust) {
            value *= scale;
        }
        norm = model.config().maximum_thrust;
    }
    if (norm < model.config().minimum_sigma) {
        thrust = {0.0, 0.0, model.config().minimum_sigma};
        norm = model.config().minimum_sigma;
    }
    return PoweredDescentControl{thrust[0], thrust[1], thrust[2], norm};
}

[[nodiscard]] double normalised_fuel(
    const PoweredDescent3DofModel& model,
    std::span<const PoweredDescentControl> controls
) {
    double total = 0.0;
    for (const auto& control : controls) {
        total += control[3];
    }
    return total /
           static_cast<double>(controls.size()) /
           model.config().maximum_thrust;
}

[[nodiscard]] std::array<double, 7> path_components(
    const PoweredDescentCqp& transcription,
    std::span<const PoweredDescentState> states,
    std::span<const PoweredDescentControl> controls
) {
    const auto diagnostics = transcription.model().path_diagnostics(states, controls);
    const auto& scales = transcription.config().state_trust_scales;
    const double position_scale = std::max({scales[0], scales[1], scales[2]});
    return {
        diagnostics.thrust_epigraph / transcription.model().config().maximum_thrust,
        diagnostics.throttle_lower / transcription.model().config().maximum_thrust,
        diagnostics.throttle_upper / transcription.model().config().maximum_thrust,
        diagnostics.tilt / transcription.model().config().maximum_thrust,
        diagnostics.minimum_mass * scales[6],
        diagnostics.altitude * position_scale,
        diagnostics.glide_slope * position_scale,
    };
}

[[nodiscard]] double actual_merit(
    const PoweredDescentCqp& transcription,
    const PoweredDescentOuterConfig& outer,
    std::span<const PoweredDescentState> states,
    std::span<const PoweredDescentControl> controls,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity
) {
    const auto& scales = transcription.config().state_trust_scales;
    const auto& terminal = states.back();
    double terminal_measure = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        terminal_measure += std::abs(
            (terminal[axis] - target_position[axis]) * scales[axis]
        );
        terminal_measure += std::abs(
            (terminal[3 + axis] - target_velocity[axis]) * scales[3 + axis]
        );
    }
    const auto components = path_components(transcription, states, controls);
    double path_measure = 0.0;
    for (double value : components) {
        path_measure += value;
    }
    return normalised_fuel(transcription.model(), controls) +
           outer.feasibility_penalty * (terminal_measure + path_measure);
}

[[nodiscard]] double model_merit(
    const PoweredDescentCqp& transcription,
    const PoweredDescentOuterConfig& outer,
    const PoweredDescentDecision& decision,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity
) {
    const auto& scales = transcription.config().state_trust_scales;
    const auto& terminal = decision.states.back();
    double terminal_measure = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        terminal_measure += std::abs(
            (terminal[axis] - target_position[axis]) * scales[axis]
        );
        terminal_measure += std::abs(
            (terminal[3 + axis] - target_velocity[axis]) * scales[3 + axis]
        );
    }
    const auto components = path_components(
        transcription,
        decision.states,
        decision.controls
    );
    double path_measure = 0.0;
    for (double value : components) {
        path_measure += value;
    }
    double virtual_measure = 0.0;
    for (const auto& virtual_control : decision.virtual_controls) {
        for (std::size_t component = 0;
             component < powered_descent_state_dimension;
             ++component) {
            virtual_measure += std::abs(virtual_control[component]) * scales[component];
        }
    }
    virtual_measure /= static_cast<double>(decision.virtual_controls.size());
    return normalised_fuel(transcription.model(), decision.controls) +
           outer.feasibility_penalty * (terminal_measure + path_measure) +
           outer.virtual_penalty * virtual_measure;
}

[[nodiscard]] double step_fraction(
    const PoweredDescentCqp& transcription,
    const PoweredDescentDecision& candidate,
    std::span<const PoweredDescentState> reference_states,
    std::span<const PoweredDescentControl> reference_controls,
    double radius
) {
    const auto& state_scales = transcription.config().state_trust_scales;
    const auto& control_scales = transcription.config().control_trust_scales;
    double maximum = 0.0;
    for (std::size_t interval = 0; interval < reference_controls.size(); ++interval) {
        double norm_squared = 0.0;
        for (std::size_t component = 0;
             component < powered_descent_state_dimension;
             ++component) {
            const double value =
                (candidate.states[interval][component] - reference_states[interval][component]) *
                state_scales[component];
            norm_squared += value * value;
        }
        for (std::size_t component = 0;
             component < powered_descent_control_dimension;
             ++component) {
            const double value =
                (candidate.controls[interval][component] -
                 reference_controls[interval][component]) *
                control_scales[component];
            norm_squared += value * value;
        }
        maximum = std::max(maximum, std::sqrt(norm_squared) / radius);
    }
    double terminal_norm_squared = 0.0;
    for (std::size_t component = 0;
         component < powered_descent_state_dimension;
         ++component) {
        const double value =
            (candidate.states.back()[component] - reference_states.back()[component]) *
            state_scales[component];
        terminal_norm_squared += value * value;
    }
    return std::max(maximum, std::sqrt(terminal_norm_squared) / radius);
}

[[nodiscard]] OuterResidual outer_residual(
    const PoweredDescentCqp& transcription,
    std::span<const PoweredDescentState> decision_states,
    std::span<const PoweredDescentControl> controls,
    std::span<const PoweredDescentState> rollout,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    double candidate_step_fraction
) {
    const auto& scales = transcription.config().state_trust_scales;
    double dynamics = 0.0;
    for (std::size_t node = 0; node < rollout.size(); ++node) {
        for (std::size_t component = 0;
             component < powered_descent_state_dimension;
             ++component) {
            dynamics = std::max(
                dynamics,
                std::abs(
                    (decision_states[node][component] - rollout[node][component]) *
                    scales[component]
                )
            );
        }
    }
    double terminal = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        terminal = std::max(
            terminal,
            std::abs((rollout.back()[axis] - target_position[axis]) * scales[axis])
        );
        terminal = std::max(
            terminal,
            std::abs(
                (rollout.back()[3 + axis] - target_velocity[axis]) * scales[3 + axis]
            )
        );
    }
    const auto components = path_components(transcription, rollout, controls);
    const double path = *std::max_element(components.begin(), components.end());
    return OuterResidual{dynamics, path, terminal, candidate_step_fraction};
}

struct Candidate {
    SessionSolveResult solve{};
    PoweredDescentDecision decision{};
    std::vector<PoweredDescentState> rollout{};
    PoweredDescentCqpDiagnostics convex{};
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

[[nodiscard]] Candidate evaluate_candidate(
    const PoweredDescentCqp& transcription,
    const PoweredDescentOuterConfig& outer,
    const OwnedCqp& problem,
    SessionSolveResult solve,
    std::span<const PoweredDescentState> current_states,
    std::span<const PoweredDescentControl> current_controls,
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    double current_merit,
    const OuterResidual& current_residual,
    double trust_radius,
    double solver_tolerance
) {
    Candidate result{};
    result.solve = std::move(solve);
    result.decision = transcription.decode(result.solve.backend.primal);
    result.convex = transcription.diagnostics(
        result.solve.backend.primal,
        problem,
        target_position,
        target_velocity
    );
    result.step_fraction = step_fraction(
        transcription,
        result.decision,
        current_states,
        current_controls,
        trust_radius
    );
    try {
        result.rollout = transcription.model().rollout_euler(
            initial_state,
            result.decision.controls,
            transcription.config().step_seconds
        );
    } catch (const std::invalid_argument&) {
        result.actual_merit = std::numeric_limits<double>::infinity();
        result.residual = OuterResidual{
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            result.step_fraction,
        };
        return result;
    }

    result.model_merit = model_merit(
        transcription,
        outer,
        result.decision,
        target_position,
        target_velocity
    );
    result.actual_merit = actual_merit(
        transcription,
        outer,
        result.rollout,
        result.decision.controls,
        target_position,
        target_velocity
    );
    result.predicted_reduction = current_merit - result.model_merit;
    result.actual_reduction = current_merit - result.actual_merit;
    if (result.predicted_reduction > outer.minimum_predicted_reduction) {
        result.agreement = result.actual_reduction / result.predicted_reduction;
    }
    result.residual = outer_residual(
        transcription,
        result.decision.states,
        result.decision.controls,
        result.rollout,
        target_position,
        target_velocity,
        result.step_fraction
    );
    result.restoration =
        result.residual.maximum() < outer.restoration_reduction * current_residual.maximum();
    const bool numerical_acceptance = result.solve.acceptable(
        std::max(solver_tolerance, 1.0e-10)
    );
    result.accepted = numerical_acceptance && std::isfinite(result.actual_merit) &&
        ((result.actual_reduction > outer.minimum_actual_reduction &&
          result.agreement >= outer.acceptance_threshold) ||
         result.restoration);
    return result;
}

}  // namespace

void PoweredDescentOuterConfig::validate() const {
    if (maximum_iterations == 0 || minimum_iterations > maximum_iterations) {
        throw std::invalid_argument("powered-descent outer iteration budget is invalid");
    }
    for (const auto& item : std::array{
             std::pair{"convergence_tolerance", convergence_tolerance},
             std::pair{"step_tolerance", step_tolerance},
             std::pair{"feasibility_penalty", feasibility_penalty},
             std::pair{"virtual_penalty", virtual_penalty},
             std::pair{"minimum_actual_reduction", minimum_actual_reduction},
             std::pair{"minimum_predicted_reduction", minimum_predicted_reduction},
         }) {
        if (!std::isfinite(item.second) || item.second <= 0.0) {
            throw std::invalid_argument(std::string(item.first) + " must be finite and positive");
        }
    }
    if (!(acceptance_threshold >= 0.0 && acceptance_threshold < 1.0) ||
        !(restoration_reduction > 0.0 && restoration_reduction < 1.0)) {
        throw std::invalid_argument("powered-descent acceptance parameters are invalid");
    }
}

PoweredDescentReference make_powered_descent_reference(
    const PoweredDescent3DofModel& model,
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    std::size_t intervals,
    double step_seconds
) {
    require_finite(initial_state, "initial state");
    require_finite(target_position, "target position");
    require_finite(target_velocity, "target velocity");
    if (intervals < 2 || !std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("reference horizon and step must be valid");
    }
    if (initial_state[6] <= model.config().minimum_mass) {
        throw std::invalid_argument("initial mass must exceed the configured reserve");
    }

    const double count = static_cast<double>(intervals);
    const double first_moment = count * (count - 1.0) / 2.0;
    const double second_moment =
        (count - 1.0) * count * (2.0 * count - 1.0) / 6.0;
    const double a00 = count;
    const double a01 = count * count;
    const double a10 = step_seconds * first_moment;
    const double a11 = step_seconds * second_moment;
    const double determinant = a00 * a11 - a01 * a10;
    if (std::abs(determinant) < 1.0e-15) {
        throw std::runtime_error("powered-descent reference interpolation is singular");
    }

    std::vector<std::array<double, 3>> velocities(intervals + 1U);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double rhs0 = target_velocity[axis] - initial_state[3 + axis];
        const double rhs1 = target_position[axis] - initial_state[axis] -
                            step_seconds * count * initial_state[3 + axis];
        const double linear = (rhs0 * a11 - a01 * rhs1) / determinant;
        const double quadratic = (a00 * rhs1 - rhs0 * a10) / determinant;
        for (std::size_t node = 0; node <= intervals; ++node) {
            const double index = static_cast<double>(node);
            velocities[node][axis] = initial_state[3 + axis] +
                                     linear * index + quadratic * index * index;
        }
    }

    PoweredDescentReference result{};
    result.states.resize(intervals + 1U);
    result.controls.resize(intervals);
    std::copy(initial_state.begin(), initial_state.end(), result.states.front().begin());
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        std::array<double, 3> requested_thrust{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double acceleration =
                (velocities[interval + 1U][axis] - velocities[interval][axis]) /
                step_seconds;
            requested_thrust[axis] = result.states[interval][6] *
                                     (acceleration - model.config().gravity[axis]);
        }
        result.controls[interval] = project_thrust(model, requested_thrust);
        result.states[interval + 1U] = model.euler_step(
            result.states[interval],
            result.controls[interval],
            step_seconds
        );
        if (result.states[interval + 1U][6] <= model.config().minimum_mass) {
            throw std::runtime_error("initial reference consumes the propellant reserve");
        }
    }
    return result;
}

PoweredDescentScvxDriver::PoweredDescentScvxDriver(
    PoweredDescentCqp transcription,
    CqpWorkspaceFactory backend_factory,
    PoweredDescentOuterConfig outer_config,
    ForcingRuleConfig forcing_config,
    TrustRegionConfig trust_config
)
    : transcription_(std::move(transcription)),
      backend_factory_(std::move(backend_factory)),
      outer_config_(outer_config),
      forcing_config_(forcing_config),
      trust_config_(trust_config) {
    if (!backend_factory_) {
        throw std::invalid_argument("SCvx driver requires a backend factory");
    }
    outer_config_.validate();
    forcing_config_.validate();
    trust_config_.validate();
}

PoweredDescentScvxResult PoweredDescentScvxDriver::solve(
    std::span<const double, powered_descent_state_dimension> initial_state,
    std::span<const double, 3> target_position,
    std::span<const double, 3> target_velocity,
    const PoweredDescentReference* initial_reference
) const {
    PoweredDescentReference reference = initial_reference == nullptr
        ? make_powered_descent_reference(
              transcription_.model(),
              initial_state,
              target_position,
              target_velocity,
              static_cast<std::size_t>(transcription_.config().intervals),
              transcription_.config().step_seconds
          )
        : *initial_reference;
    if (reference.states.size() !=
            static_cast<std::size_t>(transcription_.config().intervals + 1) ||
        reference.controls.size() !=
            static_cast<std::size_t>(transcription_.config().intervals)) {
        throw std::invalid_argument("initial SCvx reference has the wrong horizon");
    }

    auto problem = transcription_.make_cqp(
        reference.states,
        reference.controls,
        initial_state,
        target_position,
        target_velocity,
        trust_config_.initial_radius
    );
    PersistentCqpSession session(problem, backend_factory_);
    AdaptiveForcingRule forcing(forcing_config_);
    TrustRegionController trust(trust_config_);

    double current_merit = actual_merit(
        transcription_,
        outer_config_,
        reference.states,
        reference.controls,
        target_position,
        target_velocity
    );
    OuterResidual current_residual = outer_residual(
        transcription_,
        reference.states,
        reference.controls,
        reference.states,
        target_position,
        target_velocity,
        1.0
    );
    std::size_t accepted_streak = 0;
    std::size_t accepted_iterations = 0;
    std::optional<double> previous_agreement{};
    std::vector<PoweredDescentScvxIteration> records;
    std::string status = "maximum_iterations";

    for (std::size_t iteration = 0;
         iteration < outer_config_.maximum_iterations;
         ++iteration) {
        const auto request = forcing.request(
            static_cast<int>(iteration),
            current_residual,
            static_cast<int>(accepted_streak),
            previous_agreement
        );
        transcription_.update_numerical_values(
            problem,
            reference.states,
            reference.controls,
            initial_state,
            target_position,
            target_velocity,
            trust.radius()
        );
        CqpSolveOptions options{
            request.tolerance,
            request.tolerance,
            static_cast<std::size_t>(request.iteration_limit),
        };
        auto candidate = evaluate_candidate(
            transcription_,
            outer_config_,
            problem,
            session.solve(problem, options),
            reference.states,
            reference.controls,
            initial_state,
            target_position,
            target_velocity,
            current_merit,
            current_residual,
            trust.radius(),
            request.tolerance
        );

        double effective_tolerance = request.tolerance;
        bool re_solved = false;
        for (std::size_t resolve = 0;
             resolve < outer_config_.maximum_resolves_per_iteration;
             ++resolve) {
            if (!forcing.should_resolve(
                    candidate.accepted,
                    candidate.solve.backend.primal_residual,
                    candidate.solve.backend.dual_residual,
                    effective_tolerance
                )) {
                break;
            }
            effective_tolerance = forcing.refined_tolerance(effective_tolerance);
            options.optimality_tolerance = effective_tolerance;
            options.feasibility_tolerance = effective_tolerance;
            options.iteration_limit = std::max(
                options.iteration_limit,
                static_cast<std::size_t>(forcing.config().refinement_iteration_limit)
            );
            candidate = evaluate_candidate(
                transcription_,
                outer_config_,
                problem,
                session.solve(problem, options),
                reference.states,
                reference.controls,
                initial_state,
                target_position,
                target_velocity,
                current_merit,
                current_residual,
                trust.radius(),
                effective_tolerance
            );
            re_solved = true;
        }

        const auto trust_update = trust.update(
            candidate.accepted,
            candidate.agreement,
            candidate.step_fraction
        );
        records.push_back(PoweredDescentScvxIteration{
            iteration,
            request.phase,
            request.tolerance,
            effective_tolerance,
            candidate.solve.backend.iterations,
            candidate.solve.backend.status,
            candidate.solve.backend.primal_residual,
            candidate.solve.backend.dual_residual,
            trust_update.radius_before,
            trust_update.radius_after,
            trust_update.action,
            candidate.step_fraction,
            candidate.predicted_reduction,
            candidate.actual_reduction,
            candidate.agreement,
            candidate.accepted,
            candidate.accepted && candidate.restoration,
            re_solved,
            candidate.solve.warm_started,
            current_merit,
            candidate.actual_merit,
            candidate.residual,
            candidate.convex,
        });

        if (!candidate.solve.backend.solved()) {
            status = "solver_failed";
            break;
        }
        if (candidate.accepted) {
            reference.states = std::move(candidate.rollout);
            reference.controls = std::move(candidate.decision.controls);
            current_merit = candidate.actual_merit;
            current_residual = candidate.residual;
            ++accepted_streak;
            ++accepted_iterations;
            previous_agreement = candidate.agreement;
            if (iteration + 1U >= outer_config_.minimum_iterations &&
                current_residual.maximum() <= outer_config_.convergence_tolerance &&
                candidate.step_fraction <= outer_config_.step_tolerance) {
                status = "converged";
                break;
            }
        } else {
            accepted_streak = 0;
            previous_agreement.reset();
            if (trust.exhausted()) {
                status = "trust_region_exhausted";
                break;
            }
        }
    }

    return PoweredDescentScvxResult{
        std::move(status),
        reference.states,
        reference.controls,
        current_merit,
        current_residual,
        transcription_.model().path_diagnostics(
            reference.states,
            reference.controls
        ),
        std::move(records),
        accepted_iterations,
        session.update_count(),
        session.solve_count(),
    };
}

}  // namespace spacepdhcg::native
