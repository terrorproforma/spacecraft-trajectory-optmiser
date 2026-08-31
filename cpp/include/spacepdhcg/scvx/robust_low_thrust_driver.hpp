#pragma once

#include "spacepdhcg/core/host_backend.hpp"
#include "spacepdhcg/distributed/risk.hpp"
#include "spacepdhcg/distributed/risk_cqp.hpp"
#include "spacepdhcg/distributed/scenario_cqp.hpp"
#include "spacepdhcg/distributed/scenario_layout.hpp"
#include "spacepdhcg/scvx/low_thrust_driver.hpp"
#include "spacepdhcg/scvx/policies.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::scvx {

struct RobustLowThrustScenario {
    std::string name{};
    double probability{0.0};
    dynamics::LowThrustTwoBodyModel model{};
    dynamics::LowThrustState initial{};
    dynamics::LowThrustState target{};

    void validate() const {
        if (name.empty() || !std::isfinite(probability) || probability <= 0.0) {
            throw std::invalid_argument(
                "robust low-thrust scenario name and probability are invalid"
            );
        }
        const dynamics::LowThrustControl zero{0.0, 0.0, 0.0, 0.0};
        static_cast<void>(model.dynamics(initial, zero));
        static_cast<void>(model.dynamics(target, zero));
        if (initial[6U] <= model.config().minimum_mass) {
            throw std::invalid_argument(
                "robust low-thrust initial mass must exceed the scenario reserve"
            );
        }
    }
};

struct RobustLowThrustConfig {
    std::size_t common_prefix{std::numeric_limits<std::size_t>::max()};
    distributed::RiskMeasure risk_measure{distributed::RiskMeasure::conditional_value_at_risk};
    double risk_confidence{0.95};
    double risk_weight{1.0};
    NativeLowThrustOuterConfig outer{};
    ForcingRuleConfig forcing{};
    TrustRegionConfig trust{};

    void validate(const std::size_t intervals) const {
        if (common_prefix != std::numeric_limits<std::size_t>::max()
            && common_prefix > intervals) {
            throw std::invalid_argument(
                "robust low-thrust common prefix may not exceed the horizon"
            );
        }
        if (!std::isfinite(risk_confidence) || risk_confidence < 0.0
            || risk_confidence >= 1.0 || !std::isfinite(risk_weight)
            || risk_weight <= 0.0) {
            throw std::invalid_argument(
                "robust low-thrust risk confidence or weight is invalid"
            );
        }
        outer.validate();
        forcing.validate();
        trust.validate();
    }
};

struct RobustLowThrustScenarioResult {
    std::string name{};
    std::vector<dynamics::LowThrustState> states{};
    std::vector<dynamics::LowThrustControl> controls{};
    dynamics::LowThrustPathDiagnostics path{};
    double terminal_error{0.0};
    double propellant{0.0};
    double delta_v{0.0};
};

struct RobustLowThrustIterationRecord {
    std::size_t iteration{0U};
    SolvePhase phase{SolvePhase::repair};
    double requested_tolerance{0.0};
    double effective_tolerance{0.0};
    std::size_t solver_iterations{0U};
    double primal_residual{0.0};
    double dual_residual{0.0};
    double trust_radius_before{0.0};
    double trust_radius_after{0.0};
    TrustAction trust_action{TrustAction::retain};
    double step_fraction{0.0};
    double predicted_reduction{0.0};
    double actual_reduction{0.0};
    double agreement{0.0};
    double nonanticipativity_violation{0.0};
    double risk_epigraph_violation{0.0};
    bool accepted{false};
    bool restoration_accepted{false};
    bool re_solved{false};
    OuterResidual residual{};
};

struct RobustLowThrustScvxResult {
    NativeLowThrustStatus status{NativeLowThrustStatus::maximum_iterations};
    std::vector<RobustLowThrustScenarioResult> scenarios{};
    distributed::RiskSummary propellant_risk{};
    distributed::RiskSummary delta_v_risk{};
    double selected_propellant_risk{0.0};
    double selected_delta_v_risk{0.0};
    double merit{0.0};
    OuterResidual residual{};
    std::vector<RobustLowThrustIterationRecord> iterations{};
    std::size_t accepted_iterations{0U};
    std::size_t backend_creations{0U};
    std::size_t backend_updates{0U};

    [[nodiscard]] bool converged() const noexcept {
        return status == NativeLowThrustStatus::converged;
    }
};

using RobustLowThrustReferences = std::vector<LowThrustReference>;

namespace robust_low_thrust_detail {

inline double selected_risk(
    const distributed::RiskSummary& summary,
    const distributed::RiskMeasure measure
) noexcept {
    switch (measure) {
        case distributed::RiskMeasure::expected:
            return summary.expected;
        case distributed::RiskMeasure::worst_case:
            return summary.worst;
        case distributed::RiskMeasure::conditional_value_at_risk:
            return summary.conditional_value_at_risk;
    }
    return summary.expected;
}

inline double norm3(const dynamics::LowThrustControl& control) noexcept {
    return std::sqrt(
        control[0U] * control[0U] + control[1U] * control[1U]
        + control[2U] * control[2U]
    );
}

inline double integrated_delta_v(
    const std::vector<dynamics::LowThrustState>& states,
    const std::vector<dynamics::LowThrustControl>& controls,
    const dynamics::LowThrustTwoBodyModel& model,
    const double step_seconds
) {
    if (states.size() != controls.size() + 1U) {
        throw std::invalid_argument(
            "robust low-thrust delta-v requires N controls and N+1 states"
        );
    }
    double result{0.0};
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        result += step_seconds * model.config().thrust_to_acceleration
                  * controls[interval][3U] / states[interval][6U];
    }
    return result;
}

}  // namespace robust_low_thrust_detail

/// Host truth path for common-open-loop robust low-thrust SCvx.
///
/// Each scenario owns nonlinear states, virtual controls and recourse controls. Shared-prefix
/// controls are linked by the canonical block-arrow non-anticipativity rows. The monolithic
/// host CQP retains expected local objectives and adds an exact affine propellant risk term
/// (expected, worst-case or CVaR). Every candidate is then rolled out independently through
/// each uncertain nonlinear model before trust-region acceptance.
class RobustLowThrustScvxDriver {
  public:
    RobustLowThrustScvxDriver(
        transcription::LowThrustScvxConfig transcription_config,
        LowThrustHostBackendFactory backend_factory,
        RobustLowThrustConfig config = {}
    )
        : transcription_config_(transcription_config),
          backend_factory_(std::move(backend_factory)),
          config_(config) {
        transcription_config_.validate();
        config_.validate(transcription_config_.intervals);
        if (!backend_factory_) {
            throw std::invalid_argument(
                "robust low-thrust SCvx requires a host backend factory"
            );
        }
    }

    [[nodiscard]] RobustLowThrustScvxResult solve(
        std::vector<RobustLowThrustScenario> scenarios,
        std::optional<RobustLowThrustReferences> references = std::nullopt
    ) const {
        validate_scenarios(scenarios);
        const auto probabilities = scenario_probabilities(scenarios);
        const auto common_prefix = config_.common_prefix
                                           == std::numeric_limits<std::size_t>::max()
                                       ? transcription_config_.intervals
                                       : config_.common_prefix;
        std::vector<transcription::LowThrustSubproblem> subproblems{};
        subproblems.reserve(scenarios.size());
        for (const auto& scenario : scenarios) {
            subproblems.emplace_back(scenario.model, transcription_config_);
        }
        const auto& local_structure = subproblems.front().structure();
        for (const auto& subproblem : subproblems) {
            if (subproblem.structure().fingerprint() != local_structure.fingerprint()) {
                throw std::invalid_argument(
                    "robust low-thrust scenarios do not share one CQP topology"
                );
            }
        }
        const auto state_variables =
            (transcription_config_.intervals + 1U) * 7U;
        const auto control_variables = transcription_config_.intervals * 4U;
        const auto auxiliary_variables =
            static_cast<std::size_t>(local_structure.variables())
            - state_variables - control_variables;
        const auto tree = distributed::ScenarioTree::common_open_loop(
            scenarios.size(),
            transcription_config_.intervals,
            common_prefix,
            probabilities
        );
        const distributed::ScenarioCqpBundle bundle{
            tree,
            local_structure,
            7U,
            4U,
            auxiliary_variables,
        };
        const auto losses = propellant_losses(bundle, scenarios);
        std::vector<std::vector<std::size_t>> loss_patterns{};
        loss_patterns.reserve(losses.size());
        for (const auto& loss : losses) {
            loss_patterns.push_back(loss.indices);
        }
        const distributed::RiskAugmentedCqp risk{
            bundle.structure(),
            std::move(loss_patterns),
            config_.risk_measure,
            config_.risk_confidence,
        };

        auto current = references.has_value()
                           ? std::move(*references)
                           : initial_references(scenarios);
        validate_references(current, scenarios.size());
        enforce_shared_reference_controls(current, scenarios, common_prefix);
        rerollout_references(current, scenarios);
        auto current_evaluation = evaluate_references(
            current,
            scenarios,
            probabilities
        );
        auto current_merit = current_evaluation.merit;
        auto current_residual = current_evaluation.residual;

        auto trust_config = config_.trust;
        if (trust_config.initial_radius == TrustRegionConfig{}.initial_radius
            && transcription_config_.trust_radius != trust_config.initial_radius) {
            trust_config.initial_radius = transcription_config_.trust_radius;
            trust_config.maximum_radius = std::max(
                trust_config.maximum_radius,
                trust_config.initial_radius
            );
        }
        TrustRegionController trust{trust_config};
        AdaptiveForcingRule forcing{config_.forcing};
        core::HostBackendPointer backend{};
        core::HostWarmStart warm{};
        bool have_warm{false};
        std::size_t backend_creations{0U};
        std::size_t accepted_streak{0U};
        std::size_t accepted_iterations{0U};
        std::optional<double> previous_agreement{};
        std::vector<RobustLowThrustIterationRecord> records{};
        auto status = NativeLowThrustStatus::maximum_iterations;

        for (std::size_t iteration = 0;
             iteration < config_.outer.maximum_iterations;
             ++iteration) {
            const auto request = forcing.request(
                iteration,
                current_residual,
                accepted_streak,
                previous_agreement.value_or(
                    std::numeric_limits<double>::quiet_NaN()
                )
            );
            const auto local_values = make_local_values(
                subproblems,
                current,
                scenarios,
                trust.radius()
            );
            const auto base_values = bundle.values(local_values);
            auto risk_values = risk.values(
                base_values,
                losses,
                probabilities,
                config_.risk_weight
            );
            if (!backend) {
                backend = backend_factory_(
                    core::FixedCQP(risk.structure(), risk_values)
                );
                ++backend_creations;
                if (!backend || backend->structure().fingerprint()
                                    != risk.structure().fingerprint()) {
                    throw std::runtime_error(
                        "robust low-thrust backend returned incompatible topology"
                    );
                }
                warm.primal = reference_primal(
                    bundle,
                    risk,
                    subproblems,
                    current,
                    losses,
                    probabilities
                );
                warm.dual.assign(
                    static_cast<std::size_t>(risk.structure().duals()),
                    0.0
                );
                backend->warm_start(warm);
                have_warm = true;
            } else {
                backend->update(std::move(risk_values));
                if (have_warm) {
                    backend->warm_start(warm);
                }
            }

            auto effective_tolerance = request.tolerance;
            auto solution = backend->solve(
                effective_tolerance,
                request.iteration_limit
            );
            auto candidate = evaluate_candidate(
                solution,
                bundle,
                risk,
                losses,
                local_values,
                subproblems,
                scenarios,
                probabilities,
                current,
                current_merit,
                current_residual,
                trust.radius()
            );
            bool re_solved{false};
            for (std::size_t resolve = 0;
                 resolve < config_.outer.maximum_resolves_per_iteration;
                 ++resolve) {
                if (!forcing.should_resolve(
                        candidate.accepted,
                        solution.primal_residual,
                        solution.dual_residual,
                        effective_tolerance
                    )) {
                    break;
                }
                effective_tolerance = forcing.refined_tolerance(effective_tolerance);
                solution = backend->solve(
                    effective_tolerance,
                    std::max(
                        request.iteration_limit,
                        forcing.config().refinement_iteration_limit
                    )
                );
                candidate = evaluate_candidate(
                    solution,
                    bundle,
                    risk,
                    losses,
                    local_values,
                    subproblems,
                    scenarios,
                    probabilities,
                    current,
                    current_merit,
                    current_residual,
                    trust.radius()
                );
                re_solved = true;
            }

            const auto trust_update = trust.update(
                candidate.accepted,
                candidate.agreement,
                candidate.step_fraction
            );
            records.push_back(RobustLowThrustIterationRecord{
                iteration,
                request.phase,
                request.tolerance,
                effective_tolerance,
                solution.outer_iterations + solution.inner_iterations,
                solution.primal_residual,
                solution.dual_residual,
                trust_update.radius_before,
                trust_update.radius_after,
                trust_update.action,
                candidate.step_fraction,
                candidate.predicted_reduction,
                candidate.actual_reduction,
                candidate.agreement,
                candidate.nonanticipativity,
                candidate.risk_epigraph,
                candidate.accepted,
                candidate.restoration,
                re_solved,
                candidate.evaluation.residual,
            });

            if (!solution.solved()) {
                status = NativeLowThrustStatus::solver_failed;
                break;
            }
            if (candidate.accepted) {
                current = std::move(candidate.references);
                current_evaluation = std::move(candidate.evaluation);
                current_merit = current_evaluation.merit;
                current_residual = current_evaluation.residual;
                warm = core::HostWarmStart{solution.primal, solution.dual};
                have_warm = true;
                ++accepted_streak;
                ++accepted_iterations;
                previous_agreement = candidate.agreement;
                if (iteration + 1U >= config_.outer.minimum_iterations
                    && current_residual.maximum()
                           <= config_.outer.convergence_tolerance
                    && candidate.step_fraction <= config_.outer.step_tolerance) {
                    status = NativeLowThrustStatus::converged;
                    break;
                }
            } else {
                accepted_streak = 0U;
                previous_agreement.reset();
                if (trust.exhausted()) {
                    status = NativeLowThrustStatus::trust_region_exhausted;
                    break;
                }
            }
        }

        return make_result(
            status,
            current,
            scenarios,
            probabilities,
            current_merit,
            current_residual,
            std::move(records),
            accepted_iterations,
            backend_creations,
            backend ? backend->update_count() : 0U
        );
    }

  private:
    struct ReferenceEvaluation {
        std::vector<double> scenario_merits{};
        std::vector<double> propellant{};
        std::vector<double> delta_v{};
        distributed::RiskSummary propellant_risk{};
        distributed::RiskSummary delta_v_risk{};
        double merit{0.0};
        OuterResidual residual{};
    };

    struct Candidate {
        RobustLowThrustReferences references{};
        ReferenceEvaluation evaluation{};
        double model_merit{0.0};
        double step_fraction{0.0};
        double predicted_reduction{0.0};
        double actual_reduction{0.0};
        double agreement{-std::numeric_limits<double>::infinity()};
        double nonanticipativity{0.0};
        double risk_epigraph{0.0};
        bool accepted{false};
        bool restoration{false};
    };

    transcription::LowThrustScvxConfig transcription_config_{};
    LowThrustHostBackendFactory backend_factory_{};
    RobustLowThrustConfig config_{};

    void validate_scenarios(
        const std::vector<RobustLowThrustScenario>& scenarios
    ) const {
        if (scenarios.empty()) {
            throw std::invalid_argument(
                "robust low-thrust SCvx requires at least one scenario"
            );
        }
        std::vector<std::string> names{};
        std::vector<double> probabilities{};
        for (const auto& scenario : scenarios) {
            scenario.validate();
            names.push_back(scenario.name);
            probabilities.push_back(scenario.probability);
        }
        std::sort(names.begin(), names.end());
        if (std::adjacent_find(names.begin(), names.end()) != names.end()) {
            throw std::invalid_argument(
                "robust low-thrust scenario names must be unique"
            );
        }
        distributed::validate_probability_distribution(probabilities);
    }

    static std::vector<double> scenario_probabilities(
        const std::vector<RobustLowThrustScenario>& scenarios
    ) {
        std::vector<double> result{};
        result.reserve(scenarios.size());
        for (const auto& scenario : scenarios) {
            result.push_back(scenario.probability);
        }
        return result;
    }

    RobustLowThrustReferences initial_references(
        const std::vector<RobustLowThrustScenario>& scenarios
    ) const {
        RobustLowThrustReferences result{};
        result.reserve(scenarios.size());
        for (const auto& scenario : scenarios) {
            result.push_back(make_native_low_thrust_reference(
                scenario.model,
                scenario.initial,
                scenario.target,
                transcription_config_.intervals,
                transcription_config_.step_seconds,
                transcription_config_.discretisation
                    == transcription::DiscretisationMethod::rk4_finite_difference
            ));
        }
        return result;
    }

    void validate_references(
        const RobustLowThrustReferences& references,
        const std::size_t scenario_count
    ) const {
        if (references.size() != scenario_count) {
            throw std::invalid_argument(
                "one low-thrust reference is required per scenario"
            );
        }
        for (const auto& reference : references) {
            if (reference.first.size() != transcription_config_.intervals + 1U
                || reference.second.size() != transcription_config_.intervals) {
                throw std::invalid_argument(
                    "robust low-thrust reference has the wrong horizon"
                );
            }
        }
    }

    void enforce_shared_reference_controls(
        RobustLowThrustReferences& references,
        const std::vector<RobustLowThrustScenario>& scenarios,
        const std::size_t common_prefix
    ) const {
        const auto probabilities = scenario_probabilities(scenarios);
        for (std::size_t interval = 0; interval < common_prefix; ++interval) {
            dynamics::LowThrustControl shared{0.0, 0.0, 0.0, 0.0};
            for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
                for (std::size_t component = 0; component < 3U; ++component) {
                    shared[component] += probabilities[scenario]
                                         * references[scenario].second[interval][component];
                }
            }
            auto magnitude = robust_low_thrust_detail::norm3(shared);
            double maximum = std::numeric_limits<double>::infinity();
            for (const auto& scenario : scenarios) {
                maximum = std::min(maximum, scenario.model.config().maximum_thrust);
            }
            if (magnitude > maximum) {
                const auto scale = maximum / magnitude;
                for (std::size_t component = 0; component < 3U; ++component) {
                    shared[component] *= scale;
                }
                magnitude = maximum;
            }
            shared[3U] = magnitude;
            for (auto& reference : references) {
                reference.second[interval] = shared;
            }
        }
    }

    void rerollout_references(
        RobustLowThrustReferences& references,
        const std::vector<RobustLowThrustScenario>& scenarios
    ) const {
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            references[scenario].first = scenarios[scenario].model.rollout(
                scenarios[scenario].initial,
                references[scenario].second,
                transcription_config_.step_seconds,
                transcription_config_.discretisation
                    == transcription::DiscretisationMethod::rk4_finite_difference
            );
        }
    }

    std::vector<distributed::AffineScenarioLoss> propellant_losses(
        const distributed::ScenarioCqpBundle& bundle,
        const std::vector<RobustLowThrustScenario>& scenarios
    ) const {
        std::vector<distributed::AffineScenarioLoss> result{};
        result.reserve(scenarios.size());
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            distributed::AffineScenarioLoss loss{};
            for (std::size_t interval = 0;
                 interval < transcription_config_.intervals;
                 ++interval) {
                const auto [begin, end] = bundle.layout().control_range(
                    scenario,
                    interval
                );
                static_cast<void>(end);
                loss.indices.push_back(begin + 3U);
                loss.coefficients.push_back(
                    scenarios[scenario].model.config().mass_flow_coefficient
                    * transcription_config_.step_seconds
                );
            }
            loss.validate(
                static_cast<std::size_t>(bundle.structure().variables())
            );
            result.push_back(std::move(loss));
        }
        return result;
    }

    std::vector<core::NumericValues> make_local_values(
        const std::vector<transcription::LowThrustSubproblem>& subproblems,
        const RobustLowThrustReferences& references,
        const std::vector<RobustLowThrustScenario>& scenarios,
        const double trust_radius
    ) const {
        std::vector<core::NumericValues> result{};
        result.reserve(scenarios.size());
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            result.push_back(subproblems[scenario].values(
                references[scenario].first,
                references[scenario].second,
                scenarios[scenario].initial,
                scenarios[scenario].target,
                trust_radius
            ));
        }
        return result;
    }

    std::vector<double> reference_primal(
        const distributed::ScenarioCqpBundle& bundle,
        const distributed::RiskAugmentedCqp& risk,
        const std::vector<transcription::LowThrustSubproblem>& subproblems,
        const RobustLowThrustReferences& references,
        const std::vector<distributed::AffineScenarioLoss>& losses,
        const std::vector<double>& probabilities
    ) const {
        std::vector<double> base(
            static_cast<std::size_t>(bundle.structure().variables()),
            0.0
        );
        for (std::size_t scenario = 0; scenario < references.size(); ++scenario) {
            const auto local = subproblems[scenario].reference_decision(
                references[scenario].first,
                references[scenario].second
            );
            const auto [begin, end] = bundle.layout().scenario_range(scenario);
            static_cast<void>(end);
            std::copy(
                local.begin(),
                local.end(),
                base.begin() + static_cast<std::ptrdiff_t>(begin)
            );
        }
        for (const auto& block : bundle.layout().consensus_blocks()) {
            const auto scenario = block.node.scenario_indices.front();
            const auto& control = references[scenario].second[block.node.stage];
            std::copy(
                control.begin(),
                control.end(),
                base.begin() + static_cast<std::ptrdiff_t>(block.offset)
            );
        }
        std::vector<double> result(
            static_cast<std::size_t>(risk.structure().variables()),
            0.0
        );
        std::copy(base.begin(), base.end(), result.begin());
        std::vector<double> loss_values{};
        loss_values.reserve(losses.size());
        for (const auto& loss : losses) {
            loss_values.push_back(loss.evaluate(base));
        }
        const auto summary = distributed::aggregate_scenario_risk(
            loss_values,
            probabilities,
            config_.risk_confidence
        );
        if (const auto threshold = risk.threshold_index(); threshold.has_value()) {
            result[*threshold] = config_.risk_measure
                                         == distributed::RiskMeasure::worst_case
                                     ? summary.worst
                                     : summary.value_at_risk;
        }
        if (config_.risk_measure
            == distributed::RiskMeasure::conditional_value_at_risk) {
            for (std::size_t scenario = 0; scenario < losses.size(); ++scenario) {
                result[*risk.excess_index(scenario)] = std::max(
                    0.0,
                    loss_values[scenario] - summary.value_at_risk
                );
            }
        }
        return result;
    }

    struct DecodedCandidate {
        std::vector<LowThrustDecodedDecision> local{};
        distributed::ScenarioPrimal scenario{};
    };

    DecodedCandidate decode_candidate(
        const std::vector<double>& primal,
        const distributed::ScenarioCqpBundle& bundle,
        const distributed::RiskAugmentedCqp& risk,
        const std::vector<transcription::LowThrustSubproblem>& subproblems
    ) const {
        DecodedCandidate result{};
        const auto base = risk.base_primal(primal);
        result.scenario = bundle.decode_primal(base);
        result.local.reserve(subproblems.size());
        for (std::size_t scenario = 0; scenario < subproblems.size(); ++scenario) {
            result.local.push_back(decode_low_thrust_decision(
                subproblems[scenario],
                result.scenario.local[scenario]
            ));
        }
        for (std::size_t block = 0;
             block < bundle.layout().consensus_blocks().size();
             ++block) {
            const auto& descriptor = bundle.layout().consensus_blocks()[block];
            for (const auto scenario : descriptor.node.scenario_indices) {
                result.local[scenario].controls[descriptor.node.stage] =
                    dynamics::LowThrustControl{
                        result.scenario.consensus[block][0U],
                        result.scenario.consensus[block][1U],
                        result.scenario.consensus[block][2U],
                        result.scenario.consensus[block][3U],
                    };
            }
        }
        return result;
    }

    Candidate evaluate_candidate(
        const core::HostCqpSolution& solution,
        const distributed::ScenarioCqpBundle& bundle,
        const distributed::RiskAugmentedCqp& risk,
        const std::vector<distributed::AffineScenarioLoss>& losses,
        const std::vector<core::NumericValues>& local_values,
        const std::vector<transcription::LowThrustSubproblem>& subproblems,
        const std::vector<RobustLowThrustScenario>& scenarios,
        const std::vector<double>& probabilities,
        const RobustLowThrustReferences& current,
        const double current_merit,
        const OuterResidual& current_residual,
        const double trust_radius
    ) const {
        Candidate candidate{};
        const auto decoded = decode_candidate(
            solution.primal,
            bundle,
            risk,
            subproblems
        );
        candidate.nonanticipativity = bundle.maximum_nonanticipativity_violation(
            risk.base_primal(solution.primal)
        );
        candidate.risk_epigraph = risk.diagnostics(
            solution.primal,
            losses,
            probabilities
        ).maximum_epigraph_violation;
        candidate.references.resize(scenarios.size());
        bool rollout_valid{true};
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            candidate.references[scenario].second = decoded.local[scenario].controls;
            try {
                candidate.references[scenario].first = scenarios[scenario].model.rollout(
                    scenarios[scenario].initial,
                    decoded.local[scenario].controls,
                    transcription_config_.step_seconds,
                    transcription_config_.discretisation
                        == transcription::DiscretisationMethod::rk4_finite_difference
                );
            } catch (const std::exception&) {
                rollout_valid = false;
                break;
            }
        }
        candidate.step_fraction = maximum_step_fraction(
            decoded.local,
            current,
            trust_radius
        );
        candidate.evaluation = rollout_valid
                                   ? evaluate_references(
                                         candidate.references,
                                         scenarios,
                                         probabilities,
                                         &decoded.local,
                                         candidate.step_fraction
                                     )
                                   : invalid_evaluation(candidate.step_fraction);
        candidate.model_merit = model_merit(
            decoded.local,
            scenarios,
            probabilities
        );
        candidate.predicted_reduction = current_merit - candidate.model_merit;
        candidate.actual_reduction = current_merit - candidate.evaluation.merit;
        if (candidate.predicted_reduction
            > config_.outer.minimum_predicted_reduction) {
            candidate.agreement = candidate.actual_reduction
                                  / candidate.predicted_reduction;
        }
        candidate.restoration = rollout_valid
                                && candidate.evaluation.residual.maximum()
                                       < config_.outer.restoration_reduction
                                             * current_residual.maximum();
        double convex_violation{0.0};
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            convex_violation = std::max(
                convex_violation,
                subproblems[scenario]
                    .diagnostics(
                        result_local_primal(decoded, scenario),
                        local_values[scenario]
                    )
                    .maximum_violation()
            );
        }
        const auto convex_tolerance = std::max(
            1.0e-7,
            10.0 * std::max(solution.primal_residual, solution.dual_residual)
        );
        candidate.accepted = solution.solved() && rollout_valid
                             && convex_violation <= convex_tolerance
                             && candidate.nonanticipativity <= convex_tolerance
                             && candidate.risk_epigraph <= convex_tolerance
                             && std::isfinite(candidate.evaluation.merit)
                             && ((candidate.actual_reduction
                                      > config_.outer.minimum_actual_reduction
                                  && candidate.agreement
                                         >= config_.outer.acceptance_threshold)
                                 || candidate.restoration);
        return candidate;
    }

    static const std::vector<double>& result_local_primal(
        const DecodedCandidate& decoded,
        const std::size_t scenario
    ) {
        return decoded.scenario.local[scenario];
    }

    ReferenceEvaluation evaluate_references(
        const RobustLowThrustReferences& references,
        const std::vector<RobustLowThrustScenario>& scenarios,
        const std::vector<double>& probabilities,
        const std::vector<LowThrustDecodedDecision>* decision = nullptr,
        const double step_fraction_value = 0.0
    ) const {
        ReferenceEvaluation result{};
        result.scenario_merits.reserve(scenarios.size());
        result.propellant.reserve(scenarios.size());
        result.delta_v.reserve(scenarios.size());
        double maximum_dynamics{0.0};
        double maximum_path{0.0};
        double maximum_terminal{0.0};
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            const auto& states = references[scenario].first;
            const auto& controls = references[scenario].second;
            const auto path = scenarios[scenario].model.path_diagnostics(states, controls);
            const auto terminal = terminal_error(
                states.back(),
                scenarios[scenario].target
            );
            maximum_path = std::max(
                maximum_path,
                path_violation(path, scenarios[scenario].model)
            );
            maximum_terminal = std::max(maximum_terminal, terminal);
            if (decision != nullptr) {
                for (std::size_t node = 0; node < states.size(); ++node) {
                    for (std::size_t component = 0; component < 7U; ++component) {
                        maximum_dynamics = std::max(
                            maximum_dynamics,
                            std::abs(
                                ((*decision)[scenario].states[node][component]
                                 - states[node][component])
                                * transcription_config_.state_trust_scales[component]
                            )
                        );
                    }
                }
            }
            const auto propellant = std::max(
                0.0,
                scenarios[scenario].initial[6U] - states.back()[6U]
            );
            const auto delta_v = robust_low_thrust_detail::integrated_delta_v(
                states,
                controls,
                scenarios[scenario].model,
                transcription_config_.step_seconds
            );
            result.propellant.push_back(propellant);
            result.delta_v.push_back(delta_v);
            result.scenario_merits.push_back(
                normalised_fuel(controls, scenarios[scenario].model)
                + config_.outer.feasibility_penalty
                      * (terminal + path_violation(path, scenarios[scenario].model))
            );
        }
        const auto scenario_summary = distributed::aggregate_scenario_risk(
            result.scenario_merits,
            probabilities,
            config_.risk_confidence
        );
        result.propellant_risk = distributed::aggregate_scenario_risk(
            result.propellant,
            probabilities,
            config_.risk_confidence
        );
        result.delta_v_risk = distributed::aggregate_scenario_risk(
            result.delta_v,
            probabilities,
            config_.risk_confidence
        );
        result.merit = scenario_summary.expected
                       + config_.risk_weight
                             * robust_low_thrust_detail::selected_risk(
                                 result.propellant_risk,
                                 config_.risk_measure
                             );
        result.residual = OuterResidual{
            maximum_dynamics,
            maximum_path,
            maximum_terminal,
            step_fraction_value,
        };
        return result;
    }

    ReferenceEvaluation invalid_evaluation(const double step) const {
        const auto infinity = std::numeric_limits<double>::infinity();
        ReferenceEvaluation result{};
        result.merit = infinity;
        result.residual = OuterResidual{infinity, infinity, infinity, step};
        return result;
    }

    double model_merit(
        const std::vector<LowThrustDecodedDecision>& decisions,
        const std::vector<RobustLowThrustScenario>& scenarios,
        const std::vector<double>& probabilities
    ) const {
        std::vector<double> scenario_merits{};
        std::vector<double> propellant{};
        scenario_merits.reserve(scenarios.size());
        propellant.reserve(scenarios.size());
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            const auto path = scenarios[scenario].model.path_diagnostics(
                decisions[scenario].states,
                decisions[scenario].controls
            );
            double virtual_measure{0.0};
            for (const auto& virtual_control : decisions[scenario].virtual_controls) {
                for (std::size_t component = 0; component < 7U; ++component) {
                    virtual_measure += std::abs(
                        virtual_control[component]
                        * transcription_config_.state_trust_scales[component]
                    );
                }
            }
            virtual_measure /= static_cast<double>(
                decisions[scenario].virtual_controls.size() * 7U
            );
            const auto terminal = terminal_error(
                decisions[scenario].states.back(),
                scenarios[scenario].target
            );
            scenario_merits.push_back(
                normalised_fuel(
                    decisions[scenario].controls,
                    scenarios[scenario].model
                )
                + config_.outer.feasibility_penalty
                      * (terminal + path_violation(path, scenarios[scenario].model))
                + config_.outer.virtual_penalty * virtual_measure
            );
            propellant.push_back(
                propellant_from_controls(
                    decisions[scenario].controls,
                    scenarios[scenario].model
                )
            );
        }
        const auto merit_summary = distributed::aggregate_scenario_risk(
            scenario_merits,
            probabilities,
            config_.risk_confidence
        );
        const auto propellant_summary = distributed::aggregate_scenario_risk(
            propellant,
            probabilities,
            config_.risk_confidence
        );
        return merit_summary.expected
               + config_.risk_weight
                     * robust_low_thrust_detail::selected_risk(
                         propellant_summary,
                         config_.risk_measure
                     );
    }

    double maximum_step_fraction(
        const std::vector<LowThrustDecodedDecision>& decisions,
        const RobustLowThrustReferences& references,
        const double radius
    ) const {
        double maximum{0.0};
        for (std::size_t scenario = 0; scenario < decisions.size(); ++scenario) {
            for (std::size_t interval = 0;
                 interval < transcription_config_.intervals;
                 ++interval) {
                double squared{0.0};
                for (std::size_t component = 0; component < 7U; ++component) {
                    const auto value =
                        (decisions[scenario].states[interval][component]
                         - references[scenario].first[interval][component])
                        * transcription_config_.state_trust_scales[component];
                    squared += value * value;
                }
                for (std::size_t component = 0; component < 4U; ++component) {
                    const auto value =
                        (decisions[scenario].controls[interval][component]
                         - references[scenario].second[interval][component])
                        * transcription_config_.control_trust_scales[component];
                    squared += value * value;
                }
                maximum = std::max(maximum, std::sqrt(squared) / radius);
            }
        }
        return maximum;
    }

    double terminal_error(
        const dynamics::LowThrustState& actual,
        const dynamics::LowThrustState& target
    ) const noexcept {
        double maximum{0.0};
        for (std::size_t component = 0; component < 6U; ++component) {
            maximum = std::max(
                maximum,
                std::abs(
                    (actual[component] - target[component])
                    * transcription_config_.state_trust_scales[component]
                )
            );
        }
        return maximum;
    }

    double path_violation(
        const dynamics::LowThrustPathDiagnostics& diagnostics,
        const dynamics::LowThrustTwoBodyModel& model
    ) const noexcept {
        const auto position_scale = std::max(
            {transcription_config_.state_trust_scales[0U],
             transcription_config_.state_trust_scales[1U],
             transcription_config_.state_trust_scales[2U]}
        );
        return std::max(
            {diagnostics.thrust_epigraph / model.config().maximum_thrust,
             diagnostics.throttle_upper / model.config().maximum_thrust,
             diagnostics.minimum_mass * transcription_config_.state_trust_scales[6U],
             diagnostics.minimum_radius * position_scale}
        );
    }

    double normalised_fuel(
        const std::vector<dynamics::LowThrustControl>& controls,
        const dynamics::LowThrustTwoBodyModel& model
    ) const noexcept {
        double total{0.0};
        for (const auto& control : controls) {
            total += control[3U];
        }
        return controls.empty()
                   ? 0.0
                   : total
                         / (static_cast<double>(controls.size())
                            * model.config().maximum_thrust);
    }

    double propellant_from_controls(
        const std::vector<dynamics::LowThrustControl>& controls,
        const dynamics::LowThrustTwoBodyModel& model
    ) const noexcept {
        double total{0.0};
        for (const auto& control : controls) {
            total += model.config().mass_flow_coefficient
                     * transcription_config_.step_seconds * control[3U];
        }
        return total;
    }

    RobustLowThrustScvxResult make_result(
        const NativeLowThrustStatus status,
        const RobustLowThrustReferences& references,
        const std::vector<RobustLowThrustScenario>& scenarios,
        const std::vector<double>& probabilities,
        const double merit,
        const OuterResidual residual,
        std::vector<RobustLowThrustIterationRecord> records,
        const std::size_t accepted_iterations,
        const std::size_t backend_creations,
        const std::size_t backend_updates
    ) const {
        RobustLowThrustScvxResult result{};
        result.status = status;
        result.merit = merit;
        result.residual = residual;
        result.iterations = std::move(records);
        result.accepted_iterations = accepted_iterations;
        result.backend_creations = backend_creations;
        result.backend_updates = backend_updates;
        std::vector<double> propellant{};
        std::vector<double> delta_v{};
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            const auto path = scenarios[scenario].model.path_diagnostics(
                references[scenario].first,
                references[scenario].second
            );
            const auto scenario_propellant = std::max(
                0.0,
                scenarios[scenario].initial[6U]
                    - references[scenario].first.back()[6U]
            );
            const auto scenario_delta_v = robust_low_thrust_detail::integrated_delta_v(
                references[scenario].first,
                references[scenario].second,
                scenarios[scenario].model,
                transcription_config_.step_seconds
            );
            propellant.push_back(scenario_propellant);
            delta_v.push_back(scenario_delta_v);
            result.scenarios.push_back(RobustLowThrustScenarioResult{
                scenarios[scenario].name,
                references[scenario].first,
                references[scenario].second,
                path,
                terminal_error(
                    references[scenario].first.back(),
                    scenarios[scenario].target
                ),
                scenario_propellant,
                scenario_delta_v,
            });
        }
        result.propellant_risk = distributed::aggregate_scenario_risk(
            propellant,
            probabilities,
            config_.risk_confidence
        );
        result.delta_v_risk = distributed::aggregate_scenario_risk(
            delta_v,
            probabilities,
            config_.risk_confidence
        );
        result.selected_propellant_risk = robust_low_thrust_detail::selected_risk(
            result.propellant_risk,
            config_.risk_measure
        );
        result.selected_delta_v_risk = robust_low_thrust_detail::selected_risk(
            result.delta_v_risk,
            config_.risk_measure
        );
        return result;
    }
};

}  // namespace spacepdhcg::scvx
