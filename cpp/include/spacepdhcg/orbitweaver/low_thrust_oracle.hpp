#pragma once

#include "spacepdhcg/core/host_backend.hpp"
#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/orbitweaver/lambert_oracle.hpp"
#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"
#include "spacepdhcg/scvx/low_thrust_driver.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct LowThrustArcOracleConfig {
    dynamics::LowThrustTwoBodyConfig dynamics{};
    transcription::LowThrustScvxConfig transcription{};
    scvx::NativeLowThrustOuterConfig outer{};
    scvx::ForcingRuleConfig forcing{};
    scvx::TrustRegionConfig trust{};
    double cost_per_delta_v{1.0};
    double cost_per_second{0.0};
    double coarse_feasibility_tolerance{2.5e-1};
    double refined_feasibility_tolerance{5.0e-3};
    std::size_t coarse_iteration_limit{250'000U};

    void validate() const {
        dynamics.validate();
        transcription.validate();
        outer.validate();
        forcing.validate();
        trust.validate();
        for (const auto value : {
                 cost_per_delta_v,
                 cost_per_second,
                 coarse_feasibility_tolerance,
                 refined_feasibility_tolerance,
             }) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "low-thrust arc cost and feasibility values must be finite and non-negative"
                );
            }
        }
        if (coarse_feasibility_tolerance <= 0.0
            || refined_feasibility_tolerance <= 0.0
            || coarse_iteration_limit == 0U) {
            throw std::invalid_argument(
                "low-thrust arc tolerances and iteration limit must be positive"
            );
        }
    }
};

struct LowThrustWarmStartRecord {
    std::size_t from_target{0U};
    std::size_t to_target{0U};
    double departure_epoch{0.0};
    double arrival_epoch{0.0};
    double initial_mass{0.0};
    std::size_t spacecraft{0U};
    std::string model_identifier{};
    std::size_t intervals{0U};
    scvx::LowThrustReference reference{};

    [[nodiscard]] bool matches(
        const ArcRequest& request,
        const std::size_t expected_intervals
    ) const noexcept {
        return request.arrival_epoch.has_value() && from_target == request.from_target
               && to_target == request.to_target
               && departure_epoch == request.departure_epoch
               && arrival_epoch == *request.arrival_epoch
               && initial_mass == request.initial_mass && spacecraft == request.spacecraft
               && model_identifier == request.model_identifier
               && intervals == expected_intervals;
    }
};

/// Thread-safe in-process trajectory store used to transfer native state/control references
/// between OrbitWeaver fidelity stages without serialising large vectors into ArcSolution.
class LowThrustWarmStartStore {
  public:
    explicit LowThrustWarmStartStore(const std::size_t capacity = 1'024U)
        : capacity_(capacity) {
        if (capacity_ == 0U) {
            throw std::invalid_argument("low-thrust warm-start capacity must be positive");
        }
    }

    [[nodiscard]] std::uint64_t put(
        const ArcRequest& request,
        const std::size_t intervals,
        scvx::LowThrustReference reference
    ) {
        request.validate();
        if (!request.arrival_epoch.has_value() || intervals < 2U
            || reference.first.size() != intervals + 1U
            || reference.second.size() != intervals) {
            throw std::invalid_argument("low-thrust warm-start record is incompatible");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto token = next_token_++;
        records_.emplace(
            token,
            LowThrustWarmStartRecord{
                request.from_target,
                request.to_target,
                request.departure_epoch,
                *request.arrival_epoch,
                request.initial_mass,
                request.spacecraft,
                request.model_identifier,
                intervals,
                std::move(reference),
            }
        );
        while (records_.size() > capacity_) {
            records_.erase(records_.begin());
        }
        return token;
    }

    [[nodiscard]] std::optional<scvx::LowThrustReference> get(
        const std::uint64_t token,
        const ArcRequest& request,
        const std::size_t intervals
    ) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iterator = records_.find(token);
        if (iterator == records_.end() || !iterator->second.matches(request, intervals)) {
            return std::nullopt;
        }
        return iterator->second.reference;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.size();
    }

    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
    }

  private:
    std::size_t capacity_{1'024U};
    mutable std::mutex mutex_{};
    std::map<std::uint64_t, LowThrustWarmStartRecord> records_{};
    std::uint64_t next_token_{1U};
};

namespace low_thrust_oracle_detail {

struct ArcContext {
    transcription::LowThrustSubproblem subproblem;
    dynamics::LowThrustState initial{};
    dynamics::LowThrustState target{};
    double duration{0.0};
};

inline void validate_ephemeris_state(const CartesianEphemerisState& state) {
    for (const auto value : state.position) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("low-thrust ephemeris position is non-finite");
        }
    }
    for (const auto value : state.velocity) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("low-thrust ephemeris velocity is non-finite");
        }
    }
}

inline ArcContext make_context(
    const ArcRequest& request,
    const EphemerisProvider& ephemeris,
    const LowThrustArcOracleConfig& config
) {
    request.validate();
    if (!request.arrival_epoch.has_value()) {
        throw std::invalid_argument("low-thrust arc stages require an arrival epoch");
    }
    if (request.scenario_count != 1U) {
        throw std::invalid_argument(
            "deterministic low-thrust arc stages require scenario_count equal to one"
        );
    }
    const auto departure = ephemeris(request.from_target, request.departure_epoch);
    const auto arrival = ephemeris(request.to_target, *request.arrival_epoch);
    validate_ephemeris_state(departure);
    validate_ephemeris_state(arrival);
    if (request.initial_mass <= config.dynamics.minimum_mass) {
        throw std::invalid_argument(
            "low-thrust arc initial mass must exceed the configured reserve"
        );
    }
    dynamics::LowThrustState initial{
        departure.position[0U],
        departure.position[1U],
        departure.position[2U],
        departure.velocity[0U],
        departure.velocity[1U],
        departure.velocity[2U],
        request.initial_mass,
    };
    dynamics::LowThrustState target{
        arrival.position[0U],
        arrival.position[1U],
        arrival.position[2U],
        arrival.velocity[0U],
        arrival.velocity[1U],
        arrival.velocity[2U],
        request.initial_mass,
    };
    auto transcription_config = config.transcription;
    const auto duration = *request.arrival_epoch - request.departure_epoch;
    transcription_config.step_seconds =
        duration / static_cast<double>(transcription_config.intervals);
    transcription_config.validate();
    dynamics::LowThrustTwoBodyModel model{config.dynamics};
    static_cast<void>(model.dynamics(
        initial,
        dynamics::LowThrustControl{0.0, 0.0, 0.0, 0.0}
    ));
    static_cast<void>(model.dynamics(
        target,
        dynamics::LowThrustControl{0.0, 0.0, 0.0, 0.0}
    ));
    return ArcContext{
        transcription::LowThrustSubproblem{model, transcription_config},
        initial,
        target,
        duration,
    };
}

inline std::optional<scvx::LowThrustReference> previous_reference(
    const ArcRequest& request,
    const std::optional<ArcSolution>& previous,
    const std::shared_ptr<LowThrustWarmStartStore>& store,
    const std::size_t intervals
) {
    const auto token = request.warm_start_token.has_value()
                           ? request.warm_start_token
                           : previous.has_value() ? previous->warm_start_token : std::nullopt;
    if (!token.has_value()) {
        return std::nullopt;
    }
    return store->get(*token, request, intervals);
}

inline double terminal_error(
    const dynamics::LowThrustState& actual,
    const dynamics::LowThrustState& target,
    const transcription::LowThrustScvxConfig& config
) noexcept {
    double maximum{0.0};
    for (std::size_t component = 0; component < 6U; ++component) {
        maximum = std::max(
            maximum,
            std::abs(
                (actual[component] - target[component])
                * config.state_trust_scales[component]
            )
        );
    }
    return maximum;
}

inline double path_violation(
    const dynamics::LowThrustPathDiagnostics& diagnostics,
    const dynamics::LowThrustTwoBodyModel& model,
    const transcription::LowThrustScvxConfig& config
) noexcept {
    const auto position_scale = std::max(
        {config.state_trust_scales[0U],
         config.state_trust_scales[1U],
         config.state_trust_scales[2U]}
    );
    return std::max(
        {diagnostics.thrust_epigraph / model.config().maximum_thrust,
         diagnostics.throttle_upper / model.config().maximum_thrust,
         diagnostics.minimum_mass * config.state_trust_scales[6U],
         diagnostics.minimum_radius * position_scale}
    );
}

inline double integrated_delta_v(
    const std::vector<dynamics::LowThrustState>& states,
    const std::vector<dynamics::LowThrustControl>& controls,
    const dynamics::LowThrustTwoBodyModel& model,
    const double step_seconds
) {
    if (states.size() != controls.size() + 1U) {
        throw std::invalid_argument("low-thrust delta-v requires N controls and N+1 states");
    }
    double result{0.0};
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        result += step_seconds * model.config().thrust_to_acceleration
                  * controls[interval][3U] / states[interval][6U];
    }
    return result;
}

inline double arc_cost(
    const double delta_v,
    const double duration,
    const LowThrustArcOracleConfig& config
) noexcept {
    return config.cost_per_delta_v * delta_v + config.cost_per_second * duration;
}

inline double inherited_lower_bound(
    const std::optional<ArcSolution>& previous,
    const double cost
) noexcept {
    return previous.has_value() ? std::min(cost, previous->lower_bound) : 0.0;
}

inline ArcSolution infeasible(
    const ArcFidelity fidelity,
    std::string diagnostics
) {
    ArcSolution result{};
    result.achieved_fidelity = fidelity;
    result.diagnostics = std::move(diagnostics);
    return result;
}

}  // namespace low_thrust_oracle_detail

/// Concrete OrbitWeaver fidelity stages backed by the native low-thrust transcription.
///
/// `coarse_stage()` solves one convex subproblem and performs an independent nonlinear
/// rollout. `refined_stage()` runs the persistent native SCvx driver. Both return physically
/// closed mass accounting, integrated low-thrust delta-v, scaled terminal/path errors,
/// solver timing, and a token for the next fidelity stage.
class LowThrustOrbitStages {
  public:
    LowThrustOrbitStages(
        EphemerisProvider ephemeris,
        scvx::LowThrustHostBackendFactory backend_factory,
        LowThrustArcOracleConfig config = {},
        std::shared_ptr<LowThrustWarmStartStore> store =
            std::make_shared<LowThrustWarmStartStore>()
    )
        : ephemeris_(std::move(ephemeris)),
          backend_factory_(std::move(backend_factory)),
          config_(config),
          store_(std::move(store)) {
        if (!ephemeris_ || !backend_factory_ || !store_) {
            throw std::invalid_argument(
                "low-thrust OrbitWeaver stages require ephemeris, backend, and store"
            );
        }
        config_.validate();
    }

    [[nodiscard]] FidelityPipelineOracle::Stage coarse_stage() {
        return [this](
                   const ArcRequest& request,
                   const std::optional<ArcSolution>& previous
               ) { return evaluate_coarse(request, previous); };
    }

    [[nodiscard]] FidelityPipelineOracle::Stage refined_stage() {
        return [this](
                   const ArcRequest& request,
                   const std::optional<ArcSolution>& previous
               ) { return evaluate_refined(request, previous); };
    }

    void register_stages(FidelityPipelineOracle& pipeline) {
        pipeline.register_stage(ArcFidelity::coarse_convex, coarse_stage());
        pipeline.register_stage(ArcFidelity::refined_scvx, refined_stage());
    }

    [[nodiscard]] const LowThrustArcOracleConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] const std::shared_ptr<LowThrustWarmStartStore>& store() const noexcept {
        return store_;
    }

  private:
    EphemerisProvider ephemeris_{};
    scvx::LowThrustHostBackendFactory backend_factory_{};
    LowThrustArcOracleConfig config_{};
    std::shared_ptr<LowThrustWarmStartStore> store_{};

    [[nodiscard]] ArcSolution evaluate_coarse(
        const ArcRequest& request,
        const std::optional<ArcSolution>& previous
    ) {
        if (request.fidelity != ArcFidelity::coarse_convex) {
            throw std::invalid_argument(
                "low-thrust coarse stage received the wrong fidelity"
            );
        }
        auto context = low_thrust_oracle_detail::make_context(
            request,
            ephemeris_,
            config_
        );
        auto reference = low_thrust_oracle_detail::previous_reference(
            request,
            previous,
            store_,
            context.subproblem.layout().intervals
        ).value_or(scvx::make_native_low_thrust_reference(
            context.subproblem.model(),
            context.initial,
            context.target,
            context.subproblem.layout().intervals,
            context.subproblem.config().step_seconds,
            context.subproblem.config().discretisation
                == transcription::DiscretisationMethod::rk4_finite_difference
        ));
        auto problem = context.subproblem.problem(
            reference.first,
            reference.second,
            context.initial,
            context.target
        );
        auto backend = backend_factory_(std::move(problem));
        if (!backend || backend->structure().fingerprint()
                            != context.subproblem.structure().fingerprint()) {
            throw std::runtime_error(
                "low-thrust coarse backend has incompatible topology"
            );
        }
        core::HostWarmStart warm{};
        warm.primal = context.subproblem.reference_decision(
            reference.first,
            reference.second
        );
        backend->warm_start(warm);
        const auto solution = backend->solve(
            request.requested_tolerance,
            config_.coarse_iteration_limit
        );
        if (!solution.solved()
            || solution.primal.size() != context.subproblem.layout().variables()) {
            return low_thrust_oracle_detail::infeasible(
                ArcFidelity::coarse_convex,
                "native low-thrust coarse CQP did not solve"
            );
        }
        const auto decision = scvx::decode_low_thrust_decision(
            context.subproblem,
            solution.primal
        );
        std::vector<dynamics::LowThrustState> rollout{};
        try {
            rollout = context.subproblem.model().rollout(
                context.initial,
                decision.controls,
                context.subproblem.config().step_seconds,
                context.subproblem.config().discretisation
                    == transcription::DiscretisationMethod::rk4_finite_difference
            );
        } catch (const std::exception& error) {
            return low_thrust_oracle_detail::infeasible(
                ArcFidelity::coarse_convex,
                std::string{"native low-thrust coarse rollout failed: "} + error.what()
            );
        }
        const auto path = context.subproblem.model().path_diagnostics(
            rollout,
            decision.controls
        );
        const auto terminal = low_thrust_oracle_detail::terminal_error(
            rollout.back(),
            context.target,
            context.subproblem.config()
        );
        const auto path_error = low_thrust_oracle_detail::path_violation(
            path,
            context.subproblem.model(),
            context.subproblem.config()
        );
        const auto convex = context.subproblem.diagnostics(
            solution.primal,
            context.subproblem.values(
                reference.first,
                reference.second,
                context.initial,
                context.target
            )
        );
        const auto achieved = std::max(
            {terminal,
             path_error,
             convex.maximum_violation(),
             solution.primal_residual,
             solution.dual_residual,
             std::numeric_limits<double>::epsilon()}
        );
        if (terminal > config_.coarse_feasibility_tolerance
            || path_error > config_.coarse_feasibility_tolerance) {
            return low_thrust_oracle_detail::infeasible(
                ArcFidelity::coarse_convex,
                "native low-thrust coarse arc failed nonlinear feasibility"
            );
        }
        const auto final_mass = rollout.back()[6U];
        const auto propellant = std::max(0.0, request.initial_mass - final_mass);
        const auto delta_v = low_thrust_oracle_detail::integrated_delta_v(
            rollout,
            decision.controls,
            context.subproblem.model(),
            context.subproblem.config().step_seconds
        );
        const auto cost = low_thrust_oracle_detail::arc_cost(
            delta_v,
            context.duration,
            config_
        );
        const auto token = store_->put(
            request,
            context.subproblem.layout().intervals,
            scvx::LowThrustReference{rollout, decision.controls}
        );
        ArcSolution result{
            true,
            ArcFidelity::coarse_convex,
            cost,
            low_thrust_oracle_detail::inherited_lower_bound(previous, cost),
            context.duration,
            delta_v,
            propellant,
            final_mass,
            terminal,
            std::max(path_error, convex.maximum_violation()),
            achieved,
            solution.outer_iterations,
            solution.inner_iterations,
            solution.setup_seconds + solution.update_seconds,
            solution.solve_seconds,
            token,
            "native low-thrust coarse convex arc with nonlinear rollout",
        };
        result.validate(request);
        return result;
    }

    [[nodiscard]] ArcSolution evaluate_refined(
        const ArcRequest& request,
        const std::optional<ArcSolution>& previous
    ) {
        if (request.fidelity != ArcFidelity::refined_scvx) {
            throw std::invalid_argument(
                "low-thrust refined stage received the wrong fidelity"
            );
        }
        auto context = low_thrust_oracle_detail::make_context(
            request,
            ephemeris_,
            config_
        );
        auto reference = low_thrust_oracle_detail::previous_reference(
            request,
            previous,
            store_,
            context.subproblem.layout().intervals
        );
        auto outer = config_.outer;
        outer.convergence_tolerance = request.requested_tolerance;
        scvx::NativeLowThrustScvxDriver driver{
            context.subproblem,
            backend_factory_,
            outer,
            config_.forcing,
            config_.trust,
        };
        const auto solve = driver.solve(
            context.initial,
            context.target,
            std::move(reference)
        );
        if (solve.status == scvx::NativeLowThrustStatus::solver_failed
            || solve.states.empty() || solve.controls.empty()) {
            return low_thrust_oracle_detail::infeasible(
                ArcFidelity::refined_scvx,
                "native low-thrust SCvx solver failed"
            );
        }
        const auto terminal = low_thrust_oracle_detail::terminal_error(
            solve.states.back(),
            context.target,
            context.subproblem.config()
        );
        const auto path_error = low_thrust_oracle_detail::path_violation(
            solve.path_diagnostics,
            context.subproblem.model(),
            context.subproblem.config()
        );
        double inner_residual{0.0};
        std::size_t inner_iterations{0U};
        double setup_seconds{0.0};
        double solve_seconds{0.0};
        for (std::size_t index = 0; index < solve.iterations.size(); ++index) {
            const auto& record = solve.iterations[index];
            inner_residual = std::max(
                {inner_residual, record.primal_residual, record.dual_residual}
            );
            inner_iterations += record.solver_iterations;
            if (index == 0U) {
                setup_seconds += record.setup_seconds;
            }
            setup_seconds += record.update_seconds;
            solve_seconds += record.solve_seconds;
        }
        const auto achieved = std::max(
            {solve.residual.feasibility(),
             terminal,
             path_error,
             inner_residual,
             std::numeric_limits<double>::epsilon()}
        );
        const auto acceptance = std::max(
            request.requested_tolerance,
            config_.refined_feasibility_tolerance
        );
        if (terminal > acceptance || path_error > acceptance) {
            return low_thrust_oracle_detail::infeasible(
                ArcFidelity::refined_scvx,
                "native low-thrust refined arc failed nonlinear feasibility"
            );
        }
        const auto final_mass = solve.states.back()[6U];
        const auto propellant = std::max(0.0, request.initial_mass - final_mass);
        const auto delta_v = low_thrust_oracle_detail::integrated_delta_v(
            solve.states,
            solve.controls,
            context.subproblem.model(),
            context.subproblem.config().step_seconds
        );
        const auto cost = low_thrust_oracle_detail::arc_cost(
            delta_v,
            context.duration,
            config_
        );
        const auto token = store_->put(
            request,
            context.subproblem.layout().intervals,
            scvx::LowThrustReference{solve.states, solve.controls}
        );
        ArcSolution result{
            true,
            ArcFidelity::refined_scvx,
            cost,
            low_thrust_oracle_detail::inherited_lower_bound(previous, cost),
            context.duration,
            delta_v,
            propellant,
            final_mass,
            terminal,
            path_error,
            achieved,
            solve.iterations.size(),
            inner_iterations,
            setup_seconds,
            solve_seconds,
            token,
            std::string{"native low-thrust SCvx: "}
                + std::string{scvx::native_low_thrust_status_name(solve.status)},
        };
        result.validate(request);
        return result;
    }
};

}  // namespace spacepdhcg::orbitweaver
