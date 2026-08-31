#pragma once

#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"
#include "spacepdhcg/scvx/robust_low_thrust_driver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

using RobustLowThrustScenarioProvider = std::function<
    std::vector<scvx::RobustLowThrustScenario>(const ArcRequest&)>;

struct RobustLowThrustArcOracleConfig {
    transcription::LowThrustScvxConfig transcription{};
    scvx::RobustLowThrustConfig robust{};
    double cost_per_delta_v{1.0};
    double cost_per_propellant{0.0};
    double cost_per_second{0.0};
    double feasibility_tolerance{5.0e-3};

    void validate() const {
        transcription.validate();
        robust.validate(transcription.intervals);
        for (const auto value : {
                 cost_per_delta_v,
                 cost_per_propellant,
                 cost_per_second,
                 feasibility_tolerance,
             }) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "robust low-thrust arc costs and tolerance must be finite and non-negative"
                );
            }
        }
        if (feasibility_tolerance <= 0.0) {
            throw std::invalid_argument(
                "robust low-thrust arc feasibility tolerance must be positive"
            );
        }
    }
};

namespace robust_low_thrust_oracle_detail {

inline std::string risk_measure_name(const distributed::RiskMeasure measure) {
    switch (measure) {
        case distributed::RiskMeasure::expected:
            return "expected";
        case distributed::RiskMeasure::worst_case:
            return "worst-case";
        case distributed::RiskMeasure::conditional_value_at_risk:
            return "CVaR";
    }
    return "unknown";
}

inline ArcSolution infeasible(std::string diagnostics) {
    ArcSolution result{};
    result.achieved_fidelity = ArcFidelity::robust_scvx;
    result.diagnostics = std::move(diagnostics);
    return result;
}

}  // namespace robust_low_thrust_oracle_detail

/// Concrete host truth stage for OrbitWeaver `robust_scvx` arcs.
///
/// The scenario provider owns uncertainty semantics (navigation, gravity, propulsion or
/// initial-state variants). This stage enforces request/scenario consistency, derives the
/// fixed interval duration from the requested arc, runs the common-open-loop robust SCvx
/// driver, and converts the selected expected/worst/CVaR statistics into a physically closed
/// ArcSolution certificate. Production CUDA/NCCL execution can replace only the backend;
/// request, risk and acceptance semantics remain unchanged.
class RobustLowThrustOrbitStage {
  public:
    RobustLowThrustOrbitStage(
        RobustLowThrustScenarioProvider scenario_provider,
        scvx::LowThrustHostBackendFactory backend_factory,
        RobustLowThrustArcOracleConfig config = {}
    )
        : scenario_provider_(std::move(scenario_provider)),
          backend_factory_(std::move(backend_factory)),
          config_(config) {
        if (!scenario_provider_ || !backend_factory_) {
            throw std::invalid_argument(
                "robust low-thrust OrbitWeaver stage requires scenario and backend providers"
            );
        }
        config_.validate();
    }

    [[nodiscard]] FidelityPipelineOracle::Stage stage() const {
        return [*this](
                   const ArcRequest& request,
                   const std::optional<ArcSolution>& previous
               ) { return evaluate(request, previous); };
    }

    void register_stage(FidelityPipelineOracle& pipeline) const {
        pipeline.register_stage(ArcFidelity::robust_scvx, stage());
    }

    [[nodiscard]] ArcSolution evaluate(
        const ArcRequest& request,
        const std::optional<ArcSolution>& previous = std::nullopt
    ) const {
        request.validate();
        if (request.fidelity != ArcFidelity::robust_scvx) {
            throw std::invalid_argument(
                "robust low-thrust stage received the wrong fidelity"
            );
        }
        if (!request.arrival_epoch.has_value()) {
            throw std::invalid_argument(
                "robust low-thrust stage requires an arrival epoch"
            );
        }
        auto scenarios = scenario_provider_(request);
        if (scenarios.size() != request.scenario_count) {
            throw std::invalid_argument(
                "robust low-thrust scenario provider returned the wrong scenario count"
            );
        }
        const auto duration = *request.arrival_epoch - request.departure_epoch;
        auto transcription_config = config_.transcription;
        transcription_config.step_seconds =
            duration / static_cast<double>(transcription_config.intervals);
        transcription_config.validate();
        for (const auto& scenario : scenarios) {
            scenario.validate();
            if (std::abs(scenario.initial[6U] - request.initial_mass)
                > 1.0e-10 * std::max(1.0, request.initial_mass)) {
                throw std::invalid_argument(
                    "robust low-thrust scenario initial mass disagrees with the arc request"
                );
            }
        }

        scvx::RobustLowThrustScvxDriver driver{
            transcription_config,
            backend_factory_,
            config_.robust,
        };
        const auto solve = driver.solve(std::move(scenarios));
        if (solve.status == scvx::NativeLowThrustStatus::solver_failed
            || solve.scenarios.empty()) {
            return robust_low_thrust_oracle_detail::infeasible(
                "native robust low-thrust SCvx solver failed"
            );
        }

        double maximum_terminal{0.0};
        double maximum_path{0.0};
        double maximum_inner_residual{0.0};
        std::size_t inner_iterations{0U};
        double setup_seconds{0.0};
        double solve_seconds{0.0};
        for (const auto& scenario : solve.scenarios) {
            maximum_terminal = std::max(maximum_terminal, scenario.terminal_error);
            maximum_path = std::max(
                maximum_path,
                scenario.path.maximum_violation()
            );
        }
        for (std::size_t index = 0; index < solve.iterations.size(); ++index) {
            const auto& record = solve.iterations[index];
            maximum_inner_residual = std::max(
                {maximum_inner_residual,
                 record.primal_residual,
                 record.dual_residual,
                 record.nonanticipativity_violation,
                 record.risk_epigraph_violation}
            );
            inner_iterations += record.solver_iterations;
        }
        const auto achieved = std::max(
            {solve.residual.feasibility(),
             maximum_terminal,
             maximum_path,
             maximum_inner_residual,
             std::numeric_limits<double>::epsilon()}
        );
        const auto acceptance = std::max(
            request.requested_tolerance,
            config_.feasibility_tolerance
        );
        if (maximum_terminal > acceptance || maximum_path > acceptance
            || solve.residual.feasibility() > acceptance) {
            return robust_low_thrust_oracle_detail::infeasible(
                "native robust low-thrust arc failed nonlinear scenario certification"
            );
        }

        const auto propellant = solve.selected_propellant_risk;
        if (!std::isfinite(propellant) || propellant < 0.0
            || propellant >= request.initial_mass) {
            return robust_low_thrust_oracle_detail::infeasible(
                "robust low-thrust selected propellant risk is physically invalid"
            );
        }
        const auto delta_v = solve.selected_delta_v_risk;
        const auto cost = config_.cost_per_delta_v * delta_v
                          + config_.cost_per_propellant * propellant
                          + config_.cost_per_second * duration;
        const auto lower_bound = previous.has_value()
                                     ? std::min(cost, previous->lower_bound)
                                     : 0.0;
        ArcSolution result{
            true,
            ArcFidelity::robust_scvx,
            cost,
            lower_bound,
            duration,
            delta_v,
            propellant,
            request.initial_mass - propellant,
            maximum_terminal,
            maximum_path,
            achieved,
            solve.iterations.size(),
            inner_iterations,
            setup_seconds,
            solve_seconds,
            std::nullopt,
            std::string{"native robust low-thrust "}
                + robust_low_thrust_oracle_detail::risk_measure_name(
                    config_.robust.risk_measure
                )
                + ", scenarios=" + std::to_string(request.scenario_count)
                + ", status="
                + std::string{scvx::native_low_thrust_status_name(solve.status)},
        };
        result.validate(request);
        return result;
    }

    [[nodiscard]] const RobustLowThrustArcOracleConfig& config() const noexcept {
        return config_;
    }

  private:
    RobustLowThrustScenarioProvider scenario_provider_{};
    scvx::LowThrustHostBackendFactory backend_factory_{};
    RobustLowThrustArcOracleConfig config_{};
};

}  // namespace spacepdhcg::orbitweaver
