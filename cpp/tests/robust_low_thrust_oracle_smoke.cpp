#include "spacepdhcg/orbitweaver/robust_low_thrust_oracle.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct Counters {
    std::size_t creations{0U};
    std::size_t warm_starts{0U};
    std::size_t solves{0U};
};

class WarmEchoBackend final : public spacepdhcg::core::HostPersistentBackend {
  public:
    WarmEchoBackend(
        spacepdhcg::core::FixedCQP problem,
        std::shared_ptr<Counters> counters
    )
        : structure_(problem.structure()),
          values_(problem.values()),
          counters_(std::move(counters)) {
        ++counters_->creations;
    }

    [[nodiscard]] const spacepdhcg::core::FixedStructure& structure() const noexcept override {
        return structure_;
    }

    [[nodiscard]] std::size_t update_count() const noexcept override {
        return update_count_;
    }

    void update(spacepdhcg::core::NumericValues values) override {
        values.validate(structure_);
        values_ = std::move(values);
        ++update_count_;
    }

    void warm_start(const spacepdhcg::core::HostWarmStart& start) override {
        if (start.primal.size() != static_cast<std::size_t>(structure_.variables())
            || start.dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("robust oracle warm start has the wrong size");
        }
        primal_ = start.primal;
        dual_ = start.dual;
        ++counters_->warm_starts;
    }

    [[nodiscard]] spacepdhcg::core::HostCqpSolution solve(
        const double tolerance,
        const std::size_t iteration_limit
    ) override {
        if (!std::isfinite(tolerance) || tolerance <= 0.0 || iteration_limit == 0U
            || primal_.empty()) {
            throw std::invalid_argument("robust oracle solve request is invalid");
        }
        ++counters_->solves;
        spacepdhcg::core::HostCqpSolution result{};
        result.status = spacepdhcg::SolveStatus::optimal;
        result.primal = primal_;
        result.dual = dual_;
        result.primal_residual = 1.0e-10;
        result.dual_residual = 1.0e-10;
        result.outer_iterations = 1U;
        result.inner_iterations = 2U;
        return result;
    }

  private:
    spacepdhcg::core::FixedStructure structure_{};
    spacepdhcg::core::NumericValues values_{};
    std::shared_ptr<Counters> counters_{};
    std::vector<double> primal_{};
    std::vector<double> dual_{};
    std::size_t update_count_{0U};
};

}  // namespace

int main() {
    using spacepdhcg::distributed::RiskMeasure;
    using spacepdhcg::dynamics::LowThrustControl;
    using spacepdhcg::dynamics::LowThrustState;
    using spacepdhcg::dynamics::LowThrustTwoBodyConfig;
    using spacepdhcg::dynamics::LowThrustTwoBodyModel;
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::RobustLowThrustArcOracleConfig;
    using spacepdhcg::orbitweaver::RobustLowThrustOrbitStage;
    using spacepdhcg::scvx::RobustLowThrustScenario;

    LowThrustTwoBodyConfig nominal_config{};
    nominal_config.gravitational_parameter = 1.0e-12;
    nominal_config.thrust_to_acceleration = 1.0;
    nominal_config.mass_flow_coefficient = 1.0e-6;
    nominal_config.minimum_mass = 100.0;
    nominal_config.maximum_thrust = 1.0;
    nominal_config.minimum_radius = 1.0;
    auto uncertain_config = nominal_config;
    uncertain_config.mass_flow_coefficient = 2.0e-6;
    const LowThrustState initial{
        7'000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        500.0,
    };
    const auto scenario_provider = [nominal_config, uncertain_config, initial](
                                       const ArcRequest& request
                                   ) {
        const auto duration = *request.arrival_epoch - request.departure_epoch;
        const auto step = duration / 3.0;
        const LowThrustControl coast{0.0, 0.0, 0.0, 0.0};
        const std::vector<LowThrustControl> controls(3U, coast);
        const LowThrustTwoBodyModel nominal{nominal_config};
        const LowThrustTwoBodyModel uncertain{uncertain_config};
        const auto nominal_states = nominal.rollout(initial, controls, step, false);
        const auto uncertain_states = uncertain.rollout(initial, controls, step, false);
        return std::vector<RobustLowThrustScenario>{
            RobustLowThrustScenario{
                "nominal",
                0.6,
                nominal,
                initial,
                nominal_states.back(),
            },
            RobustLowThrustScenario{
                "high-mass-flow",
                0.4,
                uncertain,
                initial,
                uncertain_states.back(),
            },
        };
    };
    const auto counters = std::make_shared<Counters>();
    const auto backend_factory = [counters](spacepdhcg::core::FixedCQP problem) {
        return std::make_unique<WarmEchoBackend>(
            std::move(problem),
            counters
        );
    };

    RobustLowThrustArcOracleConfig config{};
    config.transcription.intervals = 3U;
    config.transcription.step_seconds = 10.0;
    config.robust.common_prefix = 3U;
    config.robust.risk_measure = RiskMeasure::conditional_value_at_risk;
    config.robust.risk_confidence = 0.8;
    config.robust.risk_weight = 2.0;
    config.robust.outer.maximum_iterations = 1U;
    config.robust.outer.minimum_iterations = 1U;
    config.feasibility_tolerance = 1.0e-6;
    config.cost_per_delta_v = 1.0;
    config.cost_per_propellant = 1.0;

    const RobustLowThrustOrbitStage stage{
        scenario_provider,
        backend_factory,
        config,
    };
    const ArcRequest request{
        0U,
        1U,
        0.0,
        30.0,
        500.0,
        0U,
        2U,
        ArcFidelity::robust_scvx,
        1.0e-6,
        "robust-low-thrust-smoke",
        std::nullopt,
    };
    const auto result = stage.evaluate(request);
    if (!result.feasible || result.achieved_fidelity != ArcFidelity::robust_scvx
        || result.maximum_constraint_violation > 1.0e-6
        || result.terminal_error > 1.0e-6) {
        return 1;
    }
    if (result.propellant < 0.0 || result.delta_v < 0.0 || result.cost < 0.0
        || std::abs(result.final_mass + result.propellant - request.initial_mass)
               > 1.0e-10) {
        return 2;
    }
    if (result.diagnostics.find("CVaR") == std::string::npos
        || result.diagnostics.find("scenarios=2") == std::string::npos) {
        return 3;
    }
    if (counters->creations != 1U || counters->warm_starts != 1U
        || counters->solves != 1U) {
        return 4;
    }
    return 0;
}
