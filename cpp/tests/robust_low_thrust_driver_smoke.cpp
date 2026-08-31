#include "spacepdhcg/scvx/robust_low_thrust_driver.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct Counters {
    std::size_t creations{0U};
    std::size_t updates{0U};
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
        ++counters_->updates;
    }

    void warm_start(const spacepdhcg::core::HostWarmStart& start) override {
        if (!start.primal.empty()
            && start.primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("robust echo warm primal has the wrong size");
        }
        if (!start.dual.empty()
            && start.dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("robust echo warm dual has the wrong size");
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
            throw std::invalid_argument("robust echo solve request is invalid");
        }
        ++counters_->solves;
        spacepdhcg::core::HostCqpSolution solution{};
        solution.status = spacepdhcg::SolveStatus::optimal;
        solution.primal = primal_;
        solution.dual = dual_;
        solution.primal_residual = 1.0e-10;
        solution.dual_residual = 1.0e-10;
        solution.outer_iterations = 1U;
        solution.inner_iterations = 2U;
        return solution;
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
    using spacepdhcg::scvx::NativeLowThrustStatus;
    using spacepdhcg::scvx::RobustLowThrustConfig;
    using spacepdhcg::scvx::RobustLowThrustReferences;
    using spacepdhcg::scvx::RobustLowThrustScenario;
    using spacepdhcg::scvx::RobustLowThrustScvxDriver;
    using spacepdhcg::transcription::DiscretisationMethod;
    using spacepdhcg::transcription::LowThrustScvxConfig;

    LowThrustTwoBodyConfig first_config{};
    first_config.gravitational_parameter = 1.0e-12;
    first_config.thrust_to_acceleration = 1.0;
    first_config.mass_flow_coefficient = 1.0e-6;
    first_config.minimum_mass = 100.0;
    first_config.maximum_thrust = 1.0;
    first_config.minimum_radius = 1.0;
    auto second_config = first_config;
    second_config.mass_flow_coefficient = 2.0e-6;
    const LowThrustTwoBodyModel first_model{first_config};
    const LowThrustTwoBodyModel second_model{second_config};

    LowThrustScvxConfig transcription{};
    transcription.intervals = 3U;
    transcription.step_seconds = 10.0;
    transcription.discretisation = DiscretisationMethod::euler;
    transcription.trust_radius = 1.0;
    const LowThrustState initial{
        7'000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        500.0,
    };
    const LowThrustControl coast{0.0, 0.0, 0.0, 0.0};
    const std::vector<LowThrustControl> controls(transcription.intervals, coast);
    const auto first_states = first_model.rollout(
        initial,
        controls,
        transcription.step_seconds,
        false
    );
    const auto second_states = second_model.rollout(
        initial,
        controls,
        transcription.step_seconds,
        false
    );
    std::vector<RobustLowThrustScenario> scenarios{
        RobustLowThrustScenario{
            "nominal",
            0.6,
            first_model,
            initial,
            first_states.back(),
        },
        RobustLowThrustScenario{
            "high-mass-flow",
            0.4,
            second_model,
            initial,
            second_states.back(),
        },
    };
    RobustLowThrustReferences references{
        {first_states, controls},
        {second_states, controls},
    };

    RobustLowThrustConfig config{};
    config.common_prefix = transcription.intervals;
    config.risk_measure = RiskMeasure::conditional_value_at_risk;
    config.risk_confidence = 0.8;
    config.risk_weight = 2.0;
    config.outer.maximum_iterations = 1U;
    config.outer.minimum_iterations = 1U;
    const auto counters = std::make_shared<Counters>();
    const auto factory = [counters](spacepdhcg::core::FixedCQP problem) {
        return std::make_unique<WarmEchoBackend>(
            std::move(problem),
            counters
        );
    };
    const RobustLowThrustScvxDriver driver{
        transcription,
        factory,
        config,
    };
    const auto result = driver.solve(scenarios, references);

    if (result.status != NativeLowThrustStatus::maximum_iterations
        || result.scenarios.size() != 2U || result.iterations.size() != 1U
        || result.backend_creations != 1U || result.backend_updates != 0U) {
        return 1;
    }
    if (counters->creations != 1U || counters->updates != 0U
        || counters->warm_starts != 1U || counters->solves != 1U) {
        return 2;
    }
    if (result.iterations.front().nonanticipativity_violation > 1.0e-12
        || result.iterations.front().risk_epigraph_violation > 1.0e-12
        || result.residual.maximum() > 1.0e-12) {
        return 3;
    }
    if (result.selected_propellant_risk > 1.0e-12
        || result.selected_delta_v_risk > 1.0e-12
        || result.propellant_risk.worst > 1.0e-12) {
        return 4;
    }
    for (std::size_t interval = 0; interval < transcription.intervals; ++interval) {
        for (std::size_t component = 0; component < 4U; ++component) {
            if (std::abs(
                    result.scenarios[0U].controls[interval][component]
                    - result.scenarios[1U].controls[interval][component]
                ) > 1.0e-12) {
                return 5;
            }
        }
    }
    return 0;
}
