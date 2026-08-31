#include "spacepdhcg/scvx/low_thrust_driver.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct BackendCounters {
    std::size_t creations{0U};
    std::size_t updates{0U};
    std::size_t warm_starts{0U};
    std::size_t solves{0U};
};

class EchoPersistentBackend final : public spacepdhcg::core::HostPersistentBackend {
  public:
    EchoPersistentBackend(
        spacepdhcg::core::FixedCQP problem,
        std::vector<double> primal,
        std::shared_ptr<BackendCounters> counters
    )
        : structure_(problem.structure()),
          values_(problem.values()),
          primal_(std::move(primal)),
          dual_(static_cast<std::size_t>(structure_.duals()), 0.0),
          counters_(std::move(counters)) {
        if (primal_.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("echo low-thrust primal has the wrong size");
        }
        ++counters_->creations;
    }

    [[nodiscard]] const spacepdhcg::core::FixedStructure& structure() const noexcept override {
        return structure_;
    }

    [[nodiscard]] std::size_t update_count() const noexcept override { return update_count_; }

    void update(spacepdhcg::core::NumericValues values) override {
        values.validate(structure_);
        values_ = std::move(values);
        ++update_count_;
        ++counters_->updates;
    }

    void warm_start(const spacepdhcg::core::HostWarmStart& start) override {
        if (!start.primal.empty()
            && start.primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("echo low-thrust warm primal has the wrong size");
        }
        if (!start.dual.empty()
            && start.dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("echo low-thrust warm dual has the wrong size");
        }
        ++counters_->warm_starts;
    }

    [[nodiscard]] spacepdhcg::core::HostCqpSolution solve(
        const double tolerance,
        const std::size_t iteration_limit
    ) override {
        if (!std::isfinite(tolerance) || tolerance <= 0.0 || iteration_limit == 0U) {
            throw std::invalid_argument("echo low-thrust solve request is invalid");
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
    std::vector<double> primal_{};
    std::vector<double> dual_{};
    std::shared_ptr<BackendCounters> counters_{};
    std::size_t update_count_{0U};
};

}  // namespace

int main() {
    using spacepdhcg::dynamics::LowThrustControl;
    using spacepdhcg::dynamics::LowThrustState;
    using spacepdhcg::dynamics::LowThrustTwoBodyModel;
    using spacepdhcg::scvx::NativeLowThrustOuterConfig;
    using spacepdhcg::scvx::NativeLowThrustScvxDriver;
    using spacepdhcg::scvx::NativeLowThrustStatus;
    using spacepdhcg::transcription::LowThrustScvxConfig;
    using spacepdhcg::transcription::LowThrustSubproblem;

    const LowThrustTwoBodyModel model{};
    const LowThrustScvxConfig config{
        .intervals = 4U,
        .step_seconds = 10.0,
        .trust_radius = 1.0,
    };
    const LowThrustSubproblem subproblem(model, config);
    const LowThrustState initial{
        7'000.0,
        0.0,
        0.0,
        0.0,
        std::sqrt(model.config().gravitational_parameter / 7'000.0),
        0.0,
        500.0,
    };
    const LowThrustControl coast{0.0, 0.0, 0.0, 0.0};
    const std::vector<LowThrustControl> controls(config.intervals, coast);
    const auto states = model.rollout(initial, controls, config.step_seconds, false);
    const auto primal = subproblem.reference_decision(states, controls);
    const auto counters = std::make_shared<BackendCounters>();
    const auto factory = [primal, counters](spacepdhcg::core::FixedCQP problem) {
        return std::make_unique<EchoPersistentBackend>(
            std::move(problem),
            primal,
            counters
        );
    };

    NativeLowThrustOuterConfig outer{};
    outer.maximum_iterations = 2U;
    outer.minimum_iterations = 2U;
    NativeLowThrustScvxDriver driver(subproblem, factory, outer);
    const auto result = driver.solve(
        initial,
        states.back(),
        std::make_pair(states, controls)
    );

    if (result.status != NativeLowThrustStatus::maximum_iterations
        || result.iterations.size() != 2U || result.accepted_iterations != 1U
        || result.backend_creations != 1U || result.backend_updates != 1U) {
        return 1;
    }
    if (counters->creations != 1U || counters->updates != 1U
        || counters->warm_starts != 1U || counters->solves != 2U) {
        return 2;
    }
    if (!result.iterations.front().accepted
        || !result.iterations.front().restoration_accepted
        || result.iterations.back().accepted) {
        return 3;
    }
    if (result.residual.maximum() > 1.0e-12
        || result.path_diagnostics.maximum_violation() > 1.0e-10) {
        return 4;
    }

    const auto decoded = spacepdhcg::scvx::decode_low_thrust_decision(
        subproblem,
        primal
    );
    if (decoded.states.size() != states.size()
        || decoded.controls.size() != controls.size()
        || decoded.virtual_controls.size() != controls.size()) {
        return 5;
    }
    return 0;
}
