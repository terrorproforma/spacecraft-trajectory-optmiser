#include "spacepdhcg/scvx/powered_descent_3dof_driver.hpp"

#include <array>
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
          current_(problem.values()),
          primal_(std::move(primal)),
          dual_(static_cast<std::size_t>(structure_.duals()), 0.0),
          counters_(std::move(counters)) {
        if (primal_.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("echo backend primal has the wrong size");
        }
        ++counters_->creations;
    }

    [[nodiscard]] const spacepdhcg::core::FixedStructure& structure() const noexcept override {
        return structure_;
    }

    [[nodiscard]] std::size_t update_count() const noexcept override { return update_count_; }

    void update(spacepdhcg::core::NumericValues values) override {
        values.validate(structure_);
        current_ = std::move(values);
        ++update_count_;
        ++counters_->updates;
    }

    void warm_start(const spacepdhcg::core::HostWarmStart& start) override {
        if (!start.primal.empty()
            && start.primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("echo backend primal warm start has the wrong size");
        }
        if (!start.dual.empty()
            && start.dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("echo backend dual warm start has the wrong size");
        }
        ++counters_->warm_starts;
    }

    [[nodiscard]] spacepdhcg::core::HostCqpSolution solve(
        double tolerance,
        std::size_t iteration_limit
    ) override {
        if (!std::isfinite(tolerance) || tolerance <= 0.0 || iteration_limit == 0U) {
            throw std::invalid_argument("echo backend solve request is invalid");
        }
        ++counters_->solves;
        spacepdhcg::core::HostCqpSolution solution{};
        solution.status = spacepdhcg::SolveStatus::optimal;
        solution.primal = primal_;
        solution.dual = dual_;
        solution.objective = 0.0;
        solution.primal_residual = 1.0e-10;
        solution.dual_residual = 1.0e-10;
        solution.outer_iterations = 1U;
        solution.inner_iterations = 3U;
        return solution;
    }

  private:
    spacepdhcg::core::FixedStructure structure_{};
    spacepdhcg::core::NumericValues current_{};
    std::vector<double> primal_{};
    std::vector<double> dual_{};
    std::shared_ptr<BackendCounters> counters_{};
    std::size_t update_count_{0U};
};

}  // namespace

int main() {
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::scvx::NativePoweredDescentOuterConfig;
    using spacepdhcg::scvx::NativePoweredDescentScvxDriver;
    using spacepdhcg::scvx::NativeScvxStatus;
    using spacepdhcg::transcription::PoweredDescent3DofSubproblem;
    using spacepdhcg::transcription::PoweredDescentScvxConfig;

    const PoweredDescent3DofModel model{};
    const PoweredDescentScvxConfig transcription_config{
        .intervals = 4U,
        .step_seconds = 1.0,
        .trust_radius = 1.0,
    };
    const PoweredDescent3DofSubproblem subproblem(model, transcription_config);
    const PoweredDescentState initial{0.0, 0.0, 100.0, 0.0, 0.0, -1.0, 2'000.0};
    const PoweredDescentControl control{0.0, 0.0, 7'500.0, 7'500.0};
    const std::vector<PoweredDescentControl> controls(transcription_config.intervals, control);
    const auto states = model.rollout(initial, controls, transcription_config.step_seconds, false);
    const std::array<double, 3U> target_position{
        states.back()[0U],
        states.back()[1U],
        states.back()[2U],
    };
    const std::array<double, 3U> target_velocity{
        states.back()[3U],
        states.back()[4U],
        states.back()[5U],
    };
    const auto echo_primal = subproblem.reference_decision(states, controls);
    const auto counters = std::make_shared<BackendCounters>();
    const auto factory = [echo_primal, counters](spacepdhcg::core::FixedCQP problem) {
        return std::make_unique<EchoPersistentBackend>(
            std::move(problem),
            echo_primal,
            counters
        );
    };

    NativePoweredDescentOuterConfig outer{};
    outer.maximum_iterations = 2U;
    outer.minimum_iterations = 2U;
    NativePoweredDescentScvxDriver driver(subproblem, factory, outer);
    const auto result = driver.solve(
        initial,
        target_position,
        target_velocity,
        std::make_pair(states, controls)
    );

    if (result.status != NativeScvxStatus::maximum_iterations
        || result.iterations.size() != 2U || result.accepted_iterations != 1U
        || result.backend_creations != 1U || result.backend_updates != 1U) {
        return 1;
    }
    if (counters->creations != 1U || counters->updates != 1U
        || counters->warm_starts != 1U || counters->solves != 2U) {
        return 2;
    }
    if (!result.iterations.front().accepted
        || result.iterations.front().restoration_accepted
               != result.iterations.front().accepted
        || result.iterations.back().accepted) {
        return 3;
    }
    if (result.residual.maximum() > 1.0e-12
        || result.path_diagnostics.maximum_violation() > 1.0e-9) {
        return 4;
    }
    return 0;
}
