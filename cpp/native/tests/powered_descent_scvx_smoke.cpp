#include "spacepdhcg/native/powered_descent_scvx.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace native = spacepdhcg::native;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct ScriptedCounters {
    std::size_t created{0};
    std::size_t updates{0};
    std::size_t solves{0};
};

class ReferenceWorkspace final : public native::CqpWorkspace {
  public:
    ReferenceWorkspace(
        native::OwnedCqp initial,
        std::vector<double> reference_primal,
        std::shared_ptr<ScriptedCounters> counters
    )
        : current_(std::move(initial)),
          reference_primal_(std::move(reference_primal)),
          counters_(std::move(counters)) {
        ++counters_->created;
    }

    [[nodiscard]] std::string backend_name() const override {
        return "scripted-reference";
    }

    [[nodiscard]] bool persistent() const noexcept override { return true; }

    void update(const native::OwnedCqp& problem) override {
        current_ = problem;
        ++counters_->updates;
    }

    void warm_start(std::span<const double>, std::span<const double>) override {}
    void clear_warm_start() override {}

    [[nodiscard]] native::CqpSolveResult solve(
        const native::CqpSolveOptions& options
    ) override {
        options.validate();
        ++counters_->solves;
        const auto dual_size = static_cast<std::size_t>(
            current_.scalar_constraint.rows + current_.affine_cone.rows
        );
        return native::CqpSolveResult{
            native::CqpSolveStatus::solved,
            reference_primal_,
            std::vector<double>(dual_size, 0.0),
            current_.objective(reference_primal_),
            0.0,
            0.0,
            1,
            0.0,
            0.0,
            "exact scripted reference",
        };
    }

  private:
    native::OwnedCqp current_{};
    std::vector<double> reference_primal_{};
    std::shared_ptr<ScriptedCounters> counters_{};
};

std::vector<double> encode_reference(
    const native::PoweredDescentCqp& transcription,
    const native::PoweredDescentReference& reference
) {
    std::vector<double> decision(
        static_cast<std::size_t>(transcription.layout().variables()),
        0.0
    );
    for (native::Index node = 0; node <= transcription.config().intervals; ++node) {
        const auto offset = static_cast<std::size_t>(transcription.layout().state_offset(node));
        std::copy(
            reference.states[static_cast<std::size_t>(node)].begin(),
            reference.states[static_cast<std::size_t>(node)].end(),
            decision.begin() + static_cast<std::ptrdiff_t>(offset)
        );
    }
    for (native::Index interval = 0;
         interval < transcription.config().intervals;
         ++interval) {
        const auto offset = static_cast<std::size_t>(
            transcription.layout().control_offset(interval)
        );
        std::copy(
            reference.controls[static_cast<std::size_t>(interval)].begin(),
            reference.controls[static_cast<std::size_t>(interval)].end(),
            decision.begin() + static_cast<std::ptrdiff_t>(offset)
        );
    }
    return decision;
}

void test_full_native_outer_loop() {
    native::PoweredDescentCqpConfig cqp_config{};
    cqp_config.intervals = 8;
    cqp_config.step_seconds = 1.0;
    cqp_config.trust_radius = 1.0;
    native::PoweredDescentCqp transcription(
        native::PoweredDescent3DofModel{},
        cqp_config
    );

    native::PoweredDescentState initial{0.0, 0.0, 20.0, 0.0, 0.0, -1.0, 2'000.0};
    const std::array<double, 3> target_position{0.0, 0.0, 0.0};
    const std::array<double, 3> target_velocity{0.0, 0.0, 0.0};
    const auto reference = native::make_powered_descent_reference(
        transcription.model(),
        initial,
        target_position,
        target_velocity,
        static_cast<std::size_t>(cqp_config.intervals),
        cqp_config.step_seconds
    );
    const auto reference_primal = encode_reference(transcription, reference);
    auto counters = std::make_shared<ScriptedCounters>();

    native::PoweredDescentOuterConfig outer{};
    outer.maximum_iterations = 3;
    outer.minimum_iterations = 1;
    outer.convergence_tolerance = 2.0e-8;
    outer.step_tolerance = 2.0e-8;

    native::PoweredDescentScvxDriver driver(
        transcription,
        [reference_primal, counters](const native::OwnedCqp& initial_problem) {
            return std::make_unique<ReferenceWorkspace>(
                initial_problem,
                reference_primal,
                counters
            );
        },
        outer
    );
    const auto result = driver.solve(
        initial,
        target_position,
        target_velocity,
        &reference
    );

    require(result.converged(), "native SCvx driver did not recognise an exact reference");
    require(result.accepted_iterations == 1, "native SCvx accepted-iteration count is wrong");
    require(result.workspace_updates == 1 && result.workspace_solves == 1,
            "native SCvx persistent-session counts are wrong");
    require(counters->created == 1 && counters->updates == 1 && counters->solves == 1,
            "native SCvx reconstructed or skipped the workspace lifecycle");
    require(result.residual.maximum() < 2.0e-8,
            "native SCvx final residual is above tolerance");
    require(result.path.maximum_violation() < 2.0e-8,
            "native SCvx final trajectory violates path constraints");
    require(result.iterations.size() == 1 && result.iterations.front().accepted,
            "native SCvx iteration record is incomplete");
    require(result.iterations.front().restoration_accepted,
            "zero-step exact reference should be accepted as restoration from initial step metric");
}

}  // namespace

int main() {
    test_full_native_outer_loop();
    return 0;
}
