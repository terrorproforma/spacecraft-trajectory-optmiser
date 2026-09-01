#include "spacepdhcg/scvx/low_thrust_benchmark.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
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

std::vector<double> exact_transfer_decision(
    const spacepdhcg::transcription::LowThrustSubproblem& subproblem,
    const spacepdhcg::core::NumericValues& values,
    const std::vector<spacepdhcg::dynamics::LowThrustState>& states,
    const std::vector<spacepdhcg::dynamics::LowThrustControl>& controls
) {
    auto decision = subproblem.reference_decision(states, controls);
    const auto& pattern = subproblem.structure().scalar_constraint;
    const auto& layout = subproblem.layout();
    for (std::size_t interval = 0U; interval < layout.intervals; ++interval) {
        for (std::size_t component = 0U; component < 7U; ++component) {
            const auto row = layout.dynamics_rows().start + 7U * interval + component;
            double left_hand_side{0.0};
            for (spacepdhcg::Index column = 0; column < pattern.columns; ++column) {
                const auto begin =
                    static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
                const auto end = static_cast<std::size_t>(
                    pattern.offsets[static_cast<std::size_t>(column) + 1U]
                );
                for (auto slot = begin; slot < end; ++slot) {
                    if (static_cast<std::size_t>(pattern.indices[slot]) == row) {
                        left_hand_side +=
                            values.scalar_constraint[slot]
                            * decision[static_cast<std::size_t>(column)];
                    }
                }
            }
            const auto virtual_variable =
                layout.virtual_control(interval).start + component;
            const auto virtual_value = left_hand_side - values.scalar_lower[row];
            decision[virtual_variable] = virtual_value;
            decision[layout.epigraph(interval).start + component] =
                std::abs(virtual_value);
        }
    }
    return decision;
}

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

    constexpr std::array<std::string_view, 4U> frozen_classes{
        "coast_reference",
        "radius_raise",
        "plane_change",
        "combined",
    };
    for (const auto name : frozen_classes) {
        const auto transfer_class =
            spacepdhcg::scvx::low_thrust_transfer_class(name);
        if (spacepdhcg::scvx::low_thrust_transfer_class_name(transfer_class) != name) {
            return 6;
        }
        const auto transfer = spacepdhcg::scvx::make_low_thrust_transfer_target(
            model,
            initial,
            config.intervals,
            config.step_seconds,
            transfer_class
        );
        if (transfer.first.size() != config.intervals + 1U
            || transfer.second.size() != config.intervals
            || model.path_diagnostics(transfer.first, transfer.second).maximum_violation()
                   > 1.0e-12) {
            return 7;
        }
    }

    const auto transfer = spacepdhcg::scvx::make_low_thrust_transfer_target(
        model,
        initial,
        config.intervals,
        config.step_seconds,
        spacepdhcg::scvx::LowThrustTransferClass::radius_raise
    );
    auto transfer_config = config;
    transfer_config.trust_radius = 0.25;
    transfer_config.discretisation =
        spacepdhcg::transcription::DiscretisationMethod::rk4_variational;
    const LowThrustSubproblem transfer_subproblem(model, transfer_config);
    const auto transfer_values = transfer_subproblem.values(
        states,
        controls,
        initial,
        transfer.first.back(),
        transfer_config.trust_radius
    );
    const auto transfer_primal = exact_transfer_decision(
        transfer_subproblem,
        transfer_values,
        transfer.first,
        transfer.second
    );
    const auto transfer_diagnostics =
        transfer_subproblem.diagnostics(transfer_primal, transfer_values);
    if (transfer_diagnostics.maximum_violation() > 1.0e-9) {
        return 8;
    }
    const auto transfer_counters = std::make_shared<BackendCounters>();
    const auto transfer_factory =
        [transfer_primal, transfer_counters](spacepdhcg::core::FixedCQP problem) {
            return std::make_unique<EchoPersistentBackend>(
                std::move(problem),
                transfer_primal,
                transfer_counters
            );
        };
    NativeLowThrustOuterConfig transfer_outer{};
    transfer_outer.maximum_iterations = 2U;
    transfer_outer.minimum_iterations = 2U;
    NativeLowThrustScvxDriver transfer_driver(
        transfer_subproblem,
        transfer_factory,
        transfer_outer
    );
    const auto transfer_result = transfer_driver.solve(
        initial,
        transfer.first.back(),
        std::make_pair(states, controls)
    );
    if (transfer_result.accepted_iterations == 0U) {
        return 9;
    }
    if (transfer_result.iterations.empty()
        || !transfer_result.iterations.front().accepted) {
        return 10;
    }
    if (transfer_result.iterations.front().step_fraction <= 0.0) {
        return 11;
    }
    double initial_terminal_error{0.0};
    for (std::size_t component = 0U; component < 6U; ++component) {
        initial_terminal_error = std::max(
            initial_terminal_error,
            std::abs(
                (states.back()[component] - transfer.first.back()[component])
                * transfer_config.state_trust_scales[component]
            )
        );
    }
    if (transfer_result.iterations.front().residual.terminal
        >= initial_terminal_error) {
        return 12;
    }

    auto violating_states = states;
    auto violating_controls = controls;
    violating_controls.front() = {2.0, 0.0, 0.0, 0.5};
    violating_states[1U][6U] = model.config().minimum_mass - 1.0;
    violating_states[2U][0U] = model.config().minimum_radius - 1.0;
    violating_states[2U][1U] = 0.0;
    violating_states[2U][2U] = 0.0;
    const auto path = model.path_diagnostics(violating_states, violating_controls);
    if (!(path.thrust_epigraph > 0.0)
        || !(path.minimum_mass > 0.0)
        || !(path.minimum_radius > 0.0)) {
        return 13;
    }
    violating_controls.front() = {0.0, 0.0, 0.0, 2.0};
    if (!(model.path_diagnostics(violating_states, violating_controls).throttle_upper
          > 0.0)) {
        return 14;
    }
    std::printf(
        "{\"case\":\"p1e_displaced_acceptance\","
        "\"transfer\":\"radius_raise\",\"trust_radius\":0.25,"
        "\"accepted_steps\":%zu,\"first_step_fraction\":%.17g,"
        "\"terminal_before\":%.17g,\"terminal_after\":%.17g,"
        "\"path_injections_detected\":true}\n",
        transfer_result.accepted_iterations,
        transfer_result.iterations.front().step_fraction,
        initial_terminal_error,
        transfer_result.residual.terminal
    );
    return 0;
}
