#include "spacepdhcg/distributed/scenario_cqp.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using spacepdhcg::distributed::ScenarioCqpBundle;
    using spacepdhcg::distributed::ScenarioTree;
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::transcription::PoweredDescent3DofSubproblem;
    using spacepdhcg::transcription::PoweredDescentScvxConfig;

    const PoweredDescent3DofModel model{};
    const PoweredDescentScvxConfig config{
        .intervals = 3U,
        .step_seconds = 1.0,
        .trust_radius = 1.0,
    };
    const PoweredDescent3DofSubproblem local(model, config);
    const PoweredDescentState initial{0.0, 0.0, 80.0, 0.0, 0.0, -2.0, 2'000.0};
    const PoweredDescentControl control{0.0, 0.0, 7'500.0, 7'500.0};
    const std::vector<PoweredDescentControl> controls(config.intervals, control);
    const auto states = model.rollout(initial, controls, config.step_seconds, false);
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
    const auto local_values = local.values(
        states,
        controls,
        initial,
        target_position,
        target_velocity
    );

    const auto tree = ScenarioTree::common_open_loop(2U, config.intervals);
    const auto local_auxiliary = local.layout().variables()
                                 - local.layout().state_count()
                                 - local.layout().control_count();
    const ScenarioCqpBundle bundle(
        tree,
        local.structure(),
        7U,
        4U,
        local_auxiliary
    );
    auto problem = bundle.problem({local_values, local_values});
    const auto fingerprint = problem.topology_fingerprint();
    if (bundle.structure().variables() != 176
        || bundle.structure().scalar_rows() != 182
        || bundle.structure().affine_rows() != 136
        || bundle.nonanticipativity_rows() != 24U) {
        return 1;
    }

    const auto local_decision = local.reference_decision(states, controls);
    std::vector<double> global_decision(
        static_cast<std::size_t>(bundle.structure().variables()),
        0.0
    );
    for (std::size_t scenario = 0; scenario < bundle.scenario_count(); ++scenario) {
        const auto [begin, end] = bundle.layout().scenario_range(scenario);
        static_cast<void>(end);
        std::copy(
            local_decision.begin(),
            local_decision.end(),
            global_decision.begin() + static_cast<std::ptrdiff_t>(begin)
        );
    }
    for (const auto& block : bundle.layout().consensus_blocks()) {
        const auto [begin, end] = block.range();
        static_cast<void>(end);
        std::copy(
            controls[block.node.stage].begin(),
            controls[block.node.stage].end(),
            global_decision.begin() + static_cast<std::ptrdiff_t>(begin)
        );
    }
    if (bundle.maximum_nonanticipativity_violation(global_decision) > 1.0e-12) {
        return 2;
    }

    const auto decoded = bundle.decode_primal(global_decision);
    const auto expected = bundle.expected_objective(
        decoded.local,
        {local_values, local_values}
    );
    const auto global = bundle.global_objective(global_decision, problem.values());
    if (std::abs(expected - global) > 1.0e-8) {
        return 3;
    }

    auto perturbed = local_values;
    perturbed.linear_objective.front() += 1.0e-4;
    problem.update_values(bundle.values({local_values, perturbed}));
    if (problem.update_count() != 1U || problem.topology_fingerprint() != fingerprint) {
        return 4;
    }
    return 0;
}
