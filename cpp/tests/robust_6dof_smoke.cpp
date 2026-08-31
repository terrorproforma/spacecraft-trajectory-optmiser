#include "spacepdhcg/distributed/risk.hpp"
#include "spacepdhcg/distributed/scenario_cqp.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using spacepdhcg::distributed::ScenarioCqpBundle;
    using spacepdhcg::distributed::ScenarioTree;
    using spacepdhcg::distributed::aggregate_scenario_risk;
    using spacepdhcg::dynamics::PoweredDescent6DofControl;
    using spacepdhcg::dynamics::PoweredDescent6DofModel;
    using spacepdhcg::dynamics::PoweredDescent6DofState;
    using spacepdhcg::transcription::PoweredDescent6DofScvxConfig;
    using spacepdhcg::transcription::PoweredDescent6DofSubproblem;

    const PoweredDescent6DofModel model{};
    const PoweredDescent6DofScvxConfig config{
        .intervals = 2U,
        .step_seconds = 0.5,
        .trust_radius = 1.0,
    };
    const PoweredDescent6DofSubproblem local(model, config);
    const PoweredDescent6DofState initial{
        0.0,
        0.0,
        80.0,
        0.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2'000.0,
    };
    const PoweredDescent6DofControl control{
        0.0,
        0.0,
        7'500.0,
        0.0,
        0.0,
        0.0,
        7'500.0,
    };
    const std::vector<PoweredDescent6DofControl> controls(config.intervals, control);
    std::vector<PoweredDescent6DofState> states(config.intervals + 1U);
    states.front() = initial;
    for (std::size_t interval = 0; interval < config.intervals; ++interval) {
        states[interval + 1U] = model.euler_step(
            states[interval],
            controls[interval],
            config.step_seconds
        );
    }
    const auto local_values = local.values(states, controls, initial, states.back());
    const auto local_auxiliary = local.layout().variables()
                                 - local.layout().state_count()
                                 - local.layout().control_count();
    const auto tree = ScenarioTree::common_open_loop(3U, config.intervals);
    const ScenarioCqpBundle bundle(
        tree,
        local.structure(),
        14U,
        7U,
        local_auxiliary
    );
    const auto global_values = bundle.values({local_values, local_values, local_values});
    const auto global_problem = bundle.problem({local_values, local_values, local_values});
    if (global_problem.topology_fingerprint() == 0U
        || bundle.nonanticipativity_rows() != 42U) {
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
    const auto losses = bundle.local_objectives(
        decoded.local,
        {local_values, local_values, local_values}
    );
    const auto risk = aggregate_scenario_risk(losses, tree.probabilities(), 0.8);
    if (std::abs(risk.expected - losses.front()) > 1.0e-8
        || std::abs(risk.worst - losses.front()) > 1.0e-8
        || std::abs(risk.conditional_value_at_risk - losses.front()) > 1.0e-8) {
        return 3;
    }
    if (std::abs(bundle.global_objective(global_decision, global_values) - risk.expected)
        > 1.0e-8) {
        return 4;
    }
    return 0;
}
