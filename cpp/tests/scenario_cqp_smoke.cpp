#include "spacepdhcg/core/powered_descent_6dof_cqp.hpp"
#include "spacepdhcg/core/scenario_cqp.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    PoweredDescent6DOFCQPConfig config;
    config.intervals = 2U;
    config.step_seconds = 0.25;
    PoweredDescent6DOF model{PoweredDescent6DOFConfig{}};
    const PoweredDescent6DOFCQP local_problem{model, config};

    std::vector<PoweredDescent6DOFState> states(config.intervals + 1U);
    std::vector<PoweredDescent6DOFControl> controls(config.intervals);
    states.front() = PoweredDescent6DOFState{
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2'000.0,
    };
    for (std::size_t interval = 0; interval < config.intervals; ++interval) {
        const double hover = -states[interval][13] * model.config().gravity[2];
        controls[interval] = PoweredDescent6DOFControl{
            0.0,
            0.0,
            hover,
            0.0,
            0.0,
            0.0,
            hover,
        };
        const auto derivative = model.dynamics(states[interval], controls[interval]);
        states[interval + 1U] = states[interval];
        for (std::size_t component = 0;
             component < powered_descent_6dof_state_dimension;
             ++component) {
            states[interval + 1U][component] +=
                config.step_seconds * derivative[component];
        }
    }
    const auto local_values = local_problem.values(
        states,
        controls,
        states.front(),
        states.back()
    );
    const auto local_primal = local_problem.reference_decision(states, controls);

    const auto tree = ScenarioTree::common_open_loop(2U, config.intervals);
    const std::size_t state_count =
        (config.intervals + 1U) * powered_descent_6dof_state_dimension;
    const std::size_t control_count =
        config.intervals * powered_descent_6dof_control_dimension;
    ScenarioCQPBundle bundle{
        tree,
        local_problem.structure(),
        powered_descent_6dof_state_dimension,
        powered_descent_6dof_control_dimension,
        static_cast<std::size_t>(local_problem.structure().variables()) -
            state_count - control_count,
    };

    const std::vector<CQPValues> local_value_sets{local_values, local_values};
    const std::vector<std::vector<double>> local_primals{local_primal, local_primal};
    const auto global_values = bundle.values(local_value_sets);
    const auto global_primal = bundle.lift_primal(local_primals);

    if (bundle.maximum_nonanticipativity_violation(global_primal) > 1.0e-12 ||
        bundle.maximum_scalar_violation(global_primal, global_values) > 2.0e-8 ||
        bundle.maximum_affine_cone_violation(global_primal, global_values) > 2.0e-8) {
        return 1;
    }

    CscOperator quadratic{bundle.structure().quadratic(), global_values.quadratic};
    const auto product = quadratic.multiply(global_primal);
    double global_objective = 0.0;
    for (std::size_t index = 0; index < global_primal.size(); ++index) {
        global_objective += 0.5 * global_primal[index] * product[index] +
            global_values.linear_objective[index] * global_primal[index];
    }
    const double expected = bundle.expected_objective(local_primals, local_value_sets);
    if (std::abs(global_objective - expected) > 1.0e-8) {
        return 2;
    }

    const auto decoded = bundle.decode_primal(global_primal);
    if (decoded.local.size() != 2U ||
        decoded.consensus.size() != config.intervals ||
        decoded.local.front().size() != local_primal.size()) {
        return 3;
    }

    auto perturbed = local_values;
    perturbed.linear_objective.front() += 1.0e-6;
    const std::vector<CQPValues> updated_sets{local_values, perturbed};
    const auto updated = bundle.values(updated_sets);
    if (updated.quadratic.size() != global_values.quadratic.size() ||
        updated.scalar_constraint.size() != global_values.scalar_constraint.size() ||
        updated.affine_cone.size() != global_values.affine_cone.size() ||
        updated.linear_objective == global_values.linear_objective) {
        return 4;
    }
    return 0;
}
