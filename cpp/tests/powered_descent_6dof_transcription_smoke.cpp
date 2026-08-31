#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::dynamics::PoweredDescent6DofControl;
    using spacepdhcg::dynamics::PoweredDescent6DofModel;
    using spacepdhcg::dynamics::PoweredDescent6DofState;
    using spacepdhcg::transcription::PoweredDescent6DofScvxConfig;
    using spacepdhcg::transcription::PoweredDescent6DofSubproblem;

    const PoweredDescent6DofModel model{};
    const PoweredDescent6DofScvxConfig config{
        .intervals = 3U,
        .step_seconds = 0.5,
        .trust_radius = 1.0,
    };
    const PoweredDescent6DofSubproblem subproblem(model, config);
    if (subproblem.layout().variables() != 161U
        || subproblem.layout().scalar_rows() != 160U
        || subproblem.layout().affine_rows() != 133U
        || subproblem.structure().quadratic.nonzeros() != 161U
        || subproblem.structure().scalar_constraint.nonzeros() != 1'183U
        || subproblem.structure().affine_cone->nonzeros() != 122U
        || subproblem.structure().affine_cones.size() != 18U) {
        return 1;
    }

    const PoweredDescent6DofState initial{
        0.0,
        0.0,
        100.0,
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
    auto problem = subproblem.problem(states, controls, initial, states.back());
    const auto fingerprint = problem.topology_fingerprint();
    const auto decision = subproblem.reference_decision(states, controls);
    const auto diagnostics = subproblem.diagnostics(decision, problem.values());
    if (diagnostics.maximum_violation() > 1.0e-9
        || diagnostics.linearised_dynamics_defect_inf > 1.0e-9
        || diagnostics.terminal_error_inf > 1.0e-9
        || diagnostics.quaternion_linearisation_error_inf > 1.0e-9
        || diagnostics.virtual_control_inf > 1.0e-12) {
        return 2;
    }

    problem.update_values(
        subproblem.values(states, controls, initial, states.back(), 0.5)
    );
    if (problem.update_count() != 1U || problem.topology_fingerprint() != fingerprint) {
        return 3;
    }
    return 0;
}
