#include "spacepdhcg/transcription/powered_descent_3dof.hpp"

#include <array>
#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::transcription::PoweredDescent3DofSubproblem;
    using spacepdhcg::transcription::PoweredDescentScvxConfig;

    const PoweredDescent3DofModel model{};
    const PoweredDescentScvxConfig config{
        .intervals = 4U,
        .step_seconds = 1.0,
        .trust_radius = 1.0,
    };
    const PoweredDescent3DofSubproblem subproblem(model, config);
    if (subproblem.layout().variables() != 107U
        || subproblem.layout().scalar_rows() != 101U
        || subproblem.layout().affine_rows() != 87U
        || subproblem.structure().affine_cones.size() != 14U) {
        return 1;
    }

    const PoweredDescentState initial{0.0, 0.0, 100.0, 0.0, 0.0, -1.0, 2'000.0};
    const PoweredDescentControl control{0.0, 0.0, 7'500.0, 7'500.0};
    const std::vector<PoweredDescentControl> controls(config.intervals, control);
    const auto states = model.rollout(initial, controls, config.step_seconds, false);
    const std::array<double, 3U> target_position{
        states.back()[0],
        states.back()[1],
        states.back()[2],
    };
    const std::array<double, 3U> target_velocity{
        states.back()[3],
        states.back()[4],
        states.back()[5],
    };
    auto problem = subproblem.problem(
        states,
        controls,
        initial,
        target_position,
        target_velocity
    );
    const auto fingerprint = problem.topology_fingerprint();
    const auto decision = subproblem.reference_decision(states, controls);
    const auto diagnostics = subproblem.diagnostics(decision, problem.values());
    if (diagnostics.maximum_violation() > 1.0e-9
        || diagnostics.linearised_dynamics_defect_inf > 1.0e-9
        || diagnostics.terminal_error_inf > 1.0e-9
        || diagnostics.virtual_control_inf > 1.0e-12) {
        return 2;
    }

    auto updated_values = subproblem.values(
        states,
        controls,
        initial,
        target_position,
        target_velocity,
        0.5
    );
    problem.update_values(std::move(updated_values));
    if (problem.update_count() != 1U || problem.topology_fingerprint() != fingerprint) {
        return 3;
    }
    return 0;
}
