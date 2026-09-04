#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <cmath>
#include <vector>

namespace {

bool equal_values(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    return left == right;
}

bool different_values(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    return !equal_values(left, right);
}

}  // namespace

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
    const auto nominal_values =
        subproblem.values(states, controls, initial, states.back());
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
    auto displaced_initial = initial;
    displaced_initial[6U] = std::cos(0.025);
    displaced_initial[7U] = std::sin(0.025);
    displaced_initial[8U] = 0.0;
    displaced_initial[9U] = 0.0;
    displaced_initial[10U] = 0.05;
    const auto displaced_states =
        model.rollout(displaced_initial, controls, config.step_seconds);
    const auto displaced_values = subproblem.values(
        displaced_states,
        controls,
        displaced_initial,
        states.back(),
        0.37
    );
    const auto coefficient = [&subproblem, &displaced_values](
        const std::size_t row,
        const std::size_t column
    ) {
        const auto& pattern = subproblem.structure().scalar_constraint;
        for (auto index = pattern.offsets[column];
             index < pattern.offsets[column + 1U];
             ++index) {
            if (pattern.indices[static_cast<std::size_t>(index)]
                == static_cast<int>(row)) {
                return displaced_values.scalar_constraint[
                    static_cast<std::size_t>(index)
                ];
            }
        }
        return 0.0;
    };
    const auto quaternion_rows = subproblem.layout().quaternion_rows();
    const auto terminal_quaternion_row =
        quaternion_rows.start + config.intervals;
    const auto terminal_state = subproblem.layout().state(config.intervals);
    for (std::size_t component = 0U; component < 4U; ++component) {
        if (coefficient(
                terminal_quaternion_row,
                terminal_state.start + 6U + component
            ) != 0.0) {
            return 4;
        }
    }
    if (displaced_values.scalar_lower[terminal_quaternion_row] != 0.0
        || displaced_values.scalar_upper[terminal_quaternion_row] != 0.0) {
        return 4;
    }
    const auto interior_row = quaternion_rows.start;
    const auto interior_state = subproblem.layout().state(0U);
    double interior_coefficient_maximum = 0.0;
    for (std::size_t component = 0U; component < 4U; ++component) {
        interior_coefficient_maximum = std::max(
            interior_coefficient_maximum,
            std::abs(coefficient(
                interior_row,
                interior_state.start + 6U + component
            ))
        );
    }
    if (interior_coefficient_maximum <= 0.0
        || displaced_values.scalar_lower[interior_row] <= 0.0) {
        return 4;
    }
    if (!equal_values(nominal_values.quadratic, displaced_values.quadratic)
        || !equal_values(nominal_values.affine_cone, displaced_values.affine_cone)
        || !equal_values(
            nominal_values.variable_lower,
            displaced_values.variable_lower
        )
        || !equal_values(
            nominal_values.variable_upper,
            displaced_values.variable_upper
        )
        || !different_values(
            nominal_values.scalar_constraint,
            displaced_values.scalar_constraint
        )
        || !different_values(
            nominal_values.linear_objective,
            displaced_values.linear_objective
        )
        || !different_values(
            nominal_values.scalar_lower,
            displaced_values.scalar_lower
        )
        || !different_values(
            nominal_values.scalar_upper,
            displaced_values.scalar_upper
        )
        || !different_values(
            nominal_values.affine_offset,
            displaced_values.affine_offset
        )) {
        return 4;
    }
    auto mass_only_target = states.back();
    mass_only_target[13U] -= 100.0;
    const auto mass_target_values = subproblem.values(
        states,
        controls,
        initial,
        mass_only_target
    );
    if (subproblem.layout().terminal_rows().size != 13U
        || !equal_values(
            nominal_values.scalar_lower,
            mass_target_values.scalar_lower
        )
        || !equal_values(
            nominal_values.scalar_upper,
            mass_target_values.scalar_upper
        )) {
        return 5;
    }
    auto penalty_config = config;
    penalty_config.virtual_l1_weight *= 2.0;
    const PoweredDescent6DofSubproblem penalty_subproblem(model, penalty_config);
    const auto penalty_values = penalty_subproblem.values(
        states,
        controls,
        initial,
        states.back()
    );
    if (penalty_subproblem.structure().fingerprint()
            != subproblem.structure().fingerprint()
        || !equal_values(nominal_values.quadratic, penalty_values.quadratic)
        || !different_values(
            nominal_values.linear_objective,
            penalty_values.linear_objective
        )) {
        return 6;
    }
    return 0;
}
